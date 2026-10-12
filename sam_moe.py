"""SAM ViT + sparse ASPP + coarse/fine spatial MoE semantic segmentation."""
from functools import partial
from pathlib import Path
import math
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .modeling.image_encoder import ImageEncoderViT, Attention, get_rel_pos
    from .sparse_modules import ConvBlock, EnhancedASPP, MixtureOfExperts, SqueezeExcitation, topk_attention
except ImportError:
    from modeling.image_encoder import ImageEncoderViT, Attention, get_rel_pos
    from sparse_modules import ConvBlock, EnhancedASPP, MixtureOfExperts, SqueezeExcitation, topk_attention


class SparseSAMAttention(Attention):
    """Retain upstream qkv/proj/relative-position parameter names for weights."""
    def __init__(self, original, ratio=0.1, query_chunk_size=128):
        nn.Module.__init__(self)
        self.qkv, self.proj = original.qkv, original.proj
        self.num_heads, self.scale = original.num_heads, original.scale
        self.use_rel_pos = original.use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h, self.rel_pos_w = original.rel_pos_h, original.rel_pos_w
        self.sparsity_ratio, self.query_chunk_size = ratio, query_chunk_size
        self.temperature = 1.0

    def forward(self, x):
        b, h, w, c = x.shape
        q, k, v = self.qkv(x).reshape(b, h * w, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        bias_fn = None
        if self.use_rel_pos:
            rh, rw = get_rel_pos(h, h, self.rel_pos_h), get_rel_pos(w, w, self.rel_pos_w)
            def bias_fn(scores, query, start):
                positions = torch.arange(start, start + query.shape[-2], device=x.device)
                rel_h = torch.einsum("bhqc,qkc->bhqk", query.float(), rh[positions // w].float())
                rel_w = torch.einsum("bhqc,qkc->bhqk", query.float(), rw[positions % w].float())
                bias = rel_h.unsqueeze(-1) + rel_w.unsqueeze(-2)
                return scores + bias.flatten(-2)
        output, self.sparse_loss, self.attention_entropy, _ = topk_attention(
            q, k, v, self.sparsity_ratio, self.training, self.temperature,
            self.query_chunk_size, bias_fn=bias_fn)
        return self.proj(output.transpose(1, 2).reshape(b, h, w, c))


ENCODERS = {"vit_b": (768, 12, 12, (2, 5, 8, 11)),
            "vit_l": (1024, 24, 16, (5, 11, 17, 23)),
            "vit_h": (1280, 32, 16, (7, 15, 23, 31))}


@torch.no_grad()
def predicted_signed_distance(confidence, threshold=.5):
    """Euclidean SDF from coarse predictions only, never from target masks.

    Inside is positive; distances are normalized by image diagonal. Empty/full
    predictions return -1/+1. CPU OpenCV EDT is explicit, nondifferentiable and
    incurs a device transfer; the paper leaves its inference source undefined.
    """
    if confidence.ndim != 4 or confidence.shape[1] != 1:
        raise ValueError("Expected Bx1xHxW coarse confidence")
    binary = (confidence.detach().float().cpu().numpy()[:, 0] >= threshold).astype(np.uint8)
    maps = []
    diagonal = math.hypot(*confidence.shape[-2:])
    for mask in binary:
        if not mask.any():
            distance = np.full(mask.shape, -1., dtype=np.float32)
        elif mask.all():
            distance = np.ones(mask.shape, dtype=np.float32)
        else:
            inside = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            distance = (inside - outside) / diagonal
        maps.append(distance)
    return torch.from_numpy(np.stack(maps)[:, None]).to(device=confidence.device, dtype=confidence.dtype)


class DualStageDecoder(nn.Module):
    def __init__(self, encoder_channels, low_channels, channels=256, sparsity_ratio=0.1,
                 query_chunk_size=128, lateral_channels=None, gate_hidden_dim=512,
                 gate_activation="relu", stage1_temperature=.5, stage2_temperature=.3,
                 balance_mode="frequency_st", boundary_mode="predicted_sdf"):
        super().__init__()
        if boundary_mode not in {"predicted_sdf", "none"}:
            raise ValueError("Unknown boundary mode")
        self.boundary_mode = boundary_mode
        self.aspp = EnhancedASPP(encoder_channels, channels, sparsity_ratio,
                                 query_chunk_size=query_chunk_size)
        gate_options = dict(gate_hidden_dim=gate_hidden_dim, gate_activation=gate_activation,
                            balance_mode=balance_mode)
        self.stage1 = MixtureOfExperts(channels, channels, 4, 2, temperature=stage1_temperature, **gate_options)
        self.coarse_head = nn.Conv2d(channels, 1, 1)
        self.lateral_projections = nn.ModuleList(ConvBlock(c, channels, 1)
            for c in (lateral_channels if lateral_channels is not None else [low_channels]))
        self.lateral_norm = nn.GroupNorm(math.gcd(channels, 8), channels)
        self.low_projection = ConvBlock(low_channels, channels, 1)
        self.fusion_gate = nn.Conv2d(channels * 2, channels, 1)
        self.fusion = ConvBlock(channels * 2 + (2 if boundary_mode == "predicted_sdf" else 1), channels, 1)
        self.refinement_se = SqueezeExcitation(channels)
        self.stage2 = MixtureOfExperts(channels, channels, 3, 2, refinement=True,
                                       temperature=stage2_temperature, **gate_options)
        self.output_head = nn.Conv2d(channels, 1, 1)
        from torchvision.ops import DeformConv2d
        self.offset_head = nn.Conv2d(channels, 18, 3, padding=1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        self.refinement_residual = DeformConv2d(channels, 1, 3, padding=1)

    def forward(self, high, low, input_size):
        intermediate = list(low) if isinstance(low, (list, tuple)) else [low]
        if len(intermediate) != len(self.lateral_projections):
            raise ValueError("Intermediate features must match lateral projection count")
        low = intermediate[0]
        coarse_size = tuple(max(1, (s + 3) // 4) for s in input_size)
        fine_size = tuple(2 * s for s in coarse_size)
        high = self.aspp(high)
        coarse_input = F.interpolate(high, coarse_size, mode="bilinear", align_corners=False)
        for projection, feature in zip(self.lateral_projections, intermediate):
            coarse_input = coarse_input + F.interpolate(projection(feature), coarse_size,
                                                        mode="bilinear", align_corners=False)
        coarse = self.stage1(F.relu(self.lateral_norm(coarse_input)))
        coarse_logits = self.coarse_head(coarse)
        coarse = F.interpolate(coarse, fine_size, mode="bilinear", align_corners=False)
        low = F.interpolate(self.low_projection(low), fine_size, mode="bilinear", align_corners=False)
        gate = self.fusion_gate(torch.cat([low, coarse], dim=1)).sigmoid()
        fused = gate * low + (1 - gate) * coarse
        confidence = F.interpolate(coarse_logits, fine_size, mode="bilinear", align_corners=False).sigmoid()
        refinement_inputs = [fused, coarse, confidence]
        if self.boundary_mode == "predicted_sdf":
            refinement_inputs.append(predicted_signed_distance(confidence))
        fine = self.stage2(self.refinement_se(self.fusion(torch.cat(refinement_inputs, dim=1))))
        residual = self.refinement_residual(fine, self.offset_head(fine))
        logits = F.interpolate(self.output_head(fine) + residual, input_size, mode="bilinear", align_corners=False)
        return logits, coarse_logits


class SparseMoESAM(nn.Module):
    """Inputs are ImageNet-normalized RGB; outputs are logits at input H/W.

    SAM uses the same channel mean/std on the 0..255 scale. The default encoder
    keeps its pretrained 1024 grid; dataset images may be 256. The encoder
    input is resized here so ASPP's dilations 6/12/18 remain spatially useful.
    """
    def __init__(self, model_type="vit_b", sam_checkpoint=None, encoder_size=1024,
                 channels=256, sparsity_ratio=0.1, query_chunk_size=128,
                 encoder_config=None, gate_hidden_dim=512, gate_activation="relu",
                 stage1_temperature=.5, stage2_temperature=.3,
                 balance_mode="frequency_st", boundary_mode="predicted_sdf"):
        super().__init__()
        if encoder_size < 304 or encoder_size % 16:
            raise ValueError("encoder_size must be a multiple of 16 and >=304")
        if channels % 8:
            raise ValueError("decoder channels must be divisible by 8")
        if model_type not in ENCODERS:
            raise ValueError(f"Unsupported model_type: {model_type}")
        dim, depth, heads, global_indexes = ENCODERS[model_type]
        config = dict(embed_dim=dim, depth=depth, num_heads=heads, global_attn_indexes=global_indexes)
        if encoder_config:
            config.update(encoder_config)  # Small encoder for CPU structural tests only.
        self.image_encoder = ImageEncoderViT(img_size=encoder_size, patch_size=16,
            out_chans=256, qkv_bias=True, use_rel_pos=True, window_size=14,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **config)
        self.encoder_size = encoder_size
        self.lateral_indexes = tuple(sorted({max(0, math.ceil(config["depth"] * fraction) - 1)
                                             for fraction in (.25, .5, .75)}))
        self.pretrained_loaded = False
        if sam_checkpoint:
            self.load_sam_encoder(sam_checkpoint)
        for block in self.image_encoder.blocks:
            block.attn = SparseSAMAttention(block.attn, sparsity_ratio, query_chunk_size)
        self.decoder = DualStageDecoder(256, config["embed_dim"], channels,
            sparsity_ratio, query_chunk_size, [config["embed_dim"]] * len(self.lateral_indexes),
            gate_hidden_dim, gate_activation, stage1_temperature, stage2_temperature, balance_mode, boundary_mode)

    def load_sam_encoder(self, checkpoint):
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if "model" in state:
            state = state["model"]
        encoder = {key[len("image_encoder."):]: value for key, value in state.items()
                   if key.startswith("image_encoder.")}
        if not encoder:
            encoder = state
        # Position tables are explicitly resized into configured shapes so
        # subsequent semantic checkpoints can be strictly restored.
        if "pos_embed" in encoder and encoder["pos_embed"].shape != self.image_encoder.pos_embed.shape:
            encoder["pos_embed"] = F.interpolate(encoder["pos_embed"].permute(0, 3, 1, 2),
                self.image_encoder.pos_embed.shape[1:3], mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        for name, parameter in list(self.image_encoder.named_parameters()):
            if "rel_pos_" in name and name in encoder and parameter.shape != encoder[name].shape:
                # Keep configured table shapes so fine-tuned checkpoints can
                # be resumed without reconstructing the original SAM model.
                encoder[name] = F.interpolate(encoder[name].T.unsqueeze(0),
                    size=parameter.shape[0], mode="linear", align_corners=False).squeeze(0).T
        self.image_encoder.load_state_dict(encoder, strict=True)
        self.pretrained_loaded = True

    def set_encoder_trainable(self, trainable):
        self.image_encoder.requires_grad_(trainable)
        self.image_encoder.train(self.training and trainable)

    def forward(self, x, return_aux=False):
        input_size = x.shape[-2:]
        resized = F.interpolate(x, (self.encoder_size, self.encoder_size), mode="bilinear", align_corners=False)
        frozen = not any(p.requires_grad for p in self.image_encoder.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            tokens = self.image_encoder.patch_embed(resized)
            if self.image_encoder.pos_embed is not None:
                tokens = tokens + self.image_encoder.pos_embed
            intermediate = []
            for index, block in enumerate(self.image_encoder.blocks):
                tokens = block(tokens)
                if index in self.lateral_indexes:
                    intermediate.append(tokens.permute(0, 3, 1, 2))
            high = self.image_encoder.neck(tokens.permute(0, 3, 1, 2))
        logits, coarse_logits = self.decoder(high, intermediate, input_size)
        if not return_aux:
            return logits
        attention = [block.attn for block in self.image_encoder.blocks] + [self.decoder.aspp.sparse_attention]
        aux = {"balance": (self.decoder.stage1.balance_loss + self.decoder.stage2.balance_loss) / 2,
               "sparse": torch.stack([layer.sparse_loss for layer in attention]).mean(),
               "attention_entropy": torch.stack([layer.attention_entropy for layer in attention]).mean(),
               "gate_entropy": (self.decoder.stage1.gate_entropy + self.decoder.stage2.gate_entropy) / 2,
               "usage_stage1": self.decoder.stage1.last_usage,
               "usage_stage2": self.decoder.stage2.last_usage,
               "coarse_logits": coarse_logits}
        return logits, aux

    @torch.no_grad()
    def predict(self, x):
        return self(x).sigmoid()

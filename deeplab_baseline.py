"""ResNet50 ablation, separate from the SAM training path."""
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
try:
    from .sam_moe import DualStageDecoder
except ImportError:
    from sam_moe import DualStageDecoder


class EnhancedDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, backbone="resnet50", pretrained=False, channels=256,
                 **decoder_options):
        super().__init__()
        if num_classes != 1 or backbone != "resnet50":
            raise ValueError("This ablation supports binary segmentation with ResNet50")
        self.image_encoder = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None,
                                      replace_stride_with_dilation=[False, True, True])
        self.image_encoder.fc = nn.Identity()
        self.decoder = DualStageDecoder(2048, 256, channels, **decoder_options)

    def set_encoder_trainable(self, trainable):
        self.image_encoder.requires_grad_(trainable)
        self.image_encoder.train(self.training and trainable)

    def forward(self, x, return_aux=False):
        size, enc = x.shape[-2:], self.image_encoder
        x = enc.maxpool(enc.relu(enc.bn1(enc.conv1(x))))
        low = enc.layer1(x)
        high = enc.layer4(enc.layer3(enc.layer2(low)))
        logits, coarse = self.decoder(high, low, size)
        if not return_aux:
            return logits
        stages, attn = [self.decoder.stage1, self.decoder.stage2], self.decoder.aspp.sparse_attention
        return logits, {"balance": torch.stack([s.balance_loss for s in stages]).mean(),
                        "sparse": attn.sparse_loss, "attention_entropy": attn.attention_entropy,
                        "gate_entropy": torch.stack([s.gate_entropy for s in stages]).mean(),
                        "usage_stage1": stages[0].last_usage, "usage_stage2": stages[1].last_usage,
                        "coarse_logits": coarse}

"""Sparse attention and spatially dispatched convolution experts.

Exact global top-k still scores every key: QK is O(N^2 D). Query chunking
bounds temporary memory; gathering V avoids a dense attention-by-V product.
Hard index selection is discrete. Training defaults to a straight-through
Gumbel-softmax surrogate; inference gathers selected values only. The training
surrogate uses dense soft attention for gradients and is not O(N k D).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def topk_attention(q, k, v, ratio, training=False, temperature=1.0,
                   query_chunk_size=128, dropout=0.0, bias_fn=None, capture=False,
                   straight_through=True):
    """Q/K/V: [batch, heads, tokens, channels]; exact top-k per query."""
    if not 0 < ratio <= 1 or temperature <= 0 or query_chunk_size < 1:
        raise ValueError("Invalid attention ratio, temperature or chunk size")
    batch, heads, queries, dim = q.shape
    keys = k.shape[-2]
    count = min(keys, max(1, int(keys * ratio)))
    outputs, captured = [], []
    l1, entropy = q.new_zeros(()), q.new_zeros(())
    batch_index = torch.arange(batch * heads, device=q.device)[:, None, None]
    for start in range(0, queries, query_chunk_size):
        query = q[:, :, start:start + query_chunk_size]
        # Bind start here: backward recomputes chunks after the loop ends.
        def attend(query, key, value, start=start):
            # Autocast must be disabled explicitly; .float() alone does not
            # prevent autocast from running the matmul in fp16.
            with torch.autocast(device_type=query.device.type, enabled=False):
                scores = query.float() @ key.float().transpose(-2, -1) / math.sqrt(dim)
                if bias_fn is not None:
                    scores = bias_fn(scores, query, start)
                if training:
                    uniform = torch.rand_like(scores).clamp_(1e-6, 1 - 1e-6)
                    scores = scores - torch.log(-torch.log(uniform))
                values, indices = scores.topk(count, dim=-1)
                weights = (values / temperature).softmax(dim=-1)
            if training and straight_through:
                # Hard top-k forward; continuous Gumbel-softmax gradient.
                # The paper omits the estimator details. This explicit choice
                # incurs dense AV in training, and is documented as such.
                soft = (scores / temperature).softmax(-1)
                hard = torch.zeros_like(scores).scatter(-1, indices, weights)
                relaxed = hard.detach() + (soft - soft.detach())
                used_weights = F.dropout(relaxed, p=dropout, training=training).to(value.dtype)
                result = used_weights @ value
            else:
                flat_v = value.reshape(batch * heads, keys, dim)
                selected = flat_v[batch_index, indices.reshape(batch * heads, -1, count)]
                selected = selected.reshape(batch, heads, query.shape[-2], count, dim)
                used_weights = F.dropout(weights, p=dropout, training=training).to(value.dtype)
                result = (used_weights.unsqueeze(-1) * selected).sum(dim=-2)
            chunk_l1 = weights.abs().sum()
            chunk_entropy = -(weights * weights.clamp_min(1e-8).log()).sum()
            if capture:
                return result, chunk_l1, chunk_entropy, indices.detach(), weights.detach()
            return result, chunk_l1, chunk_entropy
        if training and torch.is_grad_enabled():
            result = checkpoint(attend, query, k, v, use_reentrant=False)
        else:
            result = attend(query, k, v)
        outputs.append(result[0])
        l1, entropy = l1 + result[1], entropy + result[2]
        if capture:
            captured.append((result[3], result[4]))
    divisor = batch * heads * queries
    return torch.cat(outputs, dim=-2), l1 / divisor, entropy / divisor, captured


class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sparsity_ratio=0.1, query_chunk_size=128):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads, self.sparsity_ratio = num_heads, sparsity_ratio
        self.query_chunk_size, self.temperature = query_chunk_size, 1.0
        self.query, self.key, self.value = (nn.Linear(dim, dim) for _ in range(3))
        self.output, self.layer_norm = nn.Linear(dim, dim), nn.LayerNorm(dim)
        self.dropout = 0.1
        self.capture_attention = False

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q, k, v = [layer(tokens).reshape(b, h * w, self.num_heads, c // self.num_heads)
                   .transpose(1, 2) for layer in (self.query, self.key, self.value)]
        output, self.sparse_loss, self.attention_entropy, self.last_attention = topk_attention(
            q, k, v, self.sparsity_ratio, self.training, self.temperature,
            self.query_chunk_size, self.dropout, capture=self.capture_attention)
        output = output.transpose(1, 2).reshape(b, h * w, c)
        output = self.layer_norm(tokens + self.output(output))
        return output.transpose(1, 2).reshape(b, c, h, w)


class ConvBlock(nn.Sequential):
    """GroupNorm supports singleton batches and pooled features."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        # At least two channels per group, including the GAP 1x1 branch.
        groups = math.gcd(out_channels, min(8, max(1, out_channels // 2)))
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=dilation if kernel_size == 3 else 0,
                                  dilation=dilation, bias=False),
                         nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True))


def gather_patches(x, positions, dilation=1):
    """Gather ONLY requested 3x3 neighborhoods in Conv2d's channel-major order."""
    b, c, h, w = x.shape
    padded = F.pad(x, (dilation,) * 4).permute(0, 2, 3, 1)
    batch = positions // (h * w)
    row = (positions % (h * w)) // w
    col = positions % w
    offsets = torch.arange(3, device=x.device) * dilation
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    patches = padded[batch[:, None], row[:, None] + dy.flatten(), col[:, None] + dx.flatten()]
    return patches.transpose(1, 2)  # selected, channels, 9


class ExpertLayer(nn.Module):
    """One convolution evaluated at routed output positions only.

    LayerNorm normalizes each dispatched token independently. Spatial BN/GN
    would couple selected and omitted positions and defeat this dispatch.
    """
    def __init__(self, in_channels, out_channels, expert_type="conv", dropout=0.1):
        super().__init__()
        if expert_type not in {"conv", "dilated", "depthwise", "replica", "edge", "residual", "contour"}:
            raise ValueError(f"Unknown expert: {expert_type}")
        self.expert_type = expert_type
        self.dilation = 6 if expert_type == "dilated" else 1
        if expert_type == "depthwise":
            self.depthwise = nn.Parameter(torch.empty(in_channels, 9))
            self.pointwise = nn.Linear(in_channels, out_channels, bias=False)
            nn.init.orthogonal_(self.depthwise)
            nn.init.orthogonal_(self.pointwise.weight)
        else:
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels * 9))
            (nn.init.xavier_uniform_ if expert_type == "replica" else nn.init.orthogonal_)(self.weight)
        if expert_type == "edge":
            self.edge_projection = nn.Linear(2 * in_channels, out_channels, bias=False)
            self.register_buffer("sobel", torch.tensor([[-1., 0, 1, -2, 0, 2, -1, 0, 1],
                                                         [-1., -2, -1, 0, 0, 0, 1, 2, 1]]))
        if expert_type == "residual":
            self.skip = nn.Linear(in_channels, out_channels, bias=False)
        if expert_type == "contour":
            self.contour_gate = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions):
        patches = gather_patches(x, positions, self.dilation)
        if self.expert_type == "depthwise":
            output = self.pointwise((patches * self.depthwise).sum(-1))
        else:
            output = F.linear(patches.flatten(1), self.weight)
        if self.expert_type == "edge":
            edges = torch.einsum("nck,ek->nce", patches, self.sobel).flatten(1)
            output = output + self.edge_projection(edges)
        elif self.expert_type == "residual":
            output = output + self.skip(patches[:, :, 4])
        elif self.expert_type == "contour":
            output = output * self.contour_gate(patches[:, :, 4]).sigmoid()
        return self.dropout(F.relu(self.norm(output)))


class MixtureOfExperts(nn.Module):
    """Eq. 38/40: spatial MLP routing, top-2, one normalization, real dispatch."""
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=2,
                 noise_scale=0.1, dropout=0.1, refinement=False, dispatch_chunk_size=4096,
                 gate_hidden_dim=512, gate_activation="relu", temperature=1.,
                 balance_mode="frequency_st"):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts, self.top_k, self.out_channels = num_experts, top_k, out_channels
        if gate_hidden_dim < 1 or temperature <= 0 or dispatch_chunk_size < 1:
            raise ValueError("Gate dimensions, temperature and dispatch size must be positive")
        if gate_activation not in {"relu", "gelu"} or balance_mode not in {"frequency_st", "probability"}:
            raise ValueError("Unknown gate activation or balance mode")
        self.noise_scale, self.temperature = noise_scale, temperature
        self.gate_hidden_dim, self.gate_activation, self.balance_mode = gate_hidden_dim, gate_activation, balance_mode
        self.dispatch_chunk_size = dispatch_chunk_size
        self.gate = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_channels, gate_hidden_dim),
                                  nn.ReLU() if gate_activation == "relu" else nn.GELU(),
                                  nn.Linear(gate_hidden_dim, num_experts))
        types = ("edge", "residual", "contour") if refinement else ("conv", "dilated", "depthwise", "replica")
        self.experts = nn.ModuleList(ExpertLayer(in_channels, out_channels, types[i % len(types)], dropout)
                                     for i in range(num_experts))

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(-1, c)
        logits = self.gate(tokens)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_scale
        probabilities = (logits.float() / self.temperature).softmax(-1)
        values, indices = logits.topk(self.top_k, dim=-1)
        weights = (values.float() / self.temperature).softmax(-1).to(x.dtype)
        counts = torch.bincount(indices.detach().flatten(), minlength=self.num_experts)
        self.last_usage = counts.float() / tokens.shape[0]  # sums to top_k (Eq. 43)
        importance = probabilities.mean(0)
        if self.balance_mode == "frequency_st":
            # Table 8 / Eq.62: exact hard-frequency variance in forward.
            # Explicit straight-through probability proxy in backward; hard
            # selection counts themselves have no derivative.
            soft_usage = self.top_k * importance
            usage = self.last_usage + (soft_usage - soft_usage.detach())
            self.balance_loss = usage.var(unbiased=False)
        else:
            # Alternative Eq.44/45 probability-based ratio (uniform minimum 1).
            self.balance_loss = self.num_experts * importance.square().sum() / importance.sum().square()
        self.gate_entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()
        self.last_dispatch_counts = counts.detach()
        self.last_indices, self.last_weights = indices.detach(), weights.detach()
        output = x.new_zeros(tokens.shape[0], self.out_channels)
        for expert_id, expert in enumerate(self.experts):
            positions, slots = torch.where(indices == expert_id)
            if positions.numel() == 0:
                continue
            for offset in range(0, positions.numel(), self.dispatch_chunk_size):
                selected = positions[offset:offset + self.dispatch_chunk_size]
                selected_slots = slots[offset:offset + self.dispatch_chunk_size]
                if self.training and torch.is_grad_enabled():
                    expert_output = checkpoint(expert, x, selected, use_reentrant=False)
                else:
                    expert_output = expert(x, selected)
                output = output.index_add(0, selected, expert_output * weights[selected, selected_slots, None])
        return output.reshape(b, h, w, self.out_channels).permute(0, 3, 1, 2).contiguous()


class SqueezeExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(channels, max(8, channels // 16), 1), nn.ReLU(),
                                  nn.Conv2d(max(8, channels // 16), channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


def activation_mask(x, ratio=0.1, window=8):
    """Section 4.4: retain top responses in each spatial window."""
    b, c, h, w = x.shape
    ph, pw = (-h) % window, (-w) % window
    scores = F.pad(x.square().mean(1), (0, pw, 0, ph), value=float("-inf"))
    scores = scores.reshape(b, (h + ph) // window, window, (w + pw) // window, window)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(b, -1, window * window)
    valid = torch.isfinite(scores)
    # Boundary windows retain a fraction of valid pixels, never padding.
    ranks = scores.argsort(-1, descending=True).argsort(-1)
    counts = (valid.sum(-1, keepdim=True).float() * ratio).floor().clamp_min(1)
    mask = (ranks < counts) & valid
    mask = mask.reshape(b, (h + ph) // window, (w + pw) // window, window, window)
    mask = mask.permute(0, 1, 3, 2, 4).reshape(b, h + ph, w + pw)[:, :h, :w]
    return x * mask.unsqueeze(1)


class EnhancedASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, sparsity_ratio=0.1,
                 rates=(6, 12, 18), query_chunk_size=128):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBlock(in_channels, out_channels, 1))
        self.branches = nn.ModuleList([ConvBlock(in_channels, out_channels, 1)] +
            [ConvBlock(in_channels, out_channels, 3, dilation=r) for r in rates])
        # Eq. 49 projects BEFORE attention, avoiding 5x-channel attention.
        self.project = ConvBlock(5 * out_channels, out_channels, 1)
        self.sparse_attention = SparseAttention(out_channels, 8, sparsity_ratio, query_chunk_size)
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x):
        if min(x.shape[-2:]) <= 18:
            raise ValueError("ASPP requires a feature grid larger than dilation 18; use SAM encoder size >= 304")
        pooled = F.interpolate(self.image_pool(x), x.shape[-2:], mode="bilinear", align_corners=False)
        branches = [activation_mask(branch(x), self.sparsity_ratio) for branch in self.branches]
        features = self.project(torch.cat(branches + [pooled], dim=1))
        return self.se(self.sparse_attention(features))

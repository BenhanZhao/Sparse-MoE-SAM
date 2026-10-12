"""Optional legacy token clustering utilities; not used by SparseMoESAM."""
import math
import torch
from torch import nn


def index_points(points, idx):
    batches = torch.arange(points.shape[0], device=points.device)
    batches = batches.reshape(points.shape[0], *([1] * (idx.ndim - 1)))
    return points[batches, idx]


@torch.no_grad()
def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None):
    x = token_dict["x"]
    b, n, c = x.shape
    if n == 0:
        raise ValueError("Cannot cluster an empty token sequence")
    count = min(n, max(1, int(cluster_num)))
    valid = torch.ones(b, n, dtype=torch.bool, device=x.device) if token_mask is None else token_mask.bool()
    distance = torch.cdist(x.float(), x.float()) / math.sqrt(c)
    maximum = distance.amax(dim=(1, 2), keepdim=True) + 1
    masked = torch.where(valid[:, None, :], distance, maximum)
    nearest = masked.topk(min(n, max(1, k)), dim=-1, largest=False).values
    density = (-nearest.square().mean(-1)).exp()
    # Deterministic tie-breaking, including eval and repeated calls.
    density = (density + torch.arange(n, device=x.device).float()[None] * (1e-6 / n)) * valid
    higher = density[:, None, :] > density[:, :, None]
    delta = torch.where(higher, masked, maximum).amin(-1)
    scores = (density * delta).masked_fill(~valid, -1.)
    centers = scores.topk(count, dim=-1).indices
    center_distance = index_points(distance, centers)
    center_valid = index_points(valid.unsqueeze(-1), centers).squeeze(-1)
    center_distance = torch.where(center_valid.unsqueeze(-1), center_distance, maximum)
    assigned = center_distance.argmin(dim=1)
    batches = torch.arange(b, device=x.device)[:, None]
    assigned[batches, centers] = torch.arange(count, device=x.device)[None]
    return assigned, count


def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    x = token_dict["x"]
    b, n, c = x.shape
    if not 1 <= cluster_num <= n or idx_cluster.shape != (b, n):
        raise ValueError("Invalid cluster count or assignments")
    weight = x.new_ones(b, n, 1) if token_weight is None else token_weight
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Token weights must be finite and nonnegative")
    if token_dict.get("mask") is not None:
        weight = weight * token_dict["mask"].bool().unsqueeze(-1)
    idx = idx_cluster + torch.arange(b, device=x.device)[:, None] * cluster_num
    total = x.new_zeros(b * cluster_num, 1).index_add(0, idx.flatten(), weight.reshape(-1, 1).to(x.dtype))
    normal = weight / total[idx].clamp_min(1e-6)
    merged = x.new_zeros(b * cluster_num, c).index_add(0, idx.flatten(), (x * normal).reshape(-1, c).to(x.dtype))
    original_indices = token_dict["idx_token"]
    new_indices = index_points(idx_cluster.unsqueeze(-1), original_indices).squeeze(-1)
    aggregate = token_dict["agg_weight"] * index_points(normal, original_indices)
    aggregate = aggregate / aggregate.amax(dim=1, keepdim=True).clamp_min(1e-6)
    return {"x": merged.reshape(b, cluster_num, c), "token_num": cluster_num,
            "idx_token": new_indices, "agg_weight": aggregate,
            "mask": total.reshape(b, cluster_num) > 0}


class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5):
        super().__init__()
        if sample_ratio <= 0:
            raise ValueError("sample_ratio must be positive")
        self.sample_ratio, self.k = sample_ratio, k
        self.projection = nn.Linear(embed_dim, dim_out) if embed_dim != dim_out else nn.Identity()

    def forward(self, token_dict, sample_ratio=None):
        n = token_dict["x"].shape[1]
        ratio = self.sample_ratio if sample_ratio is None else sample_ratio
        if ratio <= 0:
            raise ValueError("sample_ratio must be positive")
        count = min(n, max(1, math.ceil(ratio if ratio > 1 else n * ratio)))
        assignments, count = cluster_dpc_knn(token_dict, count, self.k, token_dict.get("mask"))
        merged = merge_tokens(token_dict, assignments, count)
        merged["x"] = self.projection(merged["x"])
        merged["x"] = merged["x"] * merged["mask"].unsqueeze(-1)
        return merged, token_dict


class TCBlock(nn.Module):
    """Trainable masked self/cross-attention plus residual MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1, use_sr_layer=False):
        super().__init__()
        if qk_scale is not None or sr_ratio != 1 or use_sr_layer:
            raise ValueError("Custom qk_scale/spatial reduction is not supported")
        self.norm1, self.norm_kv, self.norm2 = norm_layer(dim), norm_layer(dim), norm_layer(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop,
                                               bias=qkv_bias, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), act_layer(),
                                 nn.Dropout(drop), nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop))
        self.drop_path = drop_path

    def residual(self, x):
        if not self.training or not self.drop_path:
            return x
        keep = 1 - self.drop_path
        mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < keep).to(x.dtype)
        return x * mask / keep

    def forward(self, inputs):
        query, kv = inputs if isinstance(inputs, (tuple, list)) else (inputs, inputs)
        if kv is None:
            kv = query
        x, keys = query["x"], kv["x"]
        mask = kv.get("mask")
        padding = None
        all_empty = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        if mask is not None:
            padding = ~mask.bool().clone()
            all_empty = padding.all(-1)
            padding[all_empty, 0] = False  # avoid all -inf attention rows
        attention, _ = self.attention(self.norm1(x), self.norm_kv(keys), self.norm_kv(keys),
                                      key_padding_mask=padding, need_weights=False)
        attention = attention.masked_fill(all_empty[:, None, None], 0.)
        x = x + self.residual(attention)
        x = x + self.residual(self.mlp(self.norm2(x)))
        if query.get("mask") is not None:
            x = x * query["mask"].bool().unsqueeze(-1)
        result = dict(query)
        result["x"] = x
        return result

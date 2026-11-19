"""Original graph encoder class definitions; no legacy training or figure I/O."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerLayer(nn.Module):
    def __init__(self, d: int, heads: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        assert d % heads == 0
        self.d = d
        self.heads = heads
        self.dh = d // heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.edge_msg = nn.Linear(edge_dim, d)
        self.out = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        efeat: torch.Tensor,
    ) -> torch.Tensor:
        # h: (S, N, d); src/dst: (E,); efeat: (E, edge_dim)
        s, n, _ = h.shape
        e = src.numel()
        q = self.q(h)[:, src].view(s, e, self.heads, self.dh)
        k = self.k(h)[:, dst].view(s, e, self.heads, self.dh)
        score = (q * k).sum(-1) + self.edge_bias(efeat).unsqueeze(0)
        score = F.leaky_relu(score, negative_slope=0.2)

        dst_idx = dst.view(1, e, 1).expand(s, -1, self.heads)
        max_score = torch.full(
            (s, n, self.heads), -1e9, dtype=score.dtype, device=score.device
        )
        max_score.scatter_reduce_(1, dst_idx, score, reduce="amax", include_self=True)
        score = torch.exp(score - max_score.gather(1, dst_idx))
        denom = torch.zeros_like(max_score)
        denom.scatter_add_(1, dst_idx, score)
        att = score / (denom.gather(1, dst_idx) + 1e-12)

        msg = (self.v(h)[:, src] + self.edge_msg(efeat).unsqueeze(0)).view(
            s, e, self.heads, self.dh
        )
        agg = torch.zeros(s, n, self.heads, self.dh, dtype=h.dtype, device=h.device)
        agg.scatter_add_(
            1,
            dst_idx.unsqueeze(-1).expand(s, e, self.heads, self.dh),
            att.unsqueeze(-1) * msg,
        )
        agg = agg.view(s, n, self.d)

        h = self.norm1(h + self.dropout(self.out(agg)))
        h = self.norm2(h + self.dropout(self.ff(h)))
        return h

class MaskedGraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d: int,
        heads: int,
        n_layers: int,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.virtual_token = nn.Parameter(torch.zeros(d))
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(d, heads, edge_dim, dropout) for _ in range(n_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, in_dim),
        )
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        efeat: torch.Tensor,
        return_nodes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        s, n, _ = x.shape
        h = self.inp(x)
        h = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        h = torch.cat([h, self.virtual_token.view(1, 1, -1).expand(s, 1, -1)], dim=1)
        for layer in self.layers:
            h = layer(h, src, dst, efeat)
        h = self.norm(h)
        block_h = h[:, :n]
        pred = self.decoder(block_h)
        if return_nodes:
            return pred, block_h
        return pred

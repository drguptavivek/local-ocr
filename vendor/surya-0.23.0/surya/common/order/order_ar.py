"""Autoregressive reading-order head (inference only).

Box tokens (geometry + label) cross-attend to the FULL rf-detr encoder feature map, then an AR
decoder emits the reading-order permutation as indices into the canonically (raster) sorted box
sequence. Constrained greedy decode -> a valid permutation (every box once, none invented).

Vendored from training/models/rtdetr/order_ar.py (training code stripped). The 19-class layout
taxonomy must match the layout detector the features come from.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# 19-class layout taxonomy (sorted), must match the fast_layout detector's classes.
LAYOUT_CLASSES = sorted(
    [
        "Caption",
        "Footnote",
        "Equation-Block",
        "List-Group",
        "Page-Header",
        "Page-Footer",
        "Image",
        "Section-Header",
        "Table",
        "Text",
        "Complex-Block",
        "Code-Block",
        "Form",
        "Table-Of-Contents",
        "Figure",
        "Chemical-Block",
        "Diagram",
        "Bibliography",
        "Blank-Page",
    ]
)
N_LABELS = len(LAYOUT_CLASSES)
MAX_BOXES = 128


def box_features(boxes_1000):
    """boxes 0-1000 [N,4] (x0,y0,x1,y1) -> [N,8] normalized (x0,y0,x1,y1,cx,cy,w,h)."""
    b = np.asarray(boxes_1000, dtype=np.float32) / 1000.0
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
    return np.stack([x0, y0, x1, y1, cx, cy, w, h], axis=1)


def canonical_order(boxes_1000, n_bands=24):
    """Deterministic y-banded raster order: order[p] = original index of the box at raster pos p."""
    b = np.asarray(boxes_1000, dtype=np.float32)
    band_h = 1000.0 / n_bands
    keys = [
        (int(b[i, 1] // band_h), float(b[i, 0]), float(b[i, 1]), float(b[i, 2]))
        for i in range(len(b))
    ]
    return sorted(range(len(b)), key=lambda i: keys[i])


class ReadingOrderAR(nn.Module):
    def __init__(
        self,
        d=128,
        layers=3,
        heads=4,
        feat_dim=256,
        feat_hw=28,
        max_boxes=MAX_BOXES,
        dropout=0.0,
    ):
        super().__init__()
        self.d = d
        self.max_boxes = max_boxes
        self.use_feat = bool(feat_dim)
        self.geom = nn.Linear(8, d)
        self.lab = nn.Embedding(N_LABELS, d)
        if self.use_feat:
            self.feat_proj = nn.Linear(feat_dim, d)
            self.feat_pos = nn.Parameter(torch.zeros(1, feat_hw * feat_hw, d))
            ctx_layer = nn.TransformerDecoderLayer(
                d, heads, d * 4, batch_first=True, dropout=dropout
            )
            self.ctx = nn.TransformerDecoder(ctx_layer, layers)
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d, heads, d * 4, batch_first=True, dropout=dropout
            )
            self.enc = nn.TransformerEncoder(enc_layer, layers)
        dec_layer = nn.TransformerDecoderLayer(
            d, heads, d * 4, batch_first=True, dropout=dropout
        )
        self.dec = nn.TransformerDecoder(dec_layer, layers)
        self.bos = nn.Parameter(torch.zeros(d))
        self.step = nn.Embedding(max_boxes + 1, d)
        self.out = nn.Linear(d, max_boxes)

    def encode(self, feats, labels, mask, fmap=None):
        x = self.geom(feats) + self.lab(labels)
        if self.use_feat and fmap is not None:
            f = self.feat_proj(fmap) + self.feat_pos[:, : fmap.shape[1]]
            x = self.ctx(x, f, tgt_key_padding_mask=~mask)
        else:
            x = self.enc(x, src_key_padding_mask=~mask)
        return x

    @torch.no_grad()
    def decode(self, feats, labels, mask, fmap=None):
        """Greedy constrained decode -> list (per batch item) of raster-position permutations."""
        memory = self.encode(feats, labels, mask, fmap)
        B = memory.shape[0]
        K = mask.sum(1)
        out = []
        for b in range(B):
            k = int(K[b].item())
            mem = memory[b : b + 1, :k]
            used = torch.zeros(k, dtype=torch.bool, device=memory.device)
            din = [self.bos + self.step.weight[0]]
            seq = []
            for t in range(k):
                x = torch.stack(din, 0).unsqueeze(0)
                causal = torch.triu(
                    torch.ones(t + 1, t + 1, device=memory.device, dtype=torch.bool), 1
                )
                h = self.dec(x, mem, tgt_mask=causal)
                logits = self.out(h[0, -1]).clone()
                logits[k:] = float("-inf")
                logits[:k] = logits[:k].masked_fill(used, float("-inf"))
                nxt = int(logits.argmax().item())
                seq.append(nxt)
                used[nxt] = True
                din.append(mem[0, nxt] + self.step.weight[min(t + 1, self.max_boxes)])
            out.append(seq)
        return out

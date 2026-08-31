"""
PCAG.py

PyTorch implementation of PCAG for two modalities, based on:
"Pre-gating and contextual attention gate — A new fusion method for multi-modal data tasks"
Neural Networks 179 (2024) 106553.

It contains:
  - PreGating
  - ContextualAttentionGate
  - PCAGFusionForTwoModalities

Expected input:
  x1: Tensor [batch, len1, in_dim1]
  x2: Tensor [batch, len2, in_dim2]

Output:
  logits: Tensor [batch, num_classes]
  extras: dict with intermediate tensors for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_fill_scores(
    scores: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    scores: [B, Lq, Lk]
    key_padding_mask: [B, Lk], True for valid tokens/features, False for padding.
    """
    if key_padding_mask is None:
        return scores
    mask = key_padding_mask[:, None, :].to(dtype=torch.bool)
    return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)


def masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    x: [B, L, D]
    mask: [B, L], True for valid positions.
    """
    if mask is None:
        return x.mean(dim=dim)
    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * mask_f).sum(dim=dim) / mask_f.sum(dim=dim).clamp_min(eps)


class PreGating(nn.Module):
    """
    Pre-gating applied before softmax attention.

    Given Q_i and K_j, ordinary cross-attention uses:
        A = Q_i K_j^T / sqrt(d)

    PCAG instead gates A before softmax:
        A_gated = A * P(A)

    Two settings are implemented from the paper:
      - sigmoid: P(A) = sigmoid(Q) @ sigmoid(K)^T
      - tanh:    P(A) = (tanh(Q) @ tanh(K)^T + 1) / 2

    Args:
        mode: "sigmoid", "tanh", or "none".
    """

    def __init__(self, mode: str = "tanh") -> None:
        super().__init__()
        valid = {"sigmoid", "tanh", "none"}
        if mode not in valid:
            raise ValueError(f"mode must be one of {valid}, got {mode!r}")
        self.mode = mode
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        q: [B, Lq, D]
        k: [B, Lk, D]
        returns gate matrix [B, Lq, Lk]
        """
        if self.mode == "none":
            return torch.ones(q.size(0), q.size(1), k.size(1), device=q.device, dtype=q.dtype)

        if self.mode == "sigmoid":
            q_prob = torch.sigmoid(q)
            k_prob = torch.sigmoid(k)
            return torch.matmul(q_prob, k_prob.transpose(-2, -1))
        else:  # tanh
            q_prob = torch.tanh(q)
            k_prob = torch.tanh(k)
            return (torch.matmul(q_prob, k_prob.transpose(-2, -1)) + 1.0) / 2.0



class ContextualAttentionGate(nn.Module):
    """
    Contextual Attention Gate (CAG).

    From the paper:
        G_m = ReLU(W_h Qhat_m + W_q Q_m + b_G)
        E_m = ReLU(W_E Qhat_m + b_E)
        C_m = Norm(E_m) ◦ Norm(G_m)

    This implementation preserves sequence length:
        q, q_hat: [B, L, D]
        output c: [B, L, D]
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.g_from_attended = nn.Linear(dim, dim)
        self.g_from_query = nn.Linear(dim, dim)
        self.e_from_attended = nn.Linear(dim, dim)

        self.norm_g = nn.LayerNorm(dim, eps=norm_eps)
        self.norm_e = nn.LayerNorm(dim, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, q_hat: torch.Tensor) -> torch.Tensor:
        g = F.relu(self.g_from_attended(q_hat) + self.g_from_query(q))
        e = F.relu(self.e_from_attended(q_hat))
        c = self.norm_e(e) * self.norm_g(g)
        return self.dropout(c)


class CrossAttentionWithPreGating(nn.Module):
    """
    Bidirectional two-modality cross-attention with PCAG pre-gating.

    Produces:
      q1, q2: projected modality queries
      qhat1: modality-1 attended to modality-2 values
      qhat2: modality-2 attended to modality-1 values
    """

    def __init__(
        self,
        in_dim1: int,
        in_dim2: int,
        dim: int,
        pre_gate_mode: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim

        self.q1 = nn.Linear(in_dim1, dim)
        self.k1 = nn.Linear(in_dim1, dim)
        self.v1 = nn.Linear(in_dim1, dim)

        self.q2 = nn.Linear(in_dim2, dim)
        self.k2 = nn.Linear(in_dim2, dim)
        self.v2 = nn.Linear(in_dim2, dim)

        self.pre_gating = PreGating(pre_gate_mode)
        self.attn_dropout = nn.Dropout(dropout)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        pg = self.pre_gating(q, k)
        gated_scores = raw_scores * pg
        gated_scores = _masked_fill_scores(gated_scores, key_mask)
        attn = F.softmax(gated_scores, dim=-1)
        attn = self.attn_dropout(attn)
        q_hat = torch.matmul(attn, v)
        return q_hat, attn, pg

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        q1, k1, v1 = self.q1(x1), self.k1(x1), self.v1(x1)
        q2, k2, v2 = self.q2(x2), self.k2(x2), self.v2(x2)

        qhat1, attn12, pg12 = self._attend(q1, k2, v2, key_mask=mask2)
        qhat2, attn21, pg21 = self._attend(q2, k1, v1, key_mask=mask1)

        return {
            "q1": q1,
            "q2": q2,
            "qhat1": qhat1,
            "qhat2": qhat2,
            "attn12": attn12,
            "attn21": attn21,
            "pre_gate12": pg12,
            "pre_gate21": pg21,
        }


@dataclass
class PCAGOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    joint: torch.Tensor
    extras: Dict[str, torch.Tensor]


class PCAGFusionForTwoModalities(nn.Module):
    """
    Full PCAG flow for two modalities:
      1. Project each modality into Q/K/V.
      2. Apply pre-gated bidirectional cross-attention.
      3. Apply CAG to each modality.
      4. Build inter-modal features C_hat_m.
      5. Build intra-modal features A_m from Q_m.
      6. Add all inter-modal and intra-modal vectors.
      7. Feed joint representation into classifier.

    Args:
        in_dim1: input feature dimension of modality 1.
        in_dim2: input feature dimension of modality 2.
        dim: shared hidden dimension d_dim.
        num_classes: number of target classes.
        pre_gate_mode: "sigmoid", "tanh", or "none".
        dropout: dropout used in attention/CAG/classifier path.
        classifier_hidden_dim: optional hidden layer before logits. If None, use one FC layer.
    """

    def __init__(
        self,
        in_dim1: int,
        in_dim2: int,
        dim: int,
        num_classes: int,
        pre_gate_mode: str = "tanh",
        dropout: float = 0.3,
        classifier_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.cross_attention = CrossAttentionWithPreGating(
            in_dim1=in_dim1,
            in_dim2=in_dim2,
            dim=dim,
            pre_gate_mode=pre_gate_mode,
            dropout=dropout,
        )

        self.cag1 = ContextualAttentionGate(dim=dim, dropout=dropout)
        self.cag2 = ContextualAttentionGate(dim=dim, dropout=dropout)

        # Inter-modal feature transforms: C_hat_m = ReLU(W C_m + b)
        self.inter_fc1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout))
        self.inter_fc2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout))

        # Intra-modal feature transforms: A_m = ReLU(W Q_m + b)
        self.intra_fc1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout))
        self.intra_fc2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout))

        if classifier_hidden_dim is None:
            self.classifier = nn.Linear(dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, num_classes),
            )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
        return_output_dataclass: bool = False,
    ):
        """
        x1: [B, L1, in_dim1]
        x2: [B, L2, in_dim2]
        mask1: optional [B, L1], True for valid modality-1 positions.
        mask2: optional [B, L2], True for valid modality-2 positions.

        By default returns logits. Set return_output_dataclass=True to inspect the full flow.
        """
        z = self.cross_attention(x1, x2, mask1=mask1, mask2=mask2)

        c1 = self.cag1(z["q1"], z["qhat1"])
        c2 = self.cag2(z["q2"], z["qhat2"])

        # Eq. 9 style inter-modal vectors
        c_hat1_seq = self.inter_fc1(c1)
        c_hat2_seq = self.inter_fc2(c2)
        c_hat1 = masked_mean(c_hat1_seq, mask1)
        c_hat2 = masked_mean(c_hat2_seq, mask2)

        # Eq. 10 style intra-modal vectors
        a1_seq = self.intra_fc1(z["q1"])
        a2_seq = self.intra_fc2(z["q2"])
        a1 = masked_mean(a1_seq, mask1)
        a2 = masked_mean(a2_seq, mask2)

        # Eq. 11 additive fusion
        joint = c_hat1 + c_hat2 + a1 + a2

        # Eq. 12 classifier; use CrossEntropyLoss on logits during training
        logits = self.classifier(joint)
        probabilities = F.softmax(logits, dim=-1)

        if not return_output_dataclass:
            return logits

        extras = {
            **z,
            "cag1": c1,
            "cag2": c2,
            "c_hat1": c_hat1,
            "c_hat2": c_hat2,
            "a1": a1,
            "a2": a2,
        }
        return PCAGOutput(
            logits=logits,
            probabilities=probabilities,
            joint=joint,
            extras=extras,
        )


class PCAGFusionForThreeModalities(nn.Module):
    def __init__(
        self,
        in_dim1: int,
        in_dim2: int,
        in_dim3: int,
        dim: int,
        num_classes: int,
        pre_gate_mode: str = "tanh",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dim = dim


        self.q = nn.ModuleList([
            nn.Linear(in_dim1, dim),
            nn.Linear(in_dim2, dim),
            nn.Linear(in_dim3, dim),
        ])
        self.k = nn.ModuleList([
            nn.Linear(in_dim1, dim),
            nn.Linear(in_dim2, dim),
            nn.Linear(in_dim3, dim),
        ])
        self.v = nn.ModuleList([
            nn.Linear(in_dim1, dim),
            nn.Linear(in_dim2, dim),
            nn.Linear(in_dim3, dim),
        ])

        self.pre_gating = PreGating(pre_gate_mode)
        self.attn_dropout = nn.Dropout(dropout)
        # For each modality i, it learns weights over the other two modalities.
        self.qhat_fusion_weights = nn.Parameter(torch.ones(3, 2))

        self.cag = nn.ModuleList([
            ContextualAttentionGate(dim, dropout),
            ContextualAttentionGate(dim, dropout),
            ContextualAttentionGate(dim, dropout),
        ])

        self.inter_fc = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)),
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)),
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)),
        ])

        self.intra_fc = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)),
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)),
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)),
        ])

        self.classifier = nn.Linear(dim, num_classes)

    def _attend(self, q, k, v, key_mask=None):
        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        gate = self.pre_gating(q, k)
        scores = raw_scores * gate

        if key_mask is not None:
            scores = _masked_fill_scores(scores, key_mask)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        q_hat = torch.matmul(attn, v)
        return q_hat, attn, gate

    def forward(
        self,
        x1,
        x2,
        x3,
        mask1=None,
        mask2=None,
        mask3=None,
        return_extras: bool = False,
    ):
        xs = [x1, x2, x3]
        masks = [mask1, mask2, mask3]

        qs = [self.q[i](xs[i]) for i in range(3)]
        ks = [self.k[i](xs[i]) for i in range(3)]
        vs = [self.v[i](xs[i]) for i in range(3)]

        qhats = []
        attns = {}
        gates = {}

        for i in range(3):
            incoming = []

            for j in range(3):
                if i == j:
                    continue

                q_hat_ij, attn_ij, gate_ij = self._attend(
                    qs[i],
                    ks[j],
                    vs[j],
                    key_mask=masks[j],
                )

                incoming.append(q_hat_ij)
                attns[f"{i+1}<-{j+1}"] = attn_ij
                gates[f"{i+1}<-{j+1}"] = gate_ij

            # modality i receives information from the other two modalities
            w = F.softmax(self.qhat_fusion_weights[i], dim=0)  

            q_hat_i = (
                w[0] * incoming[0]
                + w[1] * incoming[1]
            )


            qhats.append(q_hat_i)

        inter_vectors = []
        intra_vectors = []
        cag_outputs = []

        for i in range(3):
            c_i = self.cag[i](qs[i], qhats[i])
            cag_outputs.append(c_i)

            c_hat_i_seq = self.inter_fc[i](c_i)
            a_i_seq = self.intra_fc[i](qs[i])

            c_hat_i = masked_mean(c_hat_i_seq, masks[i])
            a_i = masked_mean(a_i_seq, masks[i])

            inter_vectors.append(c_hat_i)
            intra_vectors.append(a_i)

        joint = sum(inter_vectors) + sum(intra_vectors)

        logits = self.classifier(joint)

        if not return_extras:
            return logits

        return {
            "logits": logits,
            "probabilities": F.softmax(logits, dim=-1),
            "joint": joint,
            "attns": attns,
            "pre_gates": gates,
            "qhats": qhats,
            "qhat_fusion_weights": F.softmax(self.qhat_fusion_weights, dim=1),
            "cag_outputs": cag_outputs,
            "inter_vectors": inter_vectors,
            "intra_vectors": intra_vectors,
        }




if __name__ == "__main__":
    # Minimal smoke test
    batch, len1, len2 = 4, 12, 20
    in_dim1, in_dim2 = 768, 512
    dim, num_classes = 64, 3

    model = PCAGFusionForTwoModalities(
        in_dim1=in_dim1,
        in_dim2=in_dim2,
        dim=dim,
        num_classes=num_classes,
        pre_gate_mode="tanh",
        dropout=0.1,
    )

    x1 = torch.randn(batch, len1, in_dim1)
    x2 = torch.randn(batch, len2, in_dim2)
    out = model(x1, x2, return_output_dataclass=True)

    print("logits:", out.logits.shape)          # [4, 3]
    print("probabilities:", out.probabilities.shape)
    print("joint:", out.joint.shape)            # [4, 64]

    y = torch.randint(0, num_classes, (batch,))
    loss = nn.CrossEntropyLoss()(out.logits, y)
    loss.backward()
    print("loss:", float(loss))

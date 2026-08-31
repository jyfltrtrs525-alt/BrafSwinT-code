"""BrafSwinConcatFusion: Concat late fusion with SwinT V1 (torchvision).

    Image:  SwinT V1 (torchvision) → pool → [B, 768]
    Clinical: ClinicalBranch → mean → [B, D_clin]
    Concat:  [B, 768 + D_clin] → MLP → shared_feat → 3 heads (hierarchical routing)
"""

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

from .brafswin import ClinicalBranch
from .PCAG import PCAGFusionForTwoModalities


class BrafSwinConcatFusion(nn.Module):
    """Concat late fusion: SwinT V1 + Clinical, no PCAG, no radiomics.

    Args:
        pcag_dim:           shared feature dimension (default 384)
        clinical_embed_dim: clinical branch embedding dim (default 128)
        clinical_num_features: number of clinical inputs (default 3: age, gender, hashimoto)
        drop_out:           dropout rate
    """

    def __init__(self,
                 pcag_dim=384,
                 clinical_embed_dim=128,
                 clinical_num_features=3,
                 drop_out=0.2):
        super().__init__()

        # --- Image backbone: SwinT V1 (torchvision) ---
        backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        img_dim = backbone.head.in_features  # 768
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.img_dim = img_dim

        # --- Clinical branch ---
        self.clin_branch = ClinicalBranch(num_features=clinical_num_features, dim=clinical_embed_dim)

        # --- Concat fusion ---
        fusion_in_dim = img_dim + clinical_embed_dim   # 768 + 128 = 896

        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in_dim, pcag_dim * 2),
            nn.LayerNorm(pcag_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(pcag_dim * 2, pcag_dim),
        )

        # --- Heads: BCE-style (single logit, sigmoid) ---
        self.mal_head = nn.Linear(pcag_dim, 1)
        self.mutation_head_mal = nn.Linear(pcag_dim, 1)
        self.mutation_head_benign = nn.Linear(pcag_dim, 1)

    def forward(self, roi, clin_features, return_feature=False):
        """
        Args:
            roi:           [B, 3, H, W]
            clin_features: [B, 3, 1]

        Returns:
            mal_logits:          [B, 1]
            mut_mal_logits:      [B, 1]
            mut_benign_logits:   [B, 1]
            (optional) shared_feat: [B, pcag_dim]
        """
        img_feat = self.backbone(roi)                            # [B, 768]
        clin_feat = self.clin_branch(clin_features)               # [B, 3, D_clin]
        clin_pooled = clin_feat.mean(dim=1)                       # [B, D_clin]

        fused = torch.cat([img_feat, clin_pooled], dim=1)         # [B, 768 + D_clin]
        shared_feat = self.fusion_proj(fused)                     # [B, pcag_dim]

        mal_logits = self.mal_head(shared_feat)
        mut_mal_logits = self.mutation_head_mal(shared_feat)
        mut_benign_logits = self.mutation_head_benign(shared_feat)

        if return_feature:
            return mal_logits, mut_mal_logits, mut_benign_logits, shared_feat
        return mal_logits, mut_mal_logits, mut_benign_logits


class BrafSwinPCAGFusion(nn.Module):
    """PCAG cross-attention fusion: SwinT V1 + Clinical, BCE heads.

    Different from BrafSwinConcatFusion: uses PCAG cross-attention instead of concat+MLP.

    Args:
        pcag_dim:           shared feature dimension (default 384)
        clinical_embed_dim: clinical branch embedding dim (default 128)
        clinical_num_features: number of clinical inputs (default 3)
        drop_out:           dropout rate
        pre_gate_mode:      PCAG pre-gating mode ("tanh", "sigmoid", "none")
    """

    def __init__(self,
                 pcag_dim=384,
                 clinical_embed_dim=128,
                 clinical_num_features=3,
                 drop_out=0.2,
                 pre_gate_mode="tanh"):
        super().__init__()

        # --- Image backbone: SwinT V1 (torchvision) ---
        backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        img_dim = backbone.head.in_features  # 768
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.img_dim = img_dim

        # --- Clinical branch ---
        self.clin_branch = ClinicalBranch(num_features=clinical_num_features, dim=clinical_embed_dim)

        # --- PCAG cross-attention fusion (num_classes=1 for BCE) ---
        self.fusion = PCAGFusionForTwoModalities(
            in_dim1=img_dim,
            in_dim2=clinical_embed_dim,
            dim=pcag_dim,
            num_classes=1,
            pre_gate_mode=pre_gate_mode,
            dropout=drop_out,
        )
        self.pcag_dim = pcag_dim

        # --- BCE-style heads ---
        self.mutation_head_mal = nn.Linear(pcag_dim, 1)
        self.mutation_head_benign = nn.Linear(pcag_dim, 1)

    def forward(self, roi, clin_features, return_feature=False):
        """
        Args:
            roi:           [B, 3, H, W]
            clin_features: [B, 3, 1]

        Returns:
            mal_logits:          [B, 1]
            mut_mal_logits:      [B, 1]
            mut_benign_logits:   [B, 1]
            (optional) shared_feat: [B, pcag_dim]
        """
        img_feat = self.backbone(roi)                            # [B, 768]
        img_feat = img_feat.unsqueeze(1)                         # [B, 1, 768]
        clin_feat = self.clin_branch(clin_features)               # [B, 3, D_clin]

        out = self.fusion(img_feat, clin_feat, return_output_dataclass=True)
        mal_logits = out.logits                                   # [B, 1]
        shared_feat = out.joint                                   # [B, pcag_dim]

        mut_mal_logits = self.mutation_head_mal(shared_feat)
        mut_benign_logits = self.mutation_head_benign(shared_feat)

        if return_feature:
            return mal_logits, mut_mal_logits, mut_benign_logits, shared_feat
        return mal_logits, mut_mal_logits, mut_benign_logits

# BrafSwin.py


# 图像分支：使用微软官方 Swin Transformer 工程中的 `build_model()`
# 输入：

  # ROI 超声图像（256×256）
  # 930维 radiomics 特征
  # 4维 clinical feature
# 图像分支使用：

  # swinv2_tiny_patch4_window8_256.pth
# 图像特征：

  # 提取 4-stage hierarchical feature
  # multi-scale fusion
# 最终：

  # attention gating multimodal fusion

# BrafSwin.py 完整代码

def _ensure_config_defaults(cfg):
    """Add default config keys required by build_model if missing."""
    cfg.set_new_allowed(True)
    if not hasattr(cfg, "FUSED_LAYERNORM"):
        cfg.FUSED_LAYERNORM = False
    if not hasattr(cfg, "FUSED_WINDOW_PROCESS"):
        cfg.FUSED_WINDOW_PROCESS = False
    if not hasattr(cfg, "TRAIN"):
        cfg.TRAIN = CfgNode({"USE_CHECKPOINT": False})
    elif not hasattr(cfg.TRAIN, "USE_CHECKPOINT"):
        cfg.TRAIN.USE_CHECKPOINT = False
    if not hasattr(cfg.MODEL, "DROP_RATE"):
        cfg.MODEL.DROP_RATE = 0.0
    if not hasattr(cfg.MODEL, "DROP_PATH_RATE"):
        cfg.MODEL.DROP_PATH_RATE = 0.2
    if not hasattr(cfg.MODEL, "NUM_CLASSES"):
        cfg.MODEL.NUM_CLASSES = 2
    if not hasattr(cfg.MODEL, "SWIN"):
        cfg.MODEL.SWIN = CfgNode({})
    if not hasattr(cfg.MODEL.SWINV2, "PATCH_SIZE"):
        cfg.MODEL.SWINV2.PATCH_SIZE = 4
    if not hasattr(cfg.MODEL.SWINV2, "IN_CHANS"):
        cfg.MODEL.SWINV2.IN_CHANS = 3
    if not hasattr(cfg.MODEL.SWINV2, "EMBED_DIM"):
        cfg.MODEL.SWINV2.EMBED_DIM = 96
    if not hasattr(cfg.MODEL.SWINV2, "DEPTHS"):
        cfg.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
    if not hasattr(cfg.MODEL.SWINV2, "NUM_HEADS"):
        cfg.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
    if not hasattr(cfg.MODEL.SWINV2, "WINDOW_SIZE"):
        cfg.MODEL.SWINV2.WINDOW_SIZE = 8
    if not hasattr(cfg.MODEL.SWINV2, "MLP_RATIO"):
        cfg.MODEL.SWINV2.MLP_RATIO = 4.0
    if not hasattr(cfg.MODEL.SWINV2, "QKV_BIAS"):
        cfg.MODEL.SWINV2.QKV_BIAS = True
    if not hasattr(cfg.MODEL.SWINV2, "APE"):
        cfg.MODEL.SWINV2.APE = False
    if not hasattr(cfg.MODEL.SWINV2, "PATCH_NORM"):
        cfg.MODEL.SWINV2.PATCH_NORM = True
    if not hasattr(cfg.MODEL.SWINV2, "PRETRAINED_WINDOW_SIZES"):
        cfg.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]


import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import build_model
from .PCAG import PCAGFusionForThreeModalities, PCAGFusionForTwoModalities
from yacs.config import CfgNode
import pandas as pd


# Clinical Branch
class ClinicalBranch(nn.Module):
    def __init__(self,num_features=4, dim=512):
        super().__init__()
        self.feature_embed = nn.Parameter(
        torch.randn(num_features, dim)
        )
    def forward(self, x):
        # x：[B, 4, 1], 4是临床特征的个数, 1是维数.x可以预处理时对年龄进行norm.其余不管
        x = x * self.feature_embed.unsqueeze(0)  # [B, 4, 1] * [4, dim] → [B, 4, dim]
        return x

# Radiomics Branch
class RadiomicsBranch(nn.Module):
    def __init__(self, num_features=930, dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.feature_embed = nn.Parameter(torch.randn(num_features, dim) * 0.02)  #本质是positional embedding
        self.value_proj = nn.Linear(1, dim)

    def forward(self, x):
        # x: [B, 930, 1]
        x = x.squeeze(-1)          # [B, 930]
        x = self.norm(x)
        x = x.unsqueeze(-1)        # [B, 930, 1]

        value_emb = self.value_proj(x)              # [B, 930, dim]
        feat_emb = self.feature_embed.unsqueeze(0)  # [1, 930, dim]

        return value_emb + feat_emb                 # [B, 930, dim]
    
# Multi-scale Swin Feature Extractor
class SwinFeatureExtractor(nn.Module):
    """
    Extract SwinT features.
    - multi_scale=True:  return [B, 4, 768] (4 stage features, pooled + projected)
    - multi_scale=False: return [B, 768] (final stage only)
    """

    def __init__(self, backbone, drop_out=0.2, multi_scale=True):
        super().__init__()

        self.backbone = backbone
        self.multi_scale = multi_scale
        if multi_scale:
            self.linear1 = nn.Linear(96, 768)
            self.linear2 = nn.Linear(192, 768)
            self.linear3 = nn.Linear(384, 768)
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(drop_out)
        

    def pool_feature(self, x):
        """
        x:
            [B, L, C]
        """

        B, L, C = x.shape

        H = W = int(L ** 0.5)

        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = F.adaptive_avg_pool2d(x, 1)

        x = x.flatten(1)

        return x

    def forward(self, x):
        x = self.backbone.patch_embed(x)

        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed

        x = self.backbone.pos_drop(x)

        features = []

        for layer in self.backbone.layers:
            for blk in layer.blocks:
                x = blk(x)

            pooled = self.pool_feature(x)
            features.append(pooled)

            if layer.downsample is not None:
                x = layer.downsample(x)

        if self.multi_scale:
            features[0] = self.linear1(features[0])
            features[1] = self.linear2(features[1])
            features[2] = self.linear3(features[2])
            multi_scale_img_feature = torch.stack(features, dim=1)
            multi_scale_img_feature = self.norm(multi_scale_img_feature)
            multi_scale_img_feature = self.dropout(multi_scale_img_feature)
            return multi_scale_img_feature
        else:
            feat = self.norm(features[-1])      # only stage 4, already 768
            feat = self.dropout(feat)
            return feat                         # [B, 768]

def _interpolate_pos_embed(state_dict, target_pos_embed):
    """Interpolate pos_embed in state_dict to match target spatial grid."""
    key = None
    for k in state_dict:
        if k.endswith("pos_embed"):
            key = k
            break
    if key is None:
        return
    src = state_dict[key]  # [1, N_src, D]
    tgt = target_pos_embed.data  # [1, N_tgt, D]
    if src.shape == tgt.shape:
        return
    # CLS token + patch tokens
    patch_src = src[:, 1:, :]
    cls_tgt = tgt[:, :1, :]
    old_grid = int(patch_src.shape[1] ** 0.5)
    new_grid = int((tgt.shape[1] - 1) ** 0.5)
    patch_src = patch_src.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    patch_tgt = F.interpolate(patch_src, size=(new_grid, new_grid), mode="bicubic")
    patch_tgt = patch_tgt.permute(0, 2, 3, 1).reshape(1, -1, tgt.shape[-1])
    state_dict[key] = torch.cat([cls_tgt, patch_tgt], dim=1)


# Multi-scale DINOv2 Feature Extractor
class DINOv2FeatureExtractor(nn.Module):
    """Extract intermediate DINOv2 block outputs and stack as multi-scale features.

    Captures patch tokens from blocks {3, 6, 9, 12} (1-indexed), pools each into a
    single vector, and stacks them → [B, 4, 768].

    Mimics SwinFeatureExtractor's output convention so the rest of PCAG fusion
    can consume either backbone's features transparently.
    """

    def __init__(self, model_name="vit_base_patch14_dinov2", img_size=256, drop_out=0.2,
                 pretrained=True, pretrained_path=None):
        super().__init__()
        import timm

        backbone = timm.create_model(model_name, pretrained=False, img_size=img_size)
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop
        # Store blocks as a plain Python attribute (not via nn.Module.__setattr__)
        # so they can later be re-parented under BrafSwin.backbone without
        # breaking this extractor's forward().
        object.__setattr__(self, '_blocks', backbone.blocks)
        self.embed_dim = backbone.embed_dim    # 768 for ViT-B
        self.img_size = img_size

        # Load pretrained weights
        if pretrained:
            if pretrained_path is not None:
                state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
                # Interpolate pos_embed if image size differs from pretrained (e.g. 256 vs 518)
                _interpolate_pos_embed(state_dict, self.pos_embed)
                self.load_state_dict(state_dict, strict=False)
                print(f'Loaded DINOv2 pretrained from: {pretrained_path}')
            else:
                backbone_pretrained = timm.create_model(model_name, pretrained=True, img_size=img_size)
                self.load_state_dict(backbone_pretrained.state_dict(), strict=False)
                del backbone_pretrained
                print(f'Loaded DINOv2 pretrained from timm hub')

        # Project each captured layer into a common dim (768)
        self.linear3 = nn.Linear(self.embed_dim, 768)
        self.linear6 = nn.Linear(self.embed_dim, 768)
        self.linear9 = nn.Linear(self.embed_dim, 768)
        self.linear12 = nn.Linear(self.embed_dim, 768)

        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(drop_out)

        # Blocks to capture: layers 3, 6, 9, 12 (1-indexed)
        self._capture_indices = {2, 5, 8, 11}

    def pool_feature(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N_patches, C] → adaptive avg pool → [B, C]"""
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)                                     # [B, N, C]
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)  # prepend CLS
        x = x + self.pos_embed
        x = self.pos_drop(x)

        captured = []
        for i, blk in enumerate(self._blocks):
            x = blk(x)
            if i in self._capture_indices:
                patch_tokens = x[:, 1:, :]        # exclude CLS token
                pooled = self.pool_feature(patch_tokens)
                captured.append(pooled)

        # captured[0]=layer3, captured[1]=layer6, captured[2]=layer9, captured[3]=layer12
        captured[0] = self.linear3(captured[0])   # → [B, 768]
        captured[1] = self.linear6(captured[1])
        captured[2] = self.linear9(captured[2])
        captured[3] = self.linear12(captured[3])

        multi_scale_feat = torch.stack(captured, dim=1)  # [B, 4, 768]
        multi_scale_feat = self.norm(multi_scale_feat)
        multi_scale_feat = self.dropout(multi_scale_feat)
        return multi_scale_feat


def _remap_timm_swin_keys(state_dict):
    """Remap timm SwinV2 state_dict keys → official SwinTransformer layout.

    Key differences:
      - timm layers.{1,2,3}.downsample → official layers.{0,1,2}.downsample
        (timm puts PatchMerging at start of next stage; official puts it at end of current)
      - timm head.fc.* → official head.* (we drop head anyway)
    """
    import re
    new_dict = {}
    for k, v in state_dict.items():
        m = re.match(r'layers\.(\d+)\.downsample\.(.*)', k)
        if m:
            old_idx = int(m.group(1))
            new_idx = old_idx - 1
            k = f'layers.{new_idx}.downsample.{m.group(2)}'
        elif k.startswith('head.fc.'):
            k = k.replace('head.fc.', 'head.')
        new_dict[k] = v
    return new_dict


# Main BrafSwin (Clinical + Radiomics only, no image branch)
class BrafSwinClinicalRadiomics(nn.Module):
    """PCAG fusion of Clinical Branch + Radiomics Branch only.

    Drops the image backbone entirely.  Useful as a baseline to measure the
    contribution of ultrasound images.
    """

    def __init__(self,
                 pcag_dim=384,
                 rad_embed_dim=128,
                 clinical_embed_dim=128,
                 clinical_num_features=4,
                 num_classes=2,
                 pre_gate_mode='tanh',
                 drop_out=0.2):
        super().__init__()

        self.rad_branch = RadiomicsBranch(dim=rad_embed_dim)
        self.clin_branch = ClinicalBranch(num_features=clinical_num_features, dim=clinical_embed_dim)

        self.fusion = PCAGFusionForTwoModalities(
            in_dim1=rad_embed_dim,
            in_dim2=clinical_embed_dim,
            dim=pcag_dim,
            num_classes=num_classes,
            pre_gate_mode=pre_gate_mode,
            dropout=drop_out,
        )

        self.mutation_head_mal = nn.Linear(pcag_dim, 2)
        self.mutation_head_benign = nn.Linear(pcag_dim, 2)

    def forward(self, rad_features, clin_features, return_feature=False):
        rad_feat = self.rad_branch(rad_features)      # [B, 930, D_rad]
        clin_feat = self.clin_branch(clin_features)   # [B, 4, D_clin]

        out = self.fusion(rad_feat, clin_feat, return_output_dataclass=True)
        mal_logits = out.logits
        shared_feat = out.joint

        mut_mal_logits = self.mutation_head_mal(shared_feat)
        mut_benign_logits = self.mutation_head_benign(shared_feat)

        if return_feature:
            return mal_logits, mut_mal_logits, mut_benign_logits, shared_feat
        return mal_logits, mut_mal_logits, mut_benign_logits


# Main BrafSwin (Image + Clinical only, no radiomics, no TI-RADS)
class BrafSwinImageClinical(nn.Module):
    """PCAG fusion of Image Branch + Clinical Branch (3 features, no TI-RADS).

    Drops radiomics entirely and removes TI-RADS from clinical features.
    Useful for measuring clinical contribution without TI-RADS leakage.
    """

    def __init__(self,
                 config="configs/swinv2/swinv2_tiny_patch4_window8_256.yaml",
                 img_size=256,
                 in_chans=3,
                 pretrained=True,
                 pretrained_model='swinv2_tiny_patch4_window8_256.pth',
                 backbone_type="swin",
                 dinov2_model_name="vit_base_patch14_dinov2",
                 pcag_dim=384,
                 clinical_embed_dim=128,
                 clinical_num_features=3,
                 num_classes=2,
                 pre_gate_mode='tanh',
                 drop_out=0.2,
                 multi_scale=True):
        super().__init__()
        self.backbone_type = backbone_type
        self.multi_scale = multi_scale

        # --- Image backbone ---
        if backbone_type == "dinov2":
            self.image_branch = DINOv2FeatureExtractor(
                model_name=dinov2_model_name, img_size=img_size, drop_out=drop_out,
                pretrained=pretrained, pretrained_path=pretrained_model)
            self.backbone = nn.Module()
            self.backbone.blocks = self.image_branch._blocks
            print(f'Loaded DINOv2 backbone: {dinov2_model_name}')
        else:
            if isinstance(config, str):
                with open(config, "r") as f:
                    cfg_dict = __import__("yaml").safe_load(f)
                config = CfgNode(cfg_dict)
            _ensure_config_defaults(config)
            self.backbone = build_model(config)
            self.backbone.head = nn.Identity()

            if pretrained:
                checkpoint = torch.load(pretrained_model, map_location='cpu', weights_only=False)
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                if (not any(k.startswith('layers.0.downsample') for k in checkpoint.keys())
                        and any(k.startswith('layers.1.downsample') for k in checkpoint.keys())):
                    checkpoint = _remap_timm_swin_keys(checkpoint)
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print('Loaded pretrained SwinV2 weight')
                print(msg)

            self.image_branch = SwinFeatureExtractor(self.backbone, drop_out=drop_out, multi_scale=multi_scale)

        # Clinical: [B, 3, 1] → [B, 3, clinical_embed_dim]  (no TI-RADS)
        self.clin_branch = ClinicalBranch(num_features=clinical_num_features, dim=clinical_embed_dim)

        # --- PCAG cross-modal fusion (2 modalities) ---
        self.fusion = PCAGFusionForTwoModalities(
            in_dim1=768,               # image token dim
            in_dim2=clinical_embed_dim,
            dim=pcag_dim,
            num_classes=2,
            pre_gate_mode=pre_gate_mode,
            dropout=drop_out,
        )
        self.pcag_dim = pcag_dim

        # --- Mutation Heads ---
        self.mutation_head_mal = nn.Linear(pcag_dim, 2)
        self.mutation_head_benign = nn.Linear(pcag_dim, 2)

    def forward(self, roi, clin_features, return_feature=False):
        img_feat = self.image_branch(roi)           # [B, 4/1, 768]
        clin_feat = self.clin_branch(clin_features)  # [B, 3, D_clin]

        if not self.multi_scale:
            img_feat = img_feat.unsqueeze(1)         # [B, 768] → [B, 1, 768]

        out = self.fusion(img_feat, clin_feat, return_output_dataclass=True)
        mal_logits = out.logits
        shared_feat = out.joint

        mut_mal_logits = self.mutation_head_mal(shared_feat)
        mut_benign_logits = self.mutation_head_benign(shared_feat)

        if return_feature:
            return mal_logits, mut_mal_logits, mut_benign_logits, shared_feat
        return mal_logits, mut_mal_logits, mut_benign_logits


# Main BrafSwin (all three modalities)
class BrafSwin(nn.Module):

    def __init__(self,
                 config="configs/swinv2/swinv2_tiny_patch4_window8_256.yaml",
                 img_size=256,
                 in_chans=3,
                 pretrained=True,
                 pretrained_model='swinv2_tiny_patch4_window8_256.pth',
                 backbone_type="swin",
                 dinov2_model_name="vit_base_patch14_dinov2",
                 pcag_dim=384,
                 rad_embed_dim=128,
                 clinical_embed_dim=128,
                 clinical_num_features=4,
                 num_classes=2,
                 pre_gate_mode='tanh',
                 drop_out=0.2):

        super().__init__()
        self.backbone_type = backbone_type

        # --- Image backbone ---
        if backbone_type == "dinov2":
            # DINOv2: timm handles pretrained loading internally
            self.image_branch = DINOv2FeatureExtractor(
                model_name=dinov2_model_name, img_size=img_size, drop_out=drop_out,
                pretrained=pretrained, pretrained_path=pretrained_model)
            # Wrap blocks as nn.Module so named_parameters() → backbone.blocks.*
            self.backbone = nn.Module()
            self.backbone.blocks = self.image_branch._blocks
            print(f'Loaded DINOv2 backbone: {dinov2_model_name}')

        else:
            # --- SwinV2 backbone ---
            if isinstance(config, str):
                with open(config, "r") as f:
                    cfg_dict = __import__("yaml").safe_load(f)
                config = CfgNode(cfg_dict)
            _ensure_config_defaults(config)
            self.backbone = build_model(config)
            self.backbone.head = nn.Identity()

            if pretrained:
                checkpoint = torch.load(pretrained_model, map_location='cpu', weights_only=False)
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                # Detect timm-sourced SwinV2 (no layers.0.downsample, has layers.1.downsample)
                if (not any(k.startswith('layers.0.downsample') for k in checkpoint.keys())
                        and any(k.startswith('layers.1.downsample') for k in checkpoint.keys())):
                    checkpoint = _remap_timm_swin_keys(checkpoint)
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print('Loaded pretrained SwinV2 weight')
                print(msg)

            self.image_branch = SwinFeatureExtractor(self.backbone, drop_out=drop_out)

        # Radiomics:  [B, 930, 1] → [B, 930, rad_embed_dim]
        self.rad_branch = RadiomicsBranch(dim=rad_embed_dim)

        # Clinical:   [B, 4, 1] → [B, 4, clinical_embed_dim]
        self.clin_branch = ClinicalBranch(num_features=clinical_num_features, dim=clinical_embed_dim)

        # --- PCAG cross-modal fusion (Malignancy Head = PCAG internal classifier) ---
        self.fusion = PCAGFusionForThreeModalities(
            in_dim1=768,               # image token dim
            in_dim2=rad_embed_dim,     # radiomics token dim
            in_dim3=clinical_embed_dim,
            dim=pcag_dim,
            num_classes=2,             # malignancy: benign / malignant
            pre_gate_mode=pre_gate_mode,
            dropout=drop_out,
        )
        self.pcag_dim = pcag_dim
        
        # --- Mutation Heads (built on Shared Feature) ---
        self.mutation_head_mal = nn.Linear(pcag_dim, 2)     # mutation yes/no for malignant

        self.mutation_head_benign = nn.Linear(pcag_dim, 2)
       # mutation yes/no for benign


    def forward(self, roi, rad_features, clin_features, return_feature=False):
        img_feat   = self.image_branch(roi)           # [B, 4, 768]
        rad_feat   = self.rad_branch(rad_features)    # [B, 930, D_rad]
        clin_feat  = self.clin_branch(clin_features)  # [B, 4, D_clin]

        out = self.fusion(img_feat, rad_feat, clin_feat, return_extras=True)
        mal_logits  = out["logits"]         # [B, 2]
        shared_feat = out["joint"]          # [B, pcag_dim]

        mut_mal_logits    = self.mutation_head_mal(shared_feat)     # [B, 2]
        mut_benign_logits = self.mutation_head_benign(shared_feat)  # [B, 2]

        if return_feature:
            return mal_logits, mut_mal_logits, mut_benign_logits, shared_feat
        return mal_logits, mut_mal_logits, mut_benign_logits






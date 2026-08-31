# BrafSwinT — 多模态 BRAF 突变预测方法

## 任务定义

从甲状腺结节超声数据中同时预测 **良恶性（Malignancy）** 和 **BRAF 突变状态（Mutation）**，通过层次化路由（hierarchical routing）将任务解耦为三个子任务：

- **Malignancy Head**：良性 vs 恶性
- **Mutation Head (Malignant)**：恶性结节中的 BRAF 突变/未突变
- **Mutation Head (Benign)**：良性结节中的 BRAF 突变/未突变

最终输出 4 类：BN（良性+无突变）、BM（良性+有突变）、MN（恶性+无突变）、MM（恶性+有突变）。

---

## 模型架构

```
超声图像 (256×256)  ──→ SwinV2-Tiny/DINOv2-B ──→ 多尺度特征 [B, 4, 768]
Radiomics (930维)   ──→ LayerNorm + Embedding ──→ [B, 930, 128]
临床特征 (3维)       ──→ Learnable Embedding   ──→ [B, 3, 128]
                                 │
                  ┌──────────────┼──────────────┐
                  │         PCAG Fusion         │
                  │  (Pre-Gated Cross-Attention │
                  │    + Contextual Attention    │
                  │          Gate)               │
                  └──────────────┼──────────────┘
                                 │
                          Shared Feature [B, 384]
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
      Malignancy Head     Mutation Head        Mutation Head
      (benign/mal)        (malignant分支)       (benign分支)
      [B, 2]              [B, 2]               [B, 2]
```

### 1. 图像分支（Image Backbone）

支持两种 backbone，均提取 4 个尺度的层级特征：

| Backbone | 参数量 | 输入 | 输出 | 关键处理 |
|----------|--------|------|------|---------|
| SwinV2-Tiny | 27.6M | 256×256×3 | [B, 4, 768] | 4 个 stage 各取 pooled 特征，线性投影到 768 维 |
| DINOv2-B (ViT-B) | 85.8M | 256×256×3 | [B, 4, 768] | 取第 3/6/9/12 block 的 patch tokens，adaptive pooling，线性投影到 768 维 |

- SwinV2 使用微软官方实现 (`SwinTransformer.build_model`) 加载 `swinv2_tiny_patch4_window8_256.pth` 预训练权重
- DINOv2 使用 timm 加载 `dinov2_vit_base_patch14.pth`，对 pos_embed 做 bicubic 插值以适配 256×256 分辨率
- 预训练权重 key 映射：timm SwinV2 的 `layers.{1,2,3}.downsample` → 官方的 `layers.{0,1,2}.downsample`
- 训练时冻结全部 backbone stages（`freeze_indices=0,1,2,3`）

### 2. Radiomics 分支

- 输入：930 维 PyRadiomics 手工特征 [B, 930, 1]
- 处理：LayerNorm → Linear 投影（1→128） + 可学习 positional embedding [930, 128]
- 输出：[B, 930, 128]（每个 radiomics 特征为一个 token）

### 3. Clinical 分支

- 输入：4 维临床特征 [B, 4, 1]，含年龄（归一化）、性别、桥本氏甲状腺炎、TI-RADS 分级
- 处理：每个特征乘以可学习的特征嵌入 [4, dim]，输出 [B, 4, dim]
- 输出维度 dim 默认 128

### 4. PCAG 跨模态融合

基于 "Pre-gating and Contextual Attention Gate" (Neural Networks, 2024)：

1. **三模态 Q/K/V 投影**：每个模态（图像 768 维, radiomics 128 维, clinical 128 维）投影到共享维度 384
2. **Pre-Gated Cross-Attention**：每对模态间计算交叉注意力，注意力矩阵用 `tanh` 预门控 `A_gated = A * P(A)`，其中 `P(A) = (tanh(Q)@tanh(K)^T+1)/2`
3. **Contextual Attention Gate (CAG)**：`G = ReLU(W_h * Qhat + W_q * Q + b)`，`E = ReLU(W_e * Qhat + b)`，`C = Norm(E) ⊙ Norm(G)`
4. **加性融合**：Inter-modal 特征 + Intra-modal 特征求和 → Shared Feature [B, 384]
5. **模式间权重**：每个模态学习融合另两个模态信息时的权重（learnable `qhat_fusion_weights`）

### 5. 层次化分类头

- **Malignancy Head**：PCAG 内部分类器，Linear(384→2)，预测良性/恶性
- **Mutation Head (Malignant)**：Linear(384→2)，预测恶性样本的 BRAF 突变
- **Mutation Head (Benign)**：Linear(384→2)，预测良性样本的 BRAF 突变

---

## 训练策略

### 数据划分
- **数据集**：1665 例甲状腺结节，含超声 ROI 图像 + 930 维 radiomics + 4 维临床特征
- **5 折交叉验证**：`StratifiedKFold` 按 malignancy × mutation 联合标签分层
- **训练/验证切分**：每折内部按 12.5% 随机分出独立验证集

### 增强
- **图像增强**：RandomHorizontalFlip(p=0.5), RandomAffine(±10°, scale 0.95-1.05), ColorJitter(0.15), GaussianBlur(p=0.2)
- **MixUp**：α=0.2，混合图像 + 双标签损失
- **CutMix**：α=1.0

### 优化
- **损失函数**：Focal Loss (γ=1.0)，三个头独立计算再求和 `Loss = L_mal + L_mut_mal + L_mut_benign`
- **优化器**：AdamW，backbone LR = head LR × 0.1，weight_decay=1e-2
- **学习率调度**：Cosine Annealing，warmup 3 epoch 线性升温，最低 LR = initial × 0.01
- **混合精度**：`torch.cuda.amp.GradScaler`
- **EMA**：ModelEMA (decay=0.999)，验证时使用 EMA 权重

### Scheduled Routing
- 训练前半段（前 10% epochs）仅用真实 malignancy 标签路由 mutation head
- 后续逐步引入预测路由（route_prob 线性从 0 增至 0.5），使训练逼近真实推理场景

### 评估
- **主指标**：Malignancy AUC/ACC/F1/Sens/Spec，Mutation AUC/ACC/F1/Sens/Spec
- **4-Class 指标**：ACC_4c, F1_4c_weighted, AUC_4c_weighted（多类别 one-vs-rest AUC, average="weighted"）
- **4-Class 概率**：由三个头的输出联合计算 `P(BN)=(1-P_mal)(1-P_mut_benign)`, `P(BM)=(1-P_mal)P_mut_benign`, `P(MN)=P_mal(1-P_mut_mal)`, `P(MM)=P_mal·P_mut_mal`
- **Checkpoint 选择**：最佳 val_loss, 最佳 4c_AUC, 最佳 4c_ACC 三个版本分别保存

---

## 消融实验框架

| 消融实验 | 模型 | 模态 | 参数量 |
|----------|------|------|--------|
| Clinical-Only | 2层 MLP (hidden=64) | 临床 4 维 | ~5K |
| Clinical+Radiomics | BrafSwinClinicalRadiomics (PCAG 2-modal) | 临床 + Radiomics | 1.9M |
| 全模型 | BrafSwin (PCAG 3-modal) | 临床 + Radiomics + 图像 | 28M+ |

---

## 关键引文

- SwinV2: Liu et al., "Swin Transformer V2: Scaling Up Capacity and Resolution," CVPR 2022
- DINOv2: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," TMLR 2024
- PCAG: "Pre-gating and Contextual Attention Gate — A New Fusion Method for Multi-modal Data Tasks," Neural Networks 179 (2024) 106553
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017
- SupCon Loss (Stage 1): Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020

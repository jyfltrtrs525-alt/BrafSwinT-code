"""main_concat.py — BrafSwinConcatFusion single train/val/cal/test split (concat late fusion).

Architecture:
    US → SwinT V1 (torchvision) → pool → [B, 768] ─┐
    Clinical (age, gender, Hashimoto) → ClinicalBranch → mean → [B, 128] ─┤
                                                                            ├→ Concat → MLP → SharedFeat → MalHead
                                                                                         ├→ MutationHead_Mal
                                                                                         └→ MutationHead_Benign
"""

import os
import sys
import math
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

from st.models.brafswin_late import BrafSwinConcatFusion

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Dataset (Image + Clinical only, no radiomics)
# ---------------------------------------------------------------------------
class ImageClinicalDataset(Dataset):
    """Dataset returning image + 3 clinical features (no TI-RADS)."""

    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]

        img = Image.open(os.path.join(self.image_dir, f"{case_id}.png")).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Clinical: age_norm, gender, hashimoto (NO TI-RADS)
        clin = torch.tensor(
            [row["age_norm"], row["gender"], row["hashimoto"]],
            dtype=torch.float32,
        ).unsqueeze(-1)  # [3, 1]

        mal_label = int(row["malignancy"])
        mut_label = int(row["braf_mutation"])

        return img, clin, mal_label, mut_label, case_id


# ---------------------------------------------------------------------------
# Synthetic Image Dataset
# ---------------------------------------------------------------------------
SYNTH_DIR_MAP = {
    "benign_wt":      (0, 0),   # BN
    "benign_mut":     (0, 1),   # BM
    "malignant_wt":   (1, 0),   # MN
    "malignant_mut":  (1, 1),   # MM
}


class SyntheticDataset(Dataset):
    """Synthetic images with clinical features sampled from real train data of the same class."""

    def __init__(self, synth_dir, train_df, n_per_class, transform=None):
        """
        Args:
            synth_dir:   path to generated_images/ with benign_wt/ etc.
            train_df:    real training DataFrame (used to sample clinical features)
            n_per_class: dict mapping 4-class label (0=BN,1=BM,2=MN,3=MM) to count
            transform:   image transforms
        """
        self.transform = transform

        # Precompute clinical feature pools per class from train_df
        self.clin_pools = {}  # class_label → list of (age_norm, gender, hashimoto)
        for c4 in range(4):
            mal = c4 // 2
            mut = c4 % 2
            mask = (train_df["malignancy"] == mal) & (train_df["braf_mutation"] == mut)
            sub = train_df[mask]
            if len(sub) == 0:
                sub = train_df  # fallback: use all data
            self.clin_pools[c4] = sub[["age_norm", "gender", "hashimoto"]].values.astype(np.float32)

        # Collect image paths per class
        self.samples = []  # list of (img_path, c4_label)
        for subdir, (mal, mut) in SYNTH_DIR_MAP.items():
            c4 = mal * 2 + mut
            n = n_per_class.get(c4, 0)
            if n <= 0:
                continue
            subdir_path = os.path.join(synth_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            files = sorted(os.listdir(subdir_path))
            files = [f for f in files if f.endswith(".png")]
            selected = files[:n]
            for fname in selected:
                self.samples.append((os.path.join(subdir_path, fname), c4))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, c4 = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Randomly sample clinical features from the same class
        pool = self.clin_pools[c4]
        row_idx = np.random.randint(0, len(pool))
        clin = torch.tensor(pool[row_idx], dtype=torch.float32).unsqueeze(-1)  # [3, 1]

        mal_label = c4 // 2
        mut_label = c4 % 2
        case_id = os.path.basename(img_path)

        return img, clin, mal_label, mut_label, case_id


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if inputs.ndim == 2 and inputs.size(1) == 1:
            inputs = inputs.view(-1)
        if targets.ndim == 2 and targets.size(1) == 1:
            targets = targets.view(-1)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = (1 - pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------
def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Pull features of the same class together, push different classes apart."""
    features = F.normalize(features, dim=1)
    labels = labels.view(-1, 1)

    mask = torch.eq(labels, labels.T).float().to(features.device)
    logits = torch.matmul(features, features.T) / temperature

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

    loss = -mean_log_prob_pos.mean()
    return loss


# ---------------------------------------------------------------------------
# Cosine Annealing Scheduler
# ---------------------------------------------------------------------------
class CosineScheduler:
    def __init__(self, optimizer, total_epochs, eta_min, warmup_epochs=0):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = max(1, total_epochs - warmup_epochs)
        self.current_epoch = 0
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
            if warmup_epochs > 0:
                group["lr"] = eta_min

    def step(self):
        self.current_epoch += 1
        if self.warmup_epochs > 0 and self.current_epoch <= self.warmup_epochs:
            alpha = self.current_epoch / max(1, self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / self.cosine_epochs
            alpha = (1 + math.cos(math.pi * progress)) / 2
        for group in self.optimizer.param_groups:
            group["lr"] = self.eta_min + (group["initial_lr"] - self.eta_min) * alpha

    def add_param_group(self, param_group):
        param_group.setdefault("initial_lr", param_group["lr"])
        self.optimizer.add_param_group(param_group)

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(384, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_binary_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = cm[0, 0] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    tp = cm[1, 1] if cm.shape == (2, 2) else 0

    metrics = {
        "ACC": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }
    if y_prob is not None:
        try:
            metrics["AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["AUC"] = float("nan")
    return metrics


def compute_four_class_metrics(mal_true, mal_pred, mut_true, mut_pred,
                               mal_prob=None, mut_mal_prob=None,
                               mut_benign_prob=None):
    y_true_4c = np.array(mal_true) * 2 + np.array(mut_true)
    y_pred_4c = np.array(mal_pred) * 2 + np.array(mut_pred)

    metrics = {
        "ACC_4c": accuracy_score(y_true_4c, y_pred_4c),
        "F1_4c_weighted": f1_score(y_true_4c, y_pred_4c, average="weighted", zero_division=0),
    }

    class_names = ["BN", "BM", "MN", "MM"]
    for c in range(4):
        yt = (y_true_4c == c).astype(int)
        yp = (y_pred_4c == c).astype(int)
        metrics[f"Sens_{class_names[c]}"] = recall_score(yt, yp, zero_division=0)
        metrics[f"Spec_{class_names[c]}"] = recall_score(yt, yp, pos_label=0, zero_division=0)

    if mal_prob is not None and mut_mal_prob is not None and mut_benign_prob is not None:
        try:
            mal_prob = np.array(mal_prob)
            mut_mal_prob = np.array(mut_mal_prob)
            mut_benign_prob = np.array(mut_benign_prob)
            prob_4c = np.zeros((len(mal_true), 4), dtype=np.float64)
            prob_4c[:, 0] = (1 - mal_prob) * (1 - mut_benign_prob)
            prob_4c[:, 1] = (1 - mal_prob) * mut_benign_prob
            prob_4c[:, 2] = mal_prob * (1 - mut_mal_prob)
            prob_4c[:, 3] = mal_prob * mut_mal_prob
            prob_4c = prob_4c / prob_4c.sum(axis=1, keepdims=True)
            metrics["AUC_4c_weighted"] = roc_auc_score(
                y_true_4c, prob_4c, multi_class="ovr", average="weighted"
            )
        except ValueError:
            metrics["AUC_4c_weighted"] = float("nan")

    return metrics


def route_mutation_predictions(mal_preds, mut_mal_logits, mut_benign_logits):
    mal_preds = np.array(mal_preds)
    mut_mal_prob = torch.sigmoid(torch.tensor(mut_mal_logits)).squeeze(-1).numpy()
    mut_benign_prob = torch.sigmoid(torch.tensor(mut_benign_logits)).squeeze(-1).numpy()

    mut_probs = np.where(mal_preds == 1, mut_mal_prob, mut_benign_prob)
    mut_preds = (mut_probs >= 0.5).astype(int)
    return mut_probs, mut_preds, mut_mal_prob, mut_benign_prob


# ---------------------------------------------------------------------------
# Conformal Risk Control (CRC)
# ---------------------------------------------------------------------------
def select_crc_threshold(probs, labels, risk_level, grid_size=10001,
                         loss_type="classification_error"):
    """Select threshold via Conformal Risk Control.

    Scans prob_threshold t ∈ [0, 1], chooses the smallest t such that
    conservative_risk = (sum(loss) + 1) / (n_cal + 1) <= risk_level.

    Returns:
        tau              = 1 - prob_threshold  (math convention)
        prob_threshold   = probability cutoff for pred=1
        conservative_risk / empirical_risk / n_cal / risk_bound / valid
    """
    n_cal = len(probs)
    result = {
        "tau": 0.0,
        "prob_threshold": 0.5,
        "conservative_risk": 1.0,
        "empirical_risk": 1.0,
        "risk_level": risk_level,
        "n_cal": n_cal,
        "risk_bound": None if n_cal == 0 else ((n_cal + 1) / n_cal) * risk_level - 1 / n_cal,
        "valid": False,
    }

    if n_cal == 0:
        result["tau"] = 0.5
        result["warning"] = "n_cal == 0, returning default threshold=0.5"
        return result

    best_t = 2.0
    best_emp = 1.0
    best_con = 2.0

    min_con = 2.0
    min_con_t = 0.5
    min_con_emp = 1.0

    for i in range(grid_size):
        t = i / (grid_size - 1) if grid_size > 1 else 0.5
        preds = (probs >= t).astype(int)
        loss_arr = (preds != np.array(labels, dtype=int)).astype(np.float32)
        emp_risk = loss_arr.mean()
        con_risk = (loss_arr.sum() + 1) / (n_cal + 1)

        if con_risk < min_con:
            min_con = con_risk
            min_con_t = t
            min_con_emp = emp_risk

        if con_risk <= risk_level and t < best_t:
            best_t = t
            best_emp = emp_risk
            best_con = con_risk

    if best_t > 1.0:
        result["prob_threshold"] = 0.5
        result["tau"] = 0.0
        result["empirical_risk"] = float(min_con_emp)
        result["conservative_risk"] = float(min_con)
        result["valid"] = False
        result["warning"] = (
            f"No threshold satisfies risk <= {risk_level}. "
            f"min conservative_risk = {min_con:.6f} at t = {min_con_t:.4f} "
            f"(gap = {min_con - risk_level:.6f})"
        )
    else:
        result["prob_threshold"] = float(best_t)
        result["tau"] = float(1.0 - best_t)
        result["empirical_risk"] = float(best_emp)
        result["conservative_risk"] = float(best_con)
        result["valid"] = True

    return result


@torch.no_grad()
def calibrate_crc_thresholds(model, loader_u_cal, loader_d_cal, device,
                              beta, alpha_M, alpha_B,
                              grid_size=10001, save_path=None):
    """Calibrate three CRC thresholds: tau_mag, lambda_M, lambda_B.

    - loader_u_cal: calibration set for malignancy threshold tau_mag
    - loader_d_cal: calibration set for mutation thresholds (λ_M, λ_B),
                    routed by tau_mag prediction.
    """
    import json

    model.eval()

    # --- Step 1: tau_mag from U_cal ---
    u_probs, u_labels = [], []
    for img, clin, mal_label, mut_label, _ in tqdm(loader_u_cal, desc="CRC-U_cal", leave=False):
        img, clin = img.to(device), clin.to(device)
        mal_logits, _, _ = model(img, clin)
        u_probs.append(torch.sigmoid(mal_logits).squeeze(-1).cpu().numpy())
        u_labels.append(mal_label.numpy())
    u_probs = np.concatenate(u_probs)
    u_labels = np.concatenate(u_labels)

    tau_mag_result = select_crc_threshold(u_probs, u_labels, beta, grid_size=grid_size)

    # --- Step 2: lambda_M / lambda_B from D_cal routed by tau_mag ---
    d_mut_mal_probs, d_mut_mal_labels = [], []
    d_mut_benign_probs, d_mut_benign_labels = [], []

    for img, clin, mal_label, mut_label, _ in tqdm(loader_d_cal, desc="CRC-D_cal", leave=False):
        img, clin = img.to(device), clin.to(device)
        mal_logits, mut_mal_logits, mut_benign_logits = model(img, clin)
        mal_prob = torch.sigmoid(mal_logits).squeeze(-1).cpu().numpy()
        mut_mal_prob = torch.sigmoid(mut_mal_logits).squeeze(-1).cpu().numpy()
        mut_benign_prob = torch.sigmoid(mut_benign_logits).squeeze(-1).cpu().numpy()

        mask_mal = mal_prob >= tau_mag_result["prob_threshold"]
        mask_benign = ~mask_mal

        if mask_mal.any():
            d_mut_mal_probs.append(mut_mal_prob[mask_mal])
            d_mut_mal_labels.append(mut_label.numpy()[mask_mal])
        if mask_benign.any():
            d_mut_benign_probs.append(mut_benign_prob[mask_benign])
            d_mut_benign_labels.append(mut_label.numpy()[mask_benign])

    lambda_M_result = select_crc_threshold(
        np.concatenate(d_mut_mal_probs) if d_mut_mal_probs else np.array([], dtype=np.float64),
        np.concatenate(d_mut_mal_labels) if d_mut_mal_labels else np.array([], dtype=np.int64),
        alpha_M, grid_size=grid_size,
    )
    lambda_B_result = select_crc_threshold(
        np.concatenate(d_mut_benign_probs) if d_mut_benign_probs else np.array([], dtype=np.float64),
        np.concatenate(d_mut_benign_labels) if d_mut_benign_labels else np.array([], dtype=np.int64),
        alpha_B, grid_size=grid_size,
    )

    crc_thresholds = {
        "tau_mag": tau_mag_result,
        "lambda_M": lambda_M_result,
        "lambda_B": lambda_B_result,
        "beta": beta,
        "alpha_M": alpha_M,
        "alpha_B": alpha_B,
    }

    # --- Log ---
    for name, res in [("tau_mag", tau_mag_result),
                      ("lambda_M", lambda_M_result),
                      ("lambda_B", lambda_B_result)]:
        print(f"  CRC {name}: prob_thresh={res['prob_threshold']:.6f}  "
              f"tau/lambda={res['tau']:.6f}  n_cal={res['n_cal']}  "
              f"con_risk={res['conservative_risk']:.6f}  "
              f"emp_risk={res['empirical_risk']:.6f}  valid={res['valid']}")

    # --- Save JSON ---
    if save_path is not None:

        def _serialize(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[k] = _serialize(v)
                elif isinstance(v, (np.integer,)):
                    out[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    out[k] = float(v)
                elif isinstance(v, np.ndarray):
                    out[k] = v.tolist()
                else:
                    out[k] = v
            return out

        with open(save_path, "w") as f:
            json.dump(_serialize(crc_thresholds), f, indent=2)
        print(f"  CRC thresholds saved to {save_path}")

    return crc_thresholds


@torch.no_grad()
def test_and_save_crc(model, loader, criterion, device, crc_thresholds, save_path):
    """Test with CRC thresholds. Same logic as test_and_save but using CRC cutoffs."""
    model.eval()

    tau_result = crc_thresholds["tau_mag"]
    lm_result = crc_thresholds["lambda_M"]
    lb_result = crc_thresholds["lambda_B"]

    t_mag = tau_result["prob_threshold"]
    t_mut_mal = lm_result["prob_threshold"]
    t_mut_benign = lb_result["prob_threshold"]

    total_loss = 0.0
    n_batches = len(loader)

    all_mal_logits, all_mut_mal_logits, all_mut_benign_logits = [], [], []
    all_mal_labels, all_mut_labels = [], []
    all_case_ids = []

    for img, clin, mal_label, mut_label, case_ids in tqdm(loader, desc="Test-CRC", leave=False):
        img, clin = img.to(device), clin.to(device)
        mal_label_dev = mal_label.to(device)
        mut_label_dev = mut_label.to(device)

        mal_logits, mut_mal_logits, mut_benign_logits = model(img, clin)

        mal_loss = criterion(mal_logits, mal_label_dev)
        mask_mal = (mal_label_dev == 1)
        mask_benign = (mal_label_dev == 0)
        mut_loss_mal = (
            criterion(mut_mal_logits[mask_mal], mut_label_dev[mask_mal])
            if mask_mal.any() else torch.tensor(0.0, device=device)
        )
        mut_loss_benign = (
            criterion(mut_benign_logits[mask_benign], mut_label_dev[mask_benign])
            if mask_benign.any() else torch.tensor(0.0, device=device)
        )
        loss = mal_loss + mut_loss_mal + mut_loss_benign
        total_loss += loss.item()

        all_mal_logits.append(mal_logits.cpu().numpy())
        all_mut_mal_logits.append(mut_mal_logits.cpu().numpy())
        all_mut_benign_logits.append(mut_benign_logits.cpu().numpy())
        all_mal_labels.append(mal_label.numpy())
        all_mut_labels.append(mut_label.numpy())
        all_case_ids.extend(case_ids)

    mal_logits = np.concatenate(all_mal_logits, axis=0)
    mut_mal_logits = np.concatenate(all_mut_mal_logits, axis=0)
    mut_benign_logits = np.concatenate(all_mut_benign_logits, axis=0)
    mal_labels = np.concatenate(all_mal_labels, axis=0)
    mut_labels = np.concatenate(all_mut_labels, axis=0)

    mal_prob = torch.sigmoid(torch.tensor(mal_logits)).squeeze(-1).numpy()
    mut_mal_prob = torch.sigmoid(torch.tensor(mut_mal_logits)).squeeze(-1).numpy()
    mut_benign_prob = torch.sigmoid(torch.tensor(mut_benign_logits)).squeeze(-1).numpy()

    # CRC predictions
    mal_pred_crc = (mal_prob >= t_mag).astype(int)
    mut_prob_crc = np.where(
        mal_pred_crc == 1,
        mut_mal_prob,
        mut_benign_prob,
    )
    mut_pred_crc = np.where(
        mal_pred_crc == 1,
        (mut_mal_prob >= t_mut_mal).astype(int),
        (mut_benign_prob >= t_mut_benign).astype(int),
    )

    # Standard (0.5) predictions for metrics comparison
    mal_pred_05 = (mal_logits.squeeze(-1) > 0).astype(int)
    mut_probs_05, mut_preds_05, _, _ = route_mutation_predictions(
        mal_pred_05, mut_mal_logits, mut_benign_logits
    )

    mal_metrics = compute_binary_metrics(mal_labels, mal_pred_crc, mal_prob)
    mut_metrics = compute_binary_metrics(mut_labels, mut_pred_crc, mut_prob_crc)
    four_c_metrics = compute_four_class_metrics(
        mal_labels, mal_pred_crc, mut_labels, mut_pred_crc,
        mal_prob=mal_prob,
        mut_mal_prob=mut_mal_prob,
        mut_benign_prob=mut_benign_prob,
    )

    df_pred = pd.DataFrame({
        "case_id": all_case_ids,
        "malignancy_true": mal_labels,
        "malignancy_pred_crc": mal_pred_crc,
        "malignancy_prob": mal_prob,
        "malignancy_pred_05": mal_pred_05,
        "mutation_true": mut_labels,
        "mutation_pred_crc": mut_pred_crc,
        "mutation_prob_crc": mut_prob_crc,
        "mutation_pred_05": mut_preds_05,
        "mutation_prob_05": mut_probs_05,
        "mut_mal_prob": mut_mal_prob,
        "mut_benign_prob": mut_benign_prob,
        "route_crc": np.where(mal_pred_crc == 1, "malignant", "benign"),
        "route_05": np.where(mal_pred_05 == 1, "malignant", "benign"),
        "mal_prob_threshold_crc": t_mag,
        "mut_mal_prob_threshold_crc": t_mut_mal,
        "mut_benign_prob_threshold_crc": t_mut_benign,
    })
    df_pred.to_csv(save_path, index=False)

    avg_loss = total_loss / n_batches
    return avg_loss, mal_metrics, mut_metrics, four_c_metrics


# ---------------------------------------------------------------------------
# Scheduled routing
# ---------------------------------------------------------------------------
def get_route_prob(epoch, total_epochs, start_ratio=0.5, max_prob=0.5):
    start_epoch = int(total_epochs * start_ratio)
    if epoch <= start_epoch:
        return 0.0
    progress = (epoch - start_epoch) / max(1, total_epochs - start_epoch)
    return max_prob * progress


# ---------------------------------------------------------------------------
# Train / Val / Test loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device,
                    epoch, total_epochs, scaler=None,
                    scheduled_routing=False,
                    route_start_ratio=0.5, route_max_prob=0.5,
                    ema=None, lambda_contrast=0.0,
                    class_weights=None):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_contrast_loss = 0.0
    n_batches = len(loader)

    p_pred_route = 0.0
    if scheduled_routing:
        p_pred_route = get_route_prob(epoch, total_epochs,
                                      start_ratio=route_start_ratio,
                                      max_prob=route_max_prob)

    use_contrast = lambda_contrast > 0
    use_cb = class_weights is not None

    for img, clin, mal_label, mut_label, _ in tqdm(loader, desc="Train", leave=False):
        img = img.to(device)
        clin = clin.to(device)
        mal_label_dev = mal_label.to(device)
        mut_label_dev = mut_label.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            if use_contrast:
                mal_logits, mut_mal_logits, mut_benign_logits, shared_feat = model(
                    img, clin, return_feature=True
                )
            else:
                mal_logits, mut_mal_logits, mut_benign_logits = model(img, clin)

            if use_cb:
                # Class-balanced: per-sample weight × focal loss
                mal_loss_each = criterion(mal_logits, mal_label_dev)
                mal_w = torch.where(
                    mal_label_dev.float().view(-1) == 1,
                    torch.tensor(float(class_weights["w_pos_m"]), device=device),
                    torch.tensor(float(class_weights["w_neg_m"]), device=device),
                )
                mal_loss = (mal_loss_each * mal_w).mean()
            else:
                mal_loss = criterion(mal_logits, mal_label_dev)

            if scheduled_routing and p_pred_route > 0:
                with torch.no_grad():
                    mal_pred = (mal_logits > 0).long().squeeze(1)
                use_pred = torch.rand_like(mal_label_dev.float()) < p_pred_route
                route_label = torch.where(use_pred, mal_pred, mal_label_dev)
            else:
                route_label = mal_label_dev

            mask_mal = (route_label == 1)
            mask_benign = (route_label == 0)

            mut_loss_mal = torch.tensor(0.0, device=device)
            mut_loss_benign = torch.tensor(0.0, device=device)

            if mask_mal.any():
                if use_cb:
                    mut_mal_each = criterion(mut_mal_logits[mask_mal], mut_label_dev[mask_mal])
                    mut_mal_w = torch.where(
                        mut_label_dev[mask_mal].float().view(-1) == 1,
                        torch.tensor(float(class_weights["w_pos_malignant"]), device=device),
                        torch.tensor(float(class_weights["w_neg_malignant"]), device=device),
                    )
                    mut_loss_mal = (mut_mal_each * mut_mal_w).mean()
                else:
                    mut_loss_mal = criterion(mut_mal_logits[mask_mal], mut_label_dev[mask_mal])

            if mask_benign.any():
                if use_cb:
                    mut_benign_each = criterion(mut_benign_logits[mask_benign], mut_label_dev[mask_benign])
                    mut_benign_w = torch.where(
                        mut_label_dev[mask_benign].float().view(-1) == 1,
                        torch.tensor(float(class_weights["w_pos_benign"]), device=device),
                        torch.tensor(float(class_weights["w_neg_benign"]), device=device),
                    )
                    mut_loss_benign = (mut_benign_each * mut_benign_w).mean()
                else:
                    mut_loss_benign = criterion(mut_benign_logits[mask_benign], mut_label_dev[mask_benign])

            cls_loss = mal_loss + mut_loss_mal + mut_loss_benign
            loss = cls_loss

            if use_contrast:
                contrast_loss = supervised_contrastive_loss(
                    shared_feat, mal_label_dev, temperature=0.1
                )
                loss = loss + lambda_contrast * contrast_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        if use_contrast:
            total_contrast_loss += contrast_loss.item()

    avg_loss = total_loss / n_batches
    if use_contrast:
        return avg_loss, total_cls_loss / n_batches, total_contrast_loss / n_batches
    return avg_loss, None, None


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    n_batches = len(loader)

    all_mal_logits = []
    all_mut_mal_logits = []
    all_mut_benign_logits = []
    all_mal_labels = []
    all_mut_labels = []

    for img, clin, mal_label, mut_label, _ in tqdm(loader, desc="Eval", leave=False):
        img = img.to(device)
        clin = clin.to(device)
        mal_label_dev = mal_label.to(device)
        mut_label_dev = mut_label.to(device)

        mal_logits, mut_mal_logits, mut_benign_logits = model(img, clin)

        mal_loss = criterion(mal_logits, mal_label_dev)
        mask_mal = (mal_label_dev == 1)
        mask_benign = (mal_label_dev == 0)
        mut_loss_mal = (
            criterion(mut_mal_logits[mask_mal], mut_label_dev[mask_mal])
            if mask_mal.any() else torch.tensor(0.0, device=device)
        )
        mut_loss_benign = (
            criterion(mut_benign_logits[mask_benign], mut_label_dev[mask_benign])
            if mask_benign.any() else torch.tensor(0.0, device=device)
        )
        loss = mal_loss + mut_loss_mal + mut_loss_benign
        total_loss += loss.item()

        all_mal_logits.append(mal_logits.cpu().numpy())
        all_mut_mal_logits.append(mut_mal_logits.cpu().numpy())
        all_mut_benign_logits.append(mut_benign_logits.cpu().numpy())
        all_mal_labels.append(mal_label.numpy())
        all_mut_labels.append(mut_label.numpy())

    mal_logits = np.concatenate(all_mal_logits, axis=0)
    mut_mal_logits = np.concatenate(all_mut_mal_logits, axis=0)
    mut_benign_logits = np.concatenate(all_mut_benign_logits, axis=0)
    mal_labels = np.concatenate(all_mal_labels, axis=0)
    mut_labels = np.concatenate(all_mut_labels, axis=0)

    mal_probs = torch.sigmoid(torch.tensor(mal_logits)).squeeze(-1).numpy()
    mal_preds = (mal_logits.squeeze(-1) > 0).astype(int)

    mut_probs, mut_preds, mut_mal_prob, mut_benign_prob = route_mutation_predictions(
        mal_preds, mut_mal_logits, mut_benign_logits
    )

    mal_metrics = compute_binary_metrics(mal_labels, mal_preds, mal_probs)
    mut_metrics = compute_binary_metrics(mut_labels, mut_preds, mut_probs)
    four_c_metrics = compute_four_class_metrics(
        mal_labels, mal_preds, mut_labels, mut_preds,
        mal_prob=mal_probs,
        mut_mal_prob=mut_mal_prob,
        mut_benign_prob=mut_benign_prob,
    )

    avg_loss = total_loss / n_batches
    return avg_loss, mal_metrics, mut_metrics, four_c_metrics


@torch.no_grad()
def test_and_save(model, loader, criterion, device, save_path):
    model.eval()

    total_loss = 0.0
    n_batches = len(loader)

    all_mal_logits = []
    all_mut_mal_logits = []
    all_mut_benign_logits = []
    all_mal_labels = []
    all_mut_labels = []
    all_case_ids = []

    for img, clin, mal_label, mut_label, case_ids in tqdm(loader, desc="Test", leave=False):
        img = img.to(device)
        clin = clin.to(device)
        mal_label_dev = mal_label.to(device)
        mut_label_dev = mut_label.to(device)

        mal_logits, mut_mal_logits, mut_benign_logits = model(img, clin)

        mal_loss = criterion(mal_logits, mal_label_dev)
        mask_mal = (mal_label_dev == 1)
        mask_benign = (mal_label_dev == 0)
        mut_loss_mal = (
            criterion(mut_mal_logits[mask_mal], mut_label_dev[mask_mal])
            if mask_mal.any() else torch.tensor(0.0, device=device)
        )
        mut_loss_benign = (
            criterion(mut_benign_logits[mask_benign], mut_label_dev[mask_benign])
            if mask_benign.any() else torch.tensor(0.0, device=device)
        )
        loss = mal_loss + mut_loss_mal + mut_loss_benign
        total_loss += loss.item()

        all_mal_logits.append(mal_logits.cpu().numpy())
        all_mut_mal_logits.append(mut_mal_logits.cpu().numpy())
        all_mut_benign_logits.append(mut_benign_logits.cpu().numpy())
        all_mal_labels.append(mal_label.numpy())
        all_mut_labels.append(mut_label.numpy())
        all_case_ids.extend(case_ids)

    mal_logits = np.concatenate(all_mal_logits, axis=0)
    mut_mal_logits = np.concatenate(all_mut_mal_logits, axis=0)
    mut_benign_logits = np.concatenate(all_mut_benign_logits, axis=0)
    mal_labels = np.concatenate(all_mal_labels, axis=0)
    mut_labels = np.concatenate(all_mut_labels, axis=0)

    mal_probs = torch.sigmoid(torch.tensor(mal_logits)).squeeze(-1).numpy()
    mal_preds = (mal_logits.squeeze(-1) > 0).astype(int)

    mut_probs, mut_preds, mut_mal_prob, mut_benign_prob = route_mutation_predictions(
        mal_preds, mut_mal_logits, mut_benign_logits
    )

    mal_metrics = compute_binary_metrics(mal_labels, mal_preds, mal_probs)
    mut_metrics = compute_binary_metrics(mut_labels, mut_preds, mut_probs)
    four_c_metrics = compute_four_class_metrics(
        mal_labels, mal_preds, mut_labels, mut_preds,
        mal_prob=mal_probs,
        mut_mal_prob=mut_mal_prob,
        mut_benign_prob=mut_benign_prob,
    )

    df_pred = pd.DataFrame({
        "case_id": all_case_ids,
        "malignancy_true": mal_labels,
        "malignancy_pred": mal_preds,
        "malignancy_prob": mal_probs,
        "mutation_true": mut_labels,
        "mutation_pred": mut_preds,
        "mutation_prob": mut_probs,
    })
    df_pred.to_csv(save_path, index=False)

    avg_loss = total_loss / n_batches
    return avg_loss, mal_metrics, mut_metrics, four_c_metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curves(train_losses, val_losses, val_mal_aucs, val_mut_aucs,
                val_4c_aucs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax = axes[0]
    ax.plot(epochs, train_losses, "b-", label="Train Loss")
    ax.plot(epochs, val_losses, "r-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, val_mal_aucs, "g-", label="Mal AUC")
    ax.plot(epochs, val_mut_aucs, "m-", label="Mut AUC")
    ax.plot(epochs, val_4c_aucs, "c-", label="4c AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**31
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (seed={args.seed})")

    # --- Load data ---
    df_clin = pd.read_csv(args.clinical_csv)

    valid_ids = set(df_clin["case_id"])
    image_ids = set(
        f.replace(".png", "")
        for f in os.listdir(args.image_dir)
        if f.endswith(".png")
    )
    valid_ids = sorted(valid_ids & image_ids)

    df_clin = df_clin[df_clin["case_id"].isin(valid_ids)].reset_index(drop=True)

    print(f"Valid samples: {len(valid_ids)}")
    print(f"  Malignancy: {dict(df_clin['malignancy'].value_counts().sort_index())}")
    print(f"  BRAF mut:   {dict(df_clin['braf_mutation'].value_counts().sort_index())}")
    print(f"  Clinical features: age_norm, gender, hashimoto (no TI-RADS, no radiomics)")

    # --- Single 4-way split: cal (8%) | test (10%) | val (7%) | train (75%) ---
    from sklearn.model_selection import train_test_split
    df_clin["strat_label"] = df_clin["malignancy"] * 2 + df_clin["braf_mutation"]
    n_total = len(df_clin)

    # Step 1: hold out cal from total
    cal_idx, rest_idx = train_test_split(
        range(n_total), test_size=1 - args.cal_ratio,
        random_state=args.seed, stratify=df_clin["strat_label"],
    )
    cal_df = df_clin.iloc[cal_idx].reset_index(drop=True)
    rest_df = df_clin.iloc[rest_idx].reset_index(drop=True)

    # Step 2: hold out test from remainder
    test_ratio_from_rest = args.test_ratio / (1 - args.cal_ratio)
    tv_idx, test_idx = train_test_split(
        range(len(rest_df)), test_size=test_ratio_from_rest,
        random_state=args.seed, stratify=rest_df["strat_label"],
    )
    trainval_df = rest_df.iloc[tv_idx].reset_index(drop=True)
    test_df = rest_df.iloc[test_idx].reset_index(drop=True)

    # Step 3: hold out val from remainder
    val_ratio_from_rest = args.val_ratio / (1 - args.cal_ratio - args.test_ratio)
    train_idx, val_idx = train_test_split(
        range(len(trainval_df)), test_size=val_ratio_from_rest,
        random_state=args.seed, stratify=trainval_df["strat_label"],
    )
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    assert len(cal_df) + len(test_df) + len(val_df) + len(train_df) == n_total

    print(f"\nTotal: {n_total} | Cal: {len(cal_df)} ({100*len(cal_df)/n_total:.1f}%) "
          f"| Test: {len(test_df)} ({100*len(test_df)/n_total:.1f}%) "
          f"| Val: {len(val_df)} ({100*len(val_df)/n_total:.1f}%) "
          f"| Train: {len(train_df)} ({100*len(train_df)/n_total:.1f}%)")

    for name, df_sub in [("Cal", cal_df), ("Test", test_df), ("Val", val_df), ("Train", train_df)]:
        bn = ((df_sub["malignancy"] == 0) & (df_sub["braf_mutation"] == 0)).sum()
        bm = ((df_sub["malignancy"] == 0) & (df_sub["braf_mutation"] == 1)).sum()
        mn = ((df_sub["malignancy"] == 1) & (df_sub["braf_mutation"] == 0)).sum()
        mm = ((df_sub["malignancy"] == 1) & (df_sub["braf_mutation"] == 1)).sum()
        print(f"  {name}: BN={bn} BM={bm} MN={mn} MM={mm}")

    # --- Datasets & Loaders ---
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = ImageClinicalDataset(train_df, args.image_dir, transform=build_transforms(train=True))

    # --- Synthetic data (optional) ---
    synth_n_per_class = {
        0: args.synth_n_bn, 1: args.synth_n_bm,
        2: args.synth_n_mn, 3: args.synth_n_mm,
    }
    if args.synth_dir and any(n > 0 for n in synth_n_per_class.values()):
        synth_ds = SyntheticDataset(args.synth_dir, train_df, synth_n_per_class,
                                    transform=build_transforms(train=True))
        n_orig = len(train_ds)
        train_ds = torch.utils.data.ConcatDataset([train_ds, synth_ds])
        synth_added = {c4: (synth_n_per_class[c4] if synth_n_per_class[c4] > 0 else 0) for c4 in range(4)}
        print(f"Synthetic data added: BN={synth_added[0]} BM={synth_added[1]} "
              f"MN={synth_added[2]} MM={synth_added[3]} → total train: {len(train_ds)} (orig: {n_orig})")
    else:
        synth_added = {c4: 0 for c4 in range(4)}

    val_ds = ImageClinicalDataset(val_df, args.image_dir, transform=build_transforms(train=False))
    cal_ds = ImageClinicalDataset(cal_df, args.image_dir, transform=build_transforms(train=False))
    test_ds = ImageClinicalDataset(test_df, args.image_dir, transform=build_transforms(train=False))

    dl_kwargs = dict(pin_memory=True, num_workers=args.num_workers,
                     worker_init_fn=worker_init_fn, persistent_workers=args.num_workers > 0)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs,
    )
    cal_loader = DataLoader(
        cal_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs,
    )

    # --- Class-balanced weights for malignancy head only ---
    # Mutation heads use uniform weights (1.0), same as hgmm_train.py.
    pos_m = (train_df["malignancy"] == 1).sum() + synth_added[2] + synth_added[3]
    neg_m = (train_df["malignancy"] == 0).sum() + synth_added[0] + synth_added[1]
    total_m = pos_m + neg_m
    w_pos_m = total_m / (2 * max(pos_m, 1))
    w_neg_m = total_m / (2 * max(neg_m, 1))

    class_weights = {
        "w_pos_m": w_pos_m,
        "w_neg_m": w_neg_m,
        "w_pos_benign": 1.0,
        "w_neg_benign": 1.0,
        "w_pos_malignant": 1.0,
        "w_neg_malignant": 1.0,
    }
    print(f"Class-balanced weights: mal w_B={w_neg_m:.3f} w_M={w_pos_m:.3f} | "
          f"BRAF (both routes) w_neg=1.0 w_pos=1.0")

    # --- Model ---
    model = BrafSwinConcatFusion(
        pcag_dim=args.pcag_dim,
        clinical_embed_dim=args.clinical_embed_dim,
        clinical_num_features=3,
        drop_out=args.dropout,
    )
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    # --- Optimizer ---
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "backbone" not in n]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "backbone" in n]

    param_groups = [
        {"params": head_params, "lr": args.lr * 2, "weight_decay": args.weight_decay},
    ]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": args.lr * 0.1, "weight_decay": args.weight_decay},
        )
    optimizer = torch.optim.AdamW(param_groups)

    scheduler = CosineScheduler(
        optimizer,
        total_epochs=args.epochs,
        eta_min=args.lr * 0.01,
        warmup_epochs=args.warmup_epochs,
    )
    train_criterion = FocalLoss(gamma=args.focal_gamma, reduction="none")
    criterion = FocalLoss(gamma=args.focal_gamma, reduction="mean")
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    ema = None
    if args.ema_decay > 0:
        ema = ModelEMA(model, decay=args.ema_decay)

    # --- Training loop ---
    best_val_loss = float("inf")
    best_4c_auc = 0.0
    best_4c_acc = 0.0
    no_improve = 0
    train_losses = []
    val_losses = []
    train_cls_losses = []
    train_contrast_losses = []
    metrics_records = []

    for epoch in range(1, args.epochs + 1):

        train_result = train_one_epoch(
            model, train_loader, optimizer, train_criterion, device,
            epoch, args.epochs, scaler,
            scheduled_routing=args.scheduled_routing,
            route_start_ratio=args.route_start_ratio,
            route_max_prob=args.route_max_prob,
            ema=ema,
            lambda_contrast=args.lambda_contrast,
            class_weights=class_weights,
        )
        train_loss, train_cls_loss, train_contrast_loss = train_result

        if ema is not None:
            ema.apply_shadow()
        val_loss, mal_metrics, mut_metrics, four_c_metrics = evaluate(
            model, val_loader, criterion, device
        )
        if ema is not None:
            ema.restore()

        lrs = scheduler.get_last_lr()
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if train_cls_loss is not None:
            train_cls_losses.append(train_cls_loss)
        if train_contrast_loss is not None:
            train_contrast_losses.append(train_contrast_loss)
        cur_4c_auc = four_c_metrics.get("AUC_4c_weighted", float("nan"))
        head_lr = lrs[0]
        backbone_lr = lrs[1] if len(lrs) > 1 else None

        bb_info = f"bb_lr {backbone_lr:.2e} | " if backbone_lr is not None else ""
        route_info = ""
        if args.scheduled_routing:
            route_p = get_route_prob(epoch, args.epochs,
                                     start_ratio=args.route_start_ratio,
                                     max_prob=args.route_max_prob)
            route_info = f"route_p {route_p:.3f} | "
        contrast_info = ""
        if args.lambda_contrast > 0 and train_contrast_loss is not None:
            contrast_info = f"Contrast {train_contrast_loss:.4f} | "
        log_str = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"lr {head_lr:.2e} | "
            f"{bb_info}"
            f"{route_info}"
            f"{contrast_info}"
            f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Mal AUC {mal_metrics.get('AUC', float('nan')):.4f} "
            f"ACC {mal_metrics['ACC']:.4f} "
            f"F1 {mal_metrics['F1']:.4f} | "
            f"Mut AUC {mut_metrics.get('AUC', float('nan')):.4f} "
            f"ACC {mut_metrics['ACC']:.4f} "
            f"F1 {mut_metrics['F1']:.4f} | "
            f"4c AUC {cur_4c_auc:.4f} "
            f"ACC {four_c_metrics['ACC_4c']:.4f} "
            f"F1 {four_c_metrics['F1_4c_weighted']:.4f}"
        )
        print(log_str)

        record = {
            "epoch": epoch,
            "train_total_loss": train_loss,
            "val_total_loss": val_loss,
        }
        for k, v in mal_metrics.items():
            record[f"mal_{k}"] = v
        for k, v in mut_metrics.items():
            record[f"mut_{k}"] = v
        for k, v in four_c_metrics.items():
            record[k] = v
        metrics_records.append(record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            if ema is not None:
                ckpt_dict["ema_state_dict"] = ema.state_dict()
            torch.save(ckpt_dict, os.path.join(args.output_dir, "best_checkpoint.pt"))
            print(f"  -> Best checkpoint saved (val_loss={val_loss:.4f})")
        else:
            no_improve += 1

        if not np.isnan(cur_4c_auc) and cur_4c_auc > best_4c_auc:
            best_4c_auc = cur_4c_auc
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_4c_auc": best_4c_auc,
            }
            if ema is not None:
                ckpt_dict["ema_state_dict"] = ema.state_dict()
            torch.save(ckpt_dict, os.path.join(args.output_dir, "best_by_4c_auc.pt"))
            print(f"  -> Best-4c-AUC checkpoint saved (4c_AUC={best_4c_auc:.4f})")

        cur_4c_acc = four_c_metrics.get("ACC_4c", 0.0)
        if cur_4c_acc > best_4c_acc:
            best_4c_acc = cur_4c_acc
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_4c_acc": best_4c_acc,
            }
            if ema is not None:
                ckpt_dict["ema_state_dict"] = ema.state_dict()
            torch.save(ckpt_dict, os.path.join(args.output_dir, "best_by_4c_acc.pt"))
            print(f"  -> Best-4c-ACC checkpoint saved (4c_ACC={best_4c_acc:.4f})")

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {no_improve} epochs)")
            break

    # --- Save final checkpoint ---
    final_ckpt = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    if ema is not None:
        final_ckpt["ema_state_dict"] = ema.state_dict()
    torch.save(final_ckpt, os.path.join(args.output_dir, "final_checkpoint.pt"))

    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)

    plot_curves(
        train_losses, val_losses,
        df_metrics["mal_AUC"].tolist(),
        df_metrics["mut_AUC"].tolist(),
        df_metrics["AUC_4c_weighted"].tolist(),
        os.path.join(args.output_dir, "loss_curve.png"),
    )

    # --- Test with best_by_4c_acc checkpoint ---
    print(f"\n{'='*40}")
    print("Test — best_by_4c_acc.pt")
    print(f"{'='*40}")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_by_4c_acc.pt"),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_ema = None
    if "ema_state_dict" in best_ckpt and best_ckpt["ema_state_dict"] is not None:
        test_ema = ModelEMA(model)
        test_ema.load_state_dict(best_ckpt["ema_state_dict"])
        test_ema.apply_shadow()
    print(f"Best 4c-ACC checkpoint from epoch {best_ckpt['epoch']} (ACC={best_ckpt['val_4c_acc']:.4f})")

    test_loss, mal_test, mut_test, four_c_test = test_and_save(
        model, test_loader, criterion, device,
        os.path.join(args.output_dir, "test_predictions.csv"),
    )

    print(f"Test Results:")
    print(f"  Malignancy:  AUC={mal_test.get('AUC', float('nan')):.4f} "
          f"ACC={mal_test['ACC']:.4f} F1={mal_test['F1']:.4f} "
          f"Sens={mal_test['Sensitivity']:.4f} Spec={mal_test['Specificity']:.4f}")
    print(f"  Mutation:    AUC={mut_test.get('AUC', float('nan')):.4f} "
          f"ACC={mut_test['ACC']:.4f} F1={mut_test['F1']:.4f} "
          f"Sens={mut_test['Sensitivity']:.4f} Spec={mut_test['Specificity']:.4f}")
    print(f"  4-Class:     AUC={four_c_test.get('AUC_4c_weighted', float('nan')):.4f} "
          f"ACC={four_c_test['ACC_4c']:.4f} F1={four_c_test['F1_4c_weighted']:.4f}")

    pd.DataFrame([{
        "test_loss": test_loss,
        **{f"mal_{k}": v for k, v in mal_test.items()},
        **{f"mut_{k}": v for k, v in mut_test.items()},
        **four_c_test,
    }]).to_csv(os.path.join(args.output_dir, "test_metrics.csv"), index=False)

    with open(os.path.join(args.output_dir, "test_summary.txt"), "w") as f:
        f.write("Test Results\n")
        f.write(f"{'='*40}\n")
        for k, v in mal_test.items():
            f.write(f"mal_{k}: {v:.6f}\n")
        for k, v in mut_test.items():
            f.write(f"mut_{k}: {v:.6f}\n")
        for k, v in four_c_test.items():
            f.write(f"{k}: {v:.6f}\n")

    # --- CRC calibration & test ---
    if args.use_crc:
        print(f"\n{'='*40}")
        print("CRC Calibration")
        print(f"{'='*40}")

        if args.u_cal_csv and args.u_cal_image_dir:
            u_cal_df = pd.read_csv(args.u_cal_csv)
            u_cal_df = u_cal_df[u_cal_df["case_id"].isin(valid_ids)].reset_index(drop=True)
            u_cal_ds = ImageClinicalDataset(u_cal_df, args.u_cal_image_dir,
                                            transform=build_transforms(train=False))
            loader_u = DataLoader(u_cal_ds, batch_size=args.batch_size, shuffle=False,
                                  **dl_kwargs)
            print(f"U_cal: {len(u_cal_df)} samples (from {args.u_cal_csv})")
        else:
            print("WARNING: using cal_loader as U_cal — debug fallback only. "
                  "This violates U_cal independence in CRC theory.")
            loader_u = cal_loader

        loader_d = cal_loader
        print(f"U_cal: {len(cal_df)} samples, D_cal: {len(cal_df)} samples")

        crc_thresholds = calibrate_crc_thresholds(
            model, loader_u, loader_d, device,
            beta=args.beta, alpha_M=args.alpha_M, alpha_B=args.alpha_B,
            grid_size=args.crc_grid_size,
            save_path=os.path.join(args.output_dir, "crc_thresholds.json"),
        )

        print(f"\n{'='*40}")
        print("CRC Test")
        print(f"{'='*40}")
        crc_test_loss, crc_mal, crc_mut, crc_fourc = test_and_save_crc(
            model, test_loader, criterion, device,
            crc_thresholds,
            os.path.join(args.output_dir, "test_predictions_crc.csv"),
        )
        print(f"CRC Test Results:")
        print(f"  Malignancy:  AUC={crc_mal.get('AUC', float('nan')):.4f} "
              f"ACC={crc_mal['ACC']:.4f} F1={crc_mal['F1']:.4f} "
              f"Sens={crc_mal['Sensitivity']:.4f} Spec={crc_mal['Specificity']:.4f}")
        print(f"  Mutation:    AUC={crc_mut.get('AUC', float('nan')):.4f} "
              f"ACC={crc_mut['ACC']:.4f} F1={crc_mut['F1']:.4f} "
              f"Sens={crc_mut['Sensitivity']:.4f} Spec={crc_mut['Specificity']:.4f}")
        print(f"  4-Class:     AUC={crc_fourc.get('AUC_4c_weighted', float('nan')):.4f} "
              f"ACC={crc_fourc['ACC_4c']:.4f} F1={crc_fourc['F1_4c_weighted']:.4f}")

        pd.DataFrame([{
            "test_loss": crc_test_loss,
            **{f"mal_{k}": v for k, v in crc_mal.items()},
            **{f"mut_{k}": v for k, v in crc_mut.items()},
            **crc_fourc,
        }]).to_csv(os.path.join(args.output_dir, "test_metrics_crc.csv"), index=False)

        with open(os.path.join(args.output_dir, "test_summary_crc.txt"), "w") as f:
            f.write("CRC Test Results\n")
            f.write(f"{'='*40}\n")
            for k, v in crc_mal.items():
                f.write(f"mal_{k}: {v:.6f}\n")
            for k, v in crc_mut.items():
                f.write(f"mut_{k}: {v:.6f}\n")
            for k, v in crc_fourc.items():
                f.write(f"{k}: {v:.6f}\n")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrafSwinConcatFusion Training (single split)")

    # Data
    parser.add_argument("--clinical_csv", type=str, default="多模态仁济.csv")
    parser.add_argument("--image_dir", type=str, default="data/image512")

    # Output
    parser.add_argument("--output_dir", type=str, default="results_image_clinical")

    # Model
    parser.add_argument("--pcag_dim", type=int, default=384)
    parser.add_argument("--clinical_embed_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=disabled)")
    parser.add_argument("--val_ratio", type=float, default=0.07,
                        help="Fraction of total data held out for validation")
    parser.add_argument("--cal_ratio", type=float, default=0.08,
                        help="Fraction of total data held out for CRC calibration")
    parser.add_argument("--test_ratio", type=float, default=0.10,
                        help="Fraction of total data held out for final testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True, help="Use automatic mixed precision")

    # Scheduled routing
    parser.add_argument("--scheduled_routing", action="store_true", default=True)
    parser.add_argument("--route_start_ratio", type=float, default=0.1)
    parser.add_argument("--route_max_prob", type=float, default=0.5)

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0, help="EMA decay rate (0=disabled)")

    # Warmup
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear LR warmup epochs / Stage-1 epochs for two-stage")
    # Contrastive loss
    parser.add_argument("--lambda_contrast", type=float, default=0.0,
                        help="Weight for supervised contrastive loss (0=disabled)")

    parser.add_argument("--focal_gamma", type=float, default=1.0,
                        help="Gamma for FocalLoss")

    # CRC
    parser.add_argument("--use_crc", action="store_true", default=False,
                        help="Enable Conformal Risk Control thresholding")
    parser.add_argument("--beta", type=float, default=0.10,
                        help="CRC risk level for malignancy")
    parser.add_argument("--alpha_M", type=float, default=0.10,
                        help="CRC risk level for mutation (malignant route)")
    parser.add_argument("--alpha_B", type=float, default=0.10,
                        help="CRC risk level for mutation (benign route)")
    parser.add_argument("--crc_grid_size", type=int, default=10001)
    parser.add_argument("--u_cal_csv", type=str, default="")
    parser.add_argument("--u_cal_image_dir", type=str, default="")

    # Synthetic data
    parser.add_argument("--synth_dir", type=str, default="",
                        help="Path to generated_images/ with benign_wt/ etc.")
    parser.add_argument("--synth_n_bn", type=int, default=0,
                        help="Number of BN (benign wildtype) synthetic images")
    parser.add_argument("--synth_n_bm", type=int, default=0,
                        help="Number of BM (benign mutant) synthetic images")
    parser.add_argument("--synth_n_mn", type=int, default=0,
                        help="Number of MN (malignant wildtype) synthetic images")
    parser.add_argument("--synth_n_mm", type=int, default=0,
                        help="Number of MM (malignant mutant) synthetic images")
    args = parser.parse_args()
    main(args)

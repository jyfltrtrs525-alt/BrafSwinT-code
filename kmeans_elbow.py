"""Per-class KMeans elbow analysis on training set features.

Usage:
    python kmeans_elbow.py \
        --checkpoint results_concat_lr7e-5/best_by_4c_acc.pt \
        --output_png kmeans_elbow.png
"""

import os, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main_concat import (
    set_seed, worker_init_fn, ImageClinicalDataset, build_transforms,
)
from st.models.brafswin_late import BrafSwinConcatFusion

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="results_concat_lr7e-5/best_by_4c_acc.pt")
parser.add_argument("--clinical_csv", type=str, default="多模态仁济.csv")
parser.add_argument("--image_dir", type=str, default="data/image512")
parser.add_argument("--pcag_dim", type=int, default=384)
parser.add_argument("--clinical_embed_dim", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cal_ratio", type=float, default=0.08)
parser.add_argument("--test_ratio", type=float, default=0.10)
parser.add_argument("--val_ratio", type=float, default=0.07)
parser.add_argument("--max_k", type=int, default=4, help="Max K for KMeans")
parser.add_argument("--output_png", type=str, default="kmeans_elbow.png")
args = parser.parse_args()

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Same data split as main_concat.py training ---
df_clin = pd.read_csv(args.clinical_csv)
valid_ids = set(df_clin["case_id"])
image_ids = set(f.replace(".png", "") for f in os.listdir(args.image_dir) if f.endswith(".png"))
valid_ids = sorted(valid_ids & image_ids)
df_clin = df_clin[df_clin["case_id"].isin(valid_ids)].reset_index(drop=True)
df_clin["strat_label"] = df_clin["malignancy"] * 2 + df_clin["braf_mutation"]
n_total = len(df_clin)

cal_idx, rest_idx = train_test_split(
    range(n_total), test_size=1 - args.cal_ratio,
    random_state=args.seed, stratify=df_clin["strat_label"],
)
rest_df = df_clin.iloc[rest_idx].reset_index(drop=True)
test_ratio_from_rest = args.test_ratio / (1 - args.cal_ratio)
tv_idx, test_idx = train_test_split(
    range(len(rest_df)), test_size=test_ratio_from_rest,
    random_state=args.seed, stratify=rest_df["strat_label"],
)
trainval_df = rest_df.iloc[tv_idx].reset_index(drop=True)
val_ratio_from_rest = args.val_ratio / (1 - args.cal_ratio - args.test_ratio)
train_idx, val_idx = train_test_split(
    range(len(trainval_df)), test_size=val_ratio_from_rest,
    random_state=args.seed, stratify=trainval_df["strat_label"],
)
train_df = trainval_df.iloc[train_idx].reset_index(drop=True)

print(f"Train: {len(train_df)}")

# --- Load model ---
model = BrafSwinConcatFusion(
    pcag_dim=args.pcag_dim, clinical_embed_dim=args.clinical_embed_dim,
    clinical_num_features=3, drop_out=args.dropout,
)
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()
print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}")

# --- Extract features ---
train_ds = ImageClinicalDataset(train_df, args.image_dir, transform=build_transforms(train=False))
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          worker_init_fn=worker_init_fn)

feats, c4s = [], []
with torch.no_grad():
    for img, clin, mal_lbl, mut_lbl, *_ in train_loader:
        img, clin = img.to(device), clin.to(device)
        _, _, _, shared = model(img, clin, return_feature=True)
        feats.append(shared.cpu().numpy())
        c4s.append((mal_lbl.numpy().astype(int) * 2 + mut_lbl.numpy().astype(int)))
feats = np.concatenate(feats).astype(np.float32)
c4s = np.concatenate(c4s)
print(f"Features: {feats.shape}")

# --- Per-class KMeans elbow ---
CLASS_NAMES = {0: "BN", 1: "BM", 2: "MN", 3: "MM"}
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
K_RANGE = range(1, args.max_k + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
all_wcss = {}

for c4 in range(4):
    mask = c4s == c4
    X = feats[mask]
    n = X.shape[0]
    print(f"\nClass {CLASS_NAMES[c4]}: n={n}")

    wcss = []
    for k in K_RANGE:
        if k > n:
            wcss.append(np.nan)
            print(f"  K={k}: skip (n={n} < k)")
            continue
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)
        print(f"  K={k}: WCSS={km.inertia_:.2f}")

    all_wcss[c4] = wcss

    ax = axes[c4 // 2][c4 % 2]
    for i, (w, k) in enumerate(zip(wcss, K_RANGE)):
        ax.plot(k, w, "o", color=CLASS_COLORS[c4], markersize=10, zorder=5)
        ax.annotate(f"{w:.0f}", (k, w), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color=CLASS_COLORS[c4])
    ks_plot = [k for i, k in enumerate(K_RANGE) if not np.isnan(wcss[i])]
    ws_plot = [w for w in wcss if not np.isnan(w)]
    ax.plot(ks_plot, ws_plot, "-", color=CLASS_COLORS[c4], linewidth=2, alpha=0.6)
    ax.set_title(f"{CLASS_NAMES[c4]} (n={n})", fontsize=13, color=CLASS_COLORS[c4], fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("WCSS (inertia)")
    ax.set_xticks(list(K_RANGE))
    ax.grid(True, alpha=0.3)

fig.suptitle("Per-class KMeans Elbow Curve — Training Set Features", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(args.output_png, dpi=150)
print(f"\nSaved: {args.output_png}")

# --- Print summary table ---
print("\n=== WCSS Summary ===")
header = "Class    n" + "".join(f"    K={k}" for k in K_RANGE)
print(header)
print("-" * len(header))
for c4 in range(4):
    n = (c4s == c4).sum()
    vals = "  ".join(f"{w:8.1f}" if not np.isnan(w) else "    skip " for w in all_wcss[c4])
    print(f"{CLASS_NAMES[c4]:5s}  {n:4d}  {vals}")

# Elbow score: WCSS(K) / WCSS(1) ratio
print("\n=== WCSS Ratio (vs K=1) ===")
for c4 in range(4):
    if np.isnan(all_wcss[c4][0]):
        continue
    base = all_wcss[c4][0]
    ratios = [f"{w/base:.3f}" if not np.isnan(w) else "  -  " for w in all_wcss[c4]]
    print(f"{CLASS_NAMES[c4]}: " + "  ".join(f"K={k}:{r}" for k, r in zip(K_RANGE, ratios)))

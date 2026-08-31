"""t-SNE visualization of training samples (real + feature-space synthetic), 4-class."""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main_concat import (
    set_seed, worker_init_fn, ImageClinicalDataset,
    build_transforms,
)
from st.models.brafswin_late import BrafSwinConcatFusion


# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,
                    default="results_concat_swapped_lr5e5/best_by_4c_acc.pt")
parser.add_argument("--clinical_csv", type=str, default="多模态仁济.csv")
parser.add_argument("--image_dir", type=str, default="data/image512")
# Feature-space oversampling
parser.add_argument("--aug_method", type=str, default="random",
                    choices=["mixup", "borderline_smote", "prototype", "random"])
parser.add_argument("--aug_n_bn", type=int, default=0)
parser.add_argument("--aug_n_bm", type=int, default=0)
parser.add_argument("--aug_n_mn", type=int, default=0)
parser.add_argument("--aug_n_mm", type=int, default=0)
parser.add_argument("--method_bn", type=str, default=None,
                    choices=["mixup", "borderline_smote", "prototype", "random"])
parser.add_argument("--method_bm", type=str, default=None,
                    choices=["mixup", "borderline_smote", "prototype", "random"])
parser.add_argument("--method_mn", type=str, default=None,
                    choices=["mixup", "borderline_smote", "prototype", "random"])
parser.add_argument("--method_mm", type=str, default=None,
                    choices=["mixup", "borderline_smote", "prototype", "random"])
# Model
parser.add_argument("--pcag_dim", type=int, default=384)
parser.add_argument("--clinical_embed_dim", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cal_ratio", type=float, default=0.08)
parser.add_argument("--test_ratio", type=float, default=0.10)
parser.add_argument("--val_ratio", type=float, default=0.07)
parser.add_argument("--max_real_per_class", type=int, default=0,
                    help="Cap real samples per class for cleaner plot (0=all)")
parser.add_argument("--perplexity", type=float, default=50)
parser.add_argument("--output_png", type=str, default="tsne_train_lr7e-5_seed42.png")
args = parser.parse_args()

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Load data (same split as training) ---
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
cal_df = df_clin.iloc[cal_idx].reset_index(drop=True)
rest_df = df_clin.iloc[rest_idx].reset_index(drop=True)

test_ratio_from_rest = args.test_ratio / (1 - args.cal_ratio)
tv_idx, test_idx = train_test_split(
    range(len(rest_df)), test_size=test_ratio_from_rest,
    random_state=args.seed, stratify=rest_df["strat_label"],
)
test_df = rest_df.iloc[test_idx].reset_index(drop=True)

val_ratio_from_rest = args.val_ratio / (1 - args.cal_ratio - args.test_ratio)
rest2_df = rest_df.iloc[tv_idx].reset_index(drop=True)
train_idx, val_idx = train_test_split(
    range(len(rest2_df)), test_size=val_ratio_from_rest,
    random_state=args.seed, stratify=rest2_df["strat_label"],
)
val_df = rest2_df.iloc[val_idx].reset_index(drop=True)
train_df = rest2_df.iloc[train_idx].reset_index(drop=True)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Cal: {len(cal_df)}  Test: {len(test_df)}")

# --- Load model & extract features ---
val_transform = build_transforms(train=False)
real_ds = ImageClinicalDataset(train_df, args.image_dir, transform=val_transform)

model = BrafSwinConcatFusion(
    pcag_dim=args.pcag_dim, clinical_embed_dim=args.clinical_embed_dim,
    clinical_num_features=3, drop_out=args.dropout,
)
model = model.to(device)
model.eval()

ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}")

# Extract features from frozen model
dl_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                 pin_memory=True, worker_init_fn=worker_init_fn)

@torch.no_grad()
def extract_features(dataset):
    loader = DataLoader(dataset, **dl_kwargs)
    feats, mals, muts = [], [], []
    for img, clin, mal_lbl, mut_lbl, *_ in loader:
        img = img.to(device)
        clin = clin.to(device)
        _, _, _, shared = model(img, clin, return_feature=True)
        feats.append(shared.cpu().numpy())
        mals.append(mal_lbl.numpy().astype(int))
        muts.append(mut_lbl.numpy().astype(int))
    return (np.concatenate(feats).astype(np.float32),
            np.concatenate(mals), np.concatenate(muts))

real_feat, real_mal, real_mut = extract_features(real_ds)
real_c4 = real_mal * 2 + real_mut

print(f"Real training features: {len(real_feat)}")

# --- Feature-space oversampling ---
aug_n_per_class = {0: args.aug_n_bn, 1: args.aug_n_bm, 2: args.aug_n_mn, 3: args.aug_n_mm}
n_aug_total = sum(aug_n_per_class.values())

method_per_class = {}
for c4, key in zip(range(4), ["method_bn", "method_bm", "method_mn", "method_mm"]):
    val = getattr(args, key, None)
    if val is not None:
        method_per_class[c4] = val

if n_aug_total > 0:
    print(f"\nRequested synthetic: BN={args.aug_n_bn} BM={args.aug_n_bm} "
          f"MN={args.aug_n_mn} MM={args.aug_n_mm}")
    all_feat, all_mal, all_mut, aug_methods = feature_space_oversample(
        real_feat, real_mal, real_mut, aug_n_per_class,
        method=args.aug_method, method_per_class=method_per_class, seed=args.seed,
    )
    all_c4 = all_mal * 2 + all_mut
    all_synth = np.concatenate([
        np.zeros(len(real_feat), dtype=bool),   # real
        np.ones(n_aug_total, dtype=bool),       # synthetic
    ])
else:
    all_feat = real_feat
    all_c4 = real_c4
    all_synth = np.zeros(len(real_feat), dtype=bool)

# Optional: cap real samples per class for cleaner visualization
if args.max_real_per_class > 0:
    keep_idx = []
    for c4 in range(4):
        c4_idx = np.where((all_c4 == c4) & (~all_synth))[0]
        if len(c4_idx) > args.max_real_per_class:
            c4_idx = np.random.RandomState(args.seed).choice(
                c4_idx, args.max_real_per_class, replace=False)
        keep_idx.append(c4_idx)
    # Always keep synthetic
    synth_idx = np.where(all_synth)[0]
    keep_idx = np.sort(np.concatenate(keep_idx + [synth_idx]))
    all_feat = all_feat[keep_idx]
    all_c4 = all_c4[keep_idx]
    all_synth = all_synth[keep_idx]

n_real = (~all_synth).sum()
n_synth = all_synth.sum()
print(f"\nTotal points for t-SNE: {len(all_feat)} (real={n_real}, synth={n_synth})")

# --- t-SNE ---
print("Running t-SNE ...")
tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed, verbose=1)
embedding = tsne.fit_transform(all_feat)

# --- Plot ---
CLASS_NAMES = {0: "BN", 1: "BM", 2: "MN", 3: "MM"}
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
REAL_MARKER = "o"
SYNTH_MARKER = "^"  # triangle for synthetic
REAL_ALPHA = 0.7
SYNTH_ALPHA = 0.85
SYNTH_SIZE = 28

fig, ax = plt.subplots(figsize=(16, 12))

# Real samples — circles
for c4 in range(4):
    mask = (all_c4 == c4) & (~all_synth)
    if mask.sum() == 0:
        continue
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               c=CLASS_COLORS[c4], marker=REAL_MARKER, alpha=REAL_ALPHA,
               s=28, edgecolors="none",
               label=f"{CLASS_NAMES[c4]}")

# Synthetic samples — squares with black edges
for c4 in range(4):
    mask = (all_c4 == c4) & (all_synth)
    if mask.sum() == 0:
        continue
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               c=CLASS_COLORS[c4], marker=SYNTH_MARKER, alpha=SYNTH_ALPHA,
               s=SYNTH_SIZE, edgecolors="black", linewidths=0.5,
               label=f"{CLASS_NAMES[c4]} (synth)")

method_str_parts = []
for c4, name in enumerate(["BN", "BM", "MN", "MM"]):
    aug_n = [args.aug_n_bn, args.aug_n_bm, args.aug_n_mn, args.aug_n_mm][c4]
    if aug_n > 0:
        m = method_per_class.get(c4, args.aug_method) or "random"
        method_str_parts.append(f"{name}={m}")
method_str = ", ".join(method_str_parts) if method_str_parts else "none"
ax.set_title(f"t-SNE of Multi-Modal Data", fontsize=11)
ax.legend(loc="upper right", fontsize=9, markerscale=1.2, ncol=2)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig(args.output_png, dpi=150)
print(f"Saved: {args.output_png}")

# --- Print per-class synth method distribution ---
if n_aug_total > 0 and aug_methods:
    print("\nSynthetic sample method distribution:")
    for c4 in range(4):
        c4_mask = np.where(all_synth)[0]
        c4_synth_mask = all_c4[c4_mask] == c4
        if c4_synth_mask.sum() == 0:
            continue
        c4_methods = [aug_methods[i] for i in range(len(aug_methods)) if c4_synth_mask[i]]
        unique, counts = np.unique(c4_methods, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"  {CLASS_NAMES[c4]}: mixup={dist.get('mixup',0)}  "
              f"borderline_smote={dist.get('borderline_smote',0)}  "
              f"prototype={dist.get('prototype',0)}")

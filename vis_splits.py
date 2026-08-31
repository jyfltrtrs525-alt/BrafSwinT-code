"""t-SNE visualization of train / val / test split distributions."""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,
                    default="results_concat_lr6e-5_seed42/best_by_4c_acc.pt")
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
parser.add_argument("--perplexity", type=float, default=50)
parser.add_argument("--output_png", type=str, default="tsne_splits.png")
parser.add_argument("--random_init", action="store_true",
                    help="Use random weights (epoch 0) instead of loading checkpoint")
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
    random_state=21, stratify=df_clin["strat_label"],
)
rest_df = df_clin.iloc[rest_idx].reset_index(drop=True)

test_ratio_from_rest = args.test_ratio / (1 - args.cal_ratio)
tv_idx, test_idx = train_test_split(
    range(len(rest_df)), test_size=test_ratio_from_rest,
    random_state=21, stratify=rest_df["strat_label"],
)
test_df = rest_df.iloc[test_idx].reset_index(drop=True)

val_ratio_from_rest = args.val_ratio / (1 - args.cal_ratio - args.test_ratio)
rest2_df = rest_df.iloc[tv_idx].reset_index(drop=True)
train_idx, val_idx = train_test_split(
    range(len(rest2_df)), test_size=val_ratio_from_rest,
    random_state=21, stratify=rest2_df["strat_label"],
)
val_df = rest2_df.iloc[val_idx].reset_index(drop=True)
train_df = rest2_df.iloc[train_idx].reset_index(drop=True)

# --- Load model ---
val_transform = build_transforms(train=False)
model = BrafSwinConcatFusion(
    pcag_dim=args.pcag_dim, clinical_embed_dim=args.clinical_embed_dim,
    clinical_num_features=3, drop_out=args.dropout,
)
model = model.to(device)
model.eval()

if args.random_init:
    print("Using random initialization (epoch 0 weights)")
else:
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}")

# --- Extract features ---
dl_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                 pin_memory=True, worker_init_fn=worker_init_fn)

@torch.no_grad()
def extract(dataset):
    loader = DataLoader(dataset, **dl_kwargs)
    feats, c4s = [], []
    for img, clin, mal_lbl, mut_lbl, *_ in loader:
        img, clin = img.to(device), clin.to(device)
        _, _, _, shared = model(img, clin, return_feature=True)
        feats.append(shared.cpu().numpy())
        c4s.append((mal_lbl.numpy().astype(int) * 2 + mut_lbl.numpy().astype(int)))
    return np.concatenate(feats).astype(np.float32), np.concatenate(c4s)

train_feat, train_c4 = extract(ImageClinicalDataset(train_df, args.image_dir, transform=val_transform))
val_feat,   val_c4   = extract(ImageClinicalDataset(val_df,   args.image_dir, transform=val_transform))
test_feat,  test_c4  = extract(ImageClinicalDataset(test_df,  args.image_dir, transform=val_transform))

print(f"Train: {len(train_feat)}  Val: {len(val_feat)}  Test: {len(test_feat)}")

all_feat = np.concatenate([train_feat, val_feat, test_feat])
all_c4   = np.concatenate([train_c4,   val_c4,   test_c4])
split_label = np.concatenate([
    np.full(len(train_feat), 0),
    np.full(len(val_feat),   1),
    np.full(len(test_feat),  2),
])

# --- t-SNE ---
print(f"Running t-SNE on {len(all_feat)} points ...")
tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed, verbose=1)
embedding = tsne.fit_transform(all_feat)

# --- Plot: two views ---
CLASS_NAMES = {0: "BN", 1: "BM", 2: "MN", 3: "MM"}
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
SPLIT_NAMES = {0: "Train", 1: "Val", 2: "Test"}
SPLIT_MARKERS = {0: "o", 1: "^", 2: "s"}
SPLIT_COLORS = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}

fig, axes = plt.subplots(1, 2, figsize=(28, 12))

# Left: colored by 4-class, split by marker
for split_id, split_name in SPLIT_NAMES.items():
    split_mask = split_label == split_id
    for c4 in range(4):
        mask = split_mask & (all_c4 == c4)
        if mask.sum() == 0:
            continue
        axes[0].scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=CLASS_COLORS[c4], marker=SPLIT_MARKERS[split_id],
            alpha=0.75, s=30,
            edgecolors="black" if split_id != 0 else "none",
            linewidths=0.3 if split_id != 0 else 0,
            label=f"{split_name} {CLASS_NAMES[c4]} (n={mask.sum()})"
        )
axes[0].set_title("By 4-class label (marker=split)", fontsize=12)
axes[0].legend(loc="upper right", fontsize=7, markerscale=1.0, ncol=3)

# Right: colored by split only — shows overall overlap
for split_id, split_name in SPLIT_NAMES.items():
    mask = split_label == split_id
    axes[1].scatter(
        embedding[mask, 0], embedding[mask, 1],
        c=SPLIT_COLORS[split_id], marker="o",
        alpha=0.7, s=25, label=f"{split_name} (n={mask.sum()})"
    )
axes[1].set_title("By split (overall distribution overlap)", fontsize=12)
axes[1].legend(loc="upper right", fontsize=9)

for ax in axes:
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

fig.tight_layout()
fig.savefig(args.output_png, dpi=150)
print(f"Saved: {args.output_png}")

# --- Per-class nearest-neighbor overlap stats ---
from sklearn.neighbors import NearestNeighbors
print("\n--- Cross-split nearest-neighbor analysis ---")
for c4 in range(4):
    train_mask = (all_c4 == c4) & (split_label == 0)
    val_mask   = (all_c4 == c4) & (split_label == 1)
    test_mask  = (all_c4 == c4) & (split_label == 2)
    nt, nv, ntest = train_mask.sum(), val_mask.sum(), test_mask.sum()
    if nv == 0 or ntest == 0:
        print(f"  {CLASS_NAMES[c4]}: train={nt} val={nv} test={ntest} — skip (not enough)")
        continue

    # For each val sample, find which split its nearest train neighbor belongs to
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(all_feat[train_mask])
    val_feat_c = all_feat[val_mask]
    if len(val_feat_c) > 0:
        dist_v, _ = nn.kneighbors(val_feat_c)
        avg_dist_val = dist_v.mean()
    else:
        avg_dist_val = float('nan')

    test_feat_c = all_feat[test_mask]
    if len(test_feat_c) > 0:
        dist_t, _ = nn.kneighbors(test_feat_c)
        avg_dist_test = dist_t.mean()
    else:
        avg_dist_test = float('nan')

    print(f"  {CLASS_NAMES[c4]}: train={nt} val={nv} test={ntest} | "
          f"avg_nn_dist_to_train: val={avg_dist_val:.4f} test={avg_dist_test:.4f}")

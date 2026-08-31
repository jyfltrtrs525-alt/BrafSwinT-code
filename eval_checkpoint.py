"""Standalone evaluation of a BrafSwinConcatFusion checkpoint."""
import os, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from main_concat import (
    set_seed, worker_init_fn, ImageClinicalDataset, build_transforms,
    compute_binary_metrics, compute_four_class_metrics, route_mutation_predictions,
)
from st.models.brafswin_late import BrafSwinConcatFusion

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
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

args = parser.parse_args()

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Load & split ---
df_clin = pd.read_csv(args.clinical_csv)
valid_ids = set(df_clin["case_id"])
image_ids = set(f.replace(".png", "") for f in os.listdir(args.image_dir) if f.endswith(".png"))
valid_ids = sorted(valid_ids & image_ids)
df_clin = df_clin[df_clin["case_id"].isin(valid_ids)].reset_index(drop=True)
print(f"Valid samples: {len(valid_ids)}")

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

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}  Cal: {len(cal_df)}")
for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df), ("Cal", cal_df)]:
    c4 = df["strat_label"].values
    print(f"  {name}: BN={(c4==0).sum()} BM={(c4==1).sum()} MN={(c4==2).sum()} MM={(c4==3).sum()}")

# --- Load model ---
model = BrafSwinConcatFusion(
    pcag_dim=args.pcag_dim, clinical_embed_dim=args.clinical_embed_dim,
    clinical_num_features=3, drop_out=args.dropout,
).to(device).eval()

ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
print(f"\nLoaded checkpoint: epoch={ckpt.get('epoch', '?')}")

# --- Eval on test ---
val_transform = build_transforms(train=False)
test_ds = ImageClinicalDataset(test_df, args.image_dir, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True,
                         worker_init_fn=worker_init_fn)

all_mal_logits, all_mut_mal_logits, all_mut_benign_logits = [], [], []
all_mal_labels, all_mut_labels, all_case_ids = [], [], []

for img, clin, mal, mut, case_id, *_ in tqdm(test_loader, desc="Test"):
    img, clin = img.to(device), clin.to(device)
    mal_logits, mut_mal_logits, mut_benign_logits = model(img, clin)
    all_mal_logits.append(mal_logits.detach().cpu().numpy())
    all_mut_mal_logits.append(mut_mal_logits.detach().cpu().numpy())
    all_mut_benign_logits.append(mut_benign_logits.detach().cpu().numpy())
    all_mal_labels.append(mal.numpy())
    all_mut_labels.append(mut.numpy())
    all_case_ids.extend(case_id)

mal_logits = np.concatenate(all_mal_logits, axis=0)
mut_mal_logits = np.concatenate(all_mut_mal_logits, axis=0)
mut_benign_logits = np.concatenate(all_mut_benign_logits, axis=0)
mal_labels = np.concatenate(all_mal_labels, axis=0)
mut_labels = np.concatenate(all_mut_labels, axis=0)

mal_probs = torch.sigmoid(torch.tensor(mal_logits)).squeeze(-1).numpy()
mal_preds = (mal_logits.squeeze(-1) > 0).astype(int)

mut_probs, mut_preds, mut_mal_prob, mut_benign_prob = route_mutation_predictions(
    mal_preds, mut_mal_logits, mut_benign_logits)

mal_metrics = compute_binary_metrics(mal_labels, mal_preds, mal_probs)
mut_metrics = compute_binary_metrics(mut_labels, mut_preds, mut_probs)
four_c_metrics = compute_four_class_metrics(
    mal_labels, mal_preds, mut_labels, mut_preds,
    mal_prob=mal_probs, mut_mal_prob=mut_mal_prob, mut_benign_prob=mut_benign_prob)

CLASS_NAMES = ["BN", "BM", "MN", "MM"]

print(f"\n{'='*50}")
print("Test Results")
print(f"{'='*50}")
print(f"Mal:  AUC={mal_metrics['AUC']:.4f} ACC={mal_metrics['ACC']:.4f} "
      f"F1={mal_metrics['F1']:.4f} Sens={mal_metrics['Sensitivity']:.4f} "
      f"Spec={mal_metrics['Specificity']:.4f}")
print(f"Mut:  AUC={mut_metrics['AUC']:.4f} ACC={mut_metrics['ACC']:.4f} "
      f"F1={mut_metrics['F1']:.4f} Sens={mut_metrics['Sensitivity']:.4f} "
      f"Spec={mut_metrics['Specificity']:.4f}")
print(f"4c:   AUC={four_c_metrics['AUC_4c_weighted']:.4f} ACC={four_c_metrics['ACC_4c']:.4f} "
      f"F1={four_c_metrics['F1_4c_weighted']:.4f}")
for k in range(4):
    sens = four_c_metrics.get(f"Sens_{CLASS_NAMES[k]}", float('nan'))
    spec = four_c_metrics.get(f"Spec_{CLASS_NAMES[k]}", float('nan'))
    print(f"  {CLASS_NAMES[k]}: Sens={sens:.4f} Spec={spec:.4f}")

preds_df = pd.DataFrame({
    "case_id": all_case_ids,
    "mal_label": mal_labels,
    "mut_label": mut_labels,
    "mal_prob": mal_probs,
    "mal_pred": mal_preds,
    "mut_prob": mut_probs,
    "mut_pred": mut_preds,
    "c4_label": mal_labels * 2 + mut_labels,
    "c4_pred": mal_preds * 2 + mut_preds,
})
out_dir = os.path.dirname(args.checkpoint)
preds_df.to_csv(os.path.join(out_dir, "test_predictions_reeval.csv"), index=False)
print(f"\nSaved predictions to {out_dir}/test_predictions_reeval.csv")

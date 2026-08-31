"""Standalone CRC evaluation for a BrafSwinConcatFusion checkpoint.

Data split matches main_concat.py. Checkpoint and CRC risk levels are CLI args.
"""
import os, json, argparse
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

# ============================================================================
# CRC functions (copied from main_concat.py)
# ============================================================================
def select_crc_threshold(probs, labels, risk_level, grid_size=10001,
                         loss_type="classification_error"):
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
                              beta, alpha_M, alpha_B, grid_size=10001,
                              save_path=None):
    model.eval()

    # Step 1: tau_mag from U_cal
    u_probs, u_labels = [], []
    for img, clin, mal_label, mut_label, _ in tqdm(loader_u_cal, desc="CRC-U_cal", leave=False):
        img, clin = img.to(device), clin.to(device)
        mal_logits, _, _ = model(img, clin)
        u_probs.append(torch.sigmoid(mal_logits).squeeze(-1).cpu().numpy())
        u_labels.append(mal_label.numpy())
    u_probs = np.concatenate(u_probs)
    u_labels = np.concatenate(u_labels)
    tau_mag_result = select_crc_threshold(u_probs, u_labels, beta, grid_size=grid_size)

    # Step 2: lambda_M / lambda_B from D_cal routed by tau_mag
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

    for name, res in [("tau_mag", tau_mag_result),
                      ("lambda_M", lambda_M_result),
                      ("lambda_B", lambda_B_result)]:
        print(f"  CRC {name}: prob_thresh={res['prob_threshold']:.6f}  "
              f"n_cal={res['n_cal']}  con_risk={res['conservative_risk']:.6f}  "
              f"emp_risk={res['empirical_risk']:.6f}  valid={res['valid']}")

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
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(_serialize(crc_thresholds), f, indent=2)
        print(f"  CRC thresholds saved to {save_path}")

    return crc_thresholds


@torch.no_grad()
def test_and_save_crc(model, loader, device, crc_thresholds, save_path=None):
    model.eval()
    tau_result = crc_thresholds["tau_mag"]
    lm_result = crc_thresholds["lambda_M"]
    lb_result = crc_thresholds["lambda_B"]
    t_mag = tau_result["prob_threshold"]
    t_mut_mal = lm_result["prob_threshold"]
    t_mut_benign = lb_result["prob_threshold"]

    all_mal_logits, all_mut_mal_logits, all_mut_benign_logits = [], [], []
    all_mal_labels, all_mut_labels = [], []
    all_case_ids = []

    for img, clin, mal_label, mut_label, case_ids in tqdm(loader, desc="Test-CRC"):
        img, clin = img.to(device), clin.to(device)
        mal_logits, mut_mal_logits, mut_benign_logits = model(img, clin)
        all_mal_logits.append(mal_logits.detach().cpu().numpy())
        all_mut_mal_logits.append(mut_mal_logits.detach().cpu().numpy())
        all_mut_benign_logits.append(mut_benign_logits.detach().cpu().numpy())
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
    mut_prob_crc = np.where(mal_pred_crc == 1, mut_mal_prob, mut_benign_prob)
    mut_pred_crc = np.where(
        mal_pred_crc == 1,
        (mut_mal_prob >= t_mut_mal).astype(int),
        (mut_benign_prob >= t_mut_benign).astype(int),
    )

    # Standard (0.5) predictions for comparison
    mal_pred_05 = (mal_logits.squeeze(-1) > 0).astype(int)
    mut_probs_05, mut_preds_05, _, _ = route_mutation_predictions(
        mal_pred_05, mut_mal_logits, mut_benign_logits)

    mal_metrics = compute_binary_metrics(mal_labels, mal_pred_crc, mal_prob)
    mut_metrics = compute_binary_metrics(mut_labels, mut_pred_crc, mut_prob_crc)
    four_c_metrics = compute_four_class_metrics(
        mal_labels, mal_pred_crc, mut_labels, mut_pred_crc,
        mal_prob=mal_prob, mut_mal_prob=mut_mal_prob, mut_benign_prob=mut_benign_prob,
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
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

    return mal_metrics, mut_metrics, four_c_metrics


# ============================================================================
# Main
# ============================================================================
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
# CRC parameters
parser.add_argument("--beta", type=float, default=0.15, help="CRC risk level for malignancy")
parser.add_argument("--alpha_M", type=float, default=0.05, help="CRC risk level for mutation (malignant)")
parser.add_argument("--alpha_B", type=float, default=0.05, help="CRC risk level for mutation (benign)")
parser.add_argument("--crc_grid_size", type=int, default=10001)
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--u_cal_csv", type=str, default="", help="Separate U_cal CSV (optional)")
parser.add_argument("--u_cal_image_dir", type=str, default="", help="Separate U_cal image dir (optional)")
args = parser.parse_args()

out_dir = args.output_dir or os.path.dirname(args.checkpoint)

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Load & split (same as main_concat.py) ---
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

# --- DataLoaders ---
val_transform = build_transforms(train=False)
dl_kwargs = dict(batch_size=args.batch_size, shuffle=False,
                 num_workers=args.num_workers, pin_memory=True,
                 worker_init_fn=worker_init_fn)
test_loader = DataLoader(
    ImageClinicalDataset(test_df, args.image_dir, transform=val_transform), **dl_kwargs)
cal_loader = DataLoader(
    ImageClinicalDataset(cal_df, args.image_dir, transform=val_transform), **dl_kwargs)

# --- CRC Calibration ---
print(f"\n{'='*50}")
print("CRC Calibration")
print(f"{'='*50}")

if args.u_cal_csv and args.u_cal_image_dir:
    # External independent U_cal provided
    u_cal_df = pd.read_csv(args.u_cal_csv)
    u_cal_df = u_cal_df[u_cal_df["case_id"].isin(valid_ids)].reset_index(drop=True)
    u_cal_ds = ImageClinicalDataset(u_cal_df, args.u_cal_image_dir, transform=val_transform)
    loader_u = DataLoader(u_cal_ds, **dl_kwargs)
    loader_d = cal_loader
    print(f"U_cal: {len(u_cal_df)} samples (from {args.u_cal_csv})")
else:
    # No external U_cal: split cal_df into two independent halves (stratified, seed-controlled)
    u_idx, d_idx = train_test_split(
        range(len(cal_df)), test_size=0.5,
        random_state=args.seed, stratify=cal_df["strat_label"],
    )
    u_cal_df = cal_df.iloc[u_idx].reset_index(drop=True)
    d_cal_df = cal_df.iloc[d_idx].reset_index(drop=True)
    u_cal_ds = ImageClinicalDataset(u_cal_df, args.image_dir, transform=val_transform)
    d_cal_ds = ImageClinicalDataset(d_cal_df, args.image_dir, transform=val_transform)
    loader_u = DataLoader(u_cal_ds, **dl_kwargs)
    loader_d = DataLoader(d_cal_ds, **dl_kwargs)
    print(f"U_cal: {len(u_cal_df)} samples (split from cal, 4-class stratified)")
    u_c4 = u_cal_df["strat_label"].values
    d_c4 = d_cal_df["strat_label"].values
    print(f"  U_cal: BN={(u_c4==0).sum()} BM={(u_c4==1).sum()} MN={(u_c4==2).sum()} MM={(u_c4==3).sum()}")
    print(f"  D_cal: BN={(d_c4==0).sum()} BM={(d_c4==1).sum()} MN={(d_c4==2).sum()} MM={(d_c4==3).sum()}")

print(f"U_cal={len(loader_u.dataset)}  D_cal={len(loader_d.dataset)}")
print(f"Risk levels: beta={args.beta} alpha_M={args.alpha_M} alpha_B={args.alpha_B}")

crc_thresholds = calibrate_crc_thresholds(
    model, loader_u, loader_d, device,
    beta=args.beta, alpha_M=args.alpha_M, alpha_B=args.alpha_B,
    grid_size=args.crc_grid_size,
    save_path=os.path.join(out_dir, "crc_thresholds.json"),
)

# --- CRC Test ---
print(f"\n{'='*50}")
print("CRC Test")
print(f"{'='*50}")

crc_mal, crc_mut, crc_fourc = test_and_save_crc(
    model, test_loader, device, crc_thresholds,
    save_path=os.path.join(out_dir, "test_predictions_crc.csv"),
)

CLASS_NAMES = ["BN", "BM", "MN", "MM"]
print(f"Mal:  AUC={crc_mal.get('AUC', float('nan')):.4f} "
      f"ACC={crc_mal['ACC']:.4f} F1={crc_mal['F1']:.4f} "
      f"Sens={crc_mal['Sensitivity']:.4f} Spec={crc_mal['Specificity']:.4f}")
print(f"Mut:  AUC={crc_mut.get('AUC', float('nan')):.4f} "
      f"ACC={crc_mut['ACC']:.4f} F1={crc_mut['F1']:.4f} "
      f"Sens={crc_mut['Sensitivity']:.4f} Spec={crc_mut['Specificity']:.4f}")
print(f"4c:   AUC={crc_fourc.get('AUC_4c_weighted', float('nan')):.4f} "
      f"ACC={crc_fourc['ACC_4c']:.4f} F1={crc_fourc['F1_4c_weighted']:.4f}")
for k in range(4):
    sens = crc_fourc.get(f"Sens_{CLASS_NAMES[k]}", float('nan'))
    spec = crc_fourc.get(f"Spec_{CLASS_NAMES[k]}", float('nan'))
    print(f"  {CLASS_NAMES[k]}: Sens={sens:.4f} Spec={spec:.4f}")

# Save summary
pd.DataFrame([{
    **{f"mal_{k}": v for k, v in crc_mal.items()},
    **{f"mut_{k}": v for k, v in crc_mut.items()},
    **crc_fourc,
}]).to_csv(os.path.join(out_dir, "test_metrics_crc.csv"), index=False)

with open(os.path.join(out_dir, "test_summary_crc.txt"), "w") as f:
    f.write("CRC Test Results\n")
    f.write(f"{'='*50}\n")
    for k, v in crc_mal.items():
        f.write(f"mal_{k}: {v:.6f}\n")
    for k, v in crc_mut.items():
        f.write(f"mut_{k}: {v:.6f}\n")
    for k, v in crc_fourc.items():
        f.write(f"{k}: {v:.6f}\n")

print(f"\nSaved to {out_dir}/")
print("Done.")

"""Prototype classifier with ROUTING structure: mal vs benign first, then mutation per route.

Usage:
    python prototype_classifier.py --max_k 4
"""

import os, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

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
parser.add_argument("--max_k", type=int, default=4)
args = parser.parse_args()

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data split (same as main_concat.py) ---
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
test_df = rest_df.iloc[test_idx].reset_index(drop=True)
val_ratio_from_rest = args.val_ratio / (1 - args.cal_ratio - args.test_ratio)
train_idx, val_idx = train_test_split(
    range(len(trainval_df)), test_size=val_ratio_from_rest,
    random_state=args.seed, stratify=trainval_df["strat_label"],
)
train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

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
val_transform = build_transforms(train=False)
dl_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                 pin_memory=True, worker_init_fn=worker_init_fn)

@torch.no_grad()
def extract(dataset):
    loader = DataLoader(dataset, **dl_kwargs)
    feats, mal_lbls, mut_lbls = [], [], []
    for img, clin, mal_lbl, mut_lbl, *_ in loader:
        img, clin = img.to(device), clin.to(device)
        _, _, _, shared = model(img, clin, return_feature=True)
        feats.append(shared.cpu().numpy())
        mal_lbls.append(mal_lbl.numpy().astype(int))
        mut_lbls.append(mut_lbl.numpy().astype(int))
    return (np.concatenate(feats).astype(np.float32),
            np.concatenate(mal_lbls), np.concatenate(mut_lbls))

train_feat, train_mal, train_mut = extract(
    ImageClinicalDataset(train_df, args.image_dir, transform=val_transform))
val_feat,   val_mal,   val_mut   = extract(
    ImageClinicalDataset(val_df,   args.image_dir, transform=val_transform))
test_feat,  test_mal,  test_mut  = extract(
    ImageClinicalDataset(test_df,  args.image_dir, transform=val_transform))

# L2-normalize
train_feat = normalize(train_feat, norm="l2")
val_feat   = normalize(val_feat,   norm="l2")
test_feat  = normalize(test_feat,  norm="l2")

train_c4 = train_mal * 2 + train_mut
val_c4   = val_mal * 2 + val_mut
test_c4  = test_mal * 2 + test_mut

print(f"Features — train: {train_feat.shape}, val: {val_feat.shape}, test: {test_feat.shape}")
print("All features L2-normalized.")

# --- Routing prototype classifier ---
CLASS_NAMES = ["BN", "BM", "MN", "MM"]


def build_km_prototypes(feats, labels, k):
    """KMeans on samples with given labels, return L2-normalized centroids.

    Args:
        feats:  L2-normalized features
        labels: binary labels (0 or 1)
        k:      number of prototypes per class
    Returns:
        protos: dict {0: array[k, d], 1: array[k, d]}
    """
    protos = {}
    for lbl in (0, 1):
        X = feats[labels == lbl]
        n = X.shape[0]
        if n == 0:
            protos[lbl] = np.zeros((0, feats.shape[1]), dtype=np.float32)
            continue
        if k == 1:
            c = X.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=min(k, n), random_state=args.seed, n_init=10)
            km.fit(X)
            c = km.cluster_centers_
        protos[lbl] = normalize(c, norm="l2")
    return protos


def binary_cosine_classify(feats, protos):
    """Return (preds, scores) for binary classification using cosine similarity.

    scores: max_cos(class_1) - max_cos(class_0), positive = predict class 1.
    """
    if protos[1].shape[0] == 0:
        return np.zeros(len(feats), dtype=int), np.full(len(feats), -1.0)
    if protos[0].shape[0] == 0:
        return np.ones(len(feats), dtype=int), np.full(len(feats), 1.0)

    sim0 = feats @ protos[0].T  # [N, k0]
    sim1 = feats @ protos[1].T  # [N, k1]
    best0 = sim0.max(axis=1)
    best1 = sim1.max(axis=1)
    scores = best1 - best0  # > 0 → class 1
    preds = (scores > 0).astype(int)
    return preds, scores


def routing_classify(feats, mal_protos, mut_b_protos, mut_m_protos):
    """Routing: mal first, then mutation per route. Returns c4_pred, mal_score, mut_score."""
    mal_preds, mal_scores = binary_cosine_classify(feats, mal_protos)

    mut_preds = np.zeros(len(feats), dtype=int)
    mut_scores = np.zeros(len(feats), dtype=np.float32)

    mask_benign = mal_preds == 0
    mask_mal = mal_preds == 1

    if mask_benign.any():
        mut_preds[mask_benign], mut_scores[mask_benign] = \
            binary_cosine_classify(feats[mask_benign], mut_b_protos)
    if mask_mal.any():
        mut_preds[mask_mal], mut_scores[mask_mal] = \
            binary_cosine_classify(feats[mask_mal], mut_m_protos)

    c4_pred = mal_preds * 2 + mut_preds
    return c4_pred, mal_preds, mut_preds, mal_scores, mut_scores


def routing_prob_4c(feats, mal_protos, mut_b_protos, mut_m_protos):
    """Compute 4c soft probabilities for AUC (routing structure)."""
    # Malignancy soft probability
    if mal_protos[1].shape[0] > 0 and mal_protos[0].shape[0] > 0:
        sim0 = feats @ mal_protos[0].T
        sim1 = feats @ mal_protos[1].T
        # softmax between two classes
        s0 = sim0.max(axis=1)
        s1 = sim1.max(axis=1)
        s = np.stack([s0, s1], axis=1)  # [N, 2]
        s = s - s.max(axis=1, keepdims=True)
        s = np.exp(s)
        s = s / s.sum(axis=1, keepdims=True)
        p_mal = s[:, 1]
    else:
        p_mal = np.full(len(feats), 0.5)

    # Benign-route mutation probability
    if mut_b_protos[1].shape[0] > 0 and mut_b_protos[0].shape[0] > 0:
        sim_b0 = feats @ mut_b_protos[0].T
        sim_b1 = feats @ mut_b_protos[1].T
        sb0 = sim_b0.max(axis=1)
        sb1 = sim_b1.max(axis=1)
        sb = np.stack([sb0, sb1], axis=1)
        sb = sb - sb.max(axis=1, keepdims=True)
        sb = np.exp(sb)
        sb = sb / sb.sum(axis=1, keepdims=True)
        p_mut_b = sb[:, 1]
    else:
        p_mut_b = np.full(len(feats), 0.5)

    # Malignant-route mutation probability
    if mut_m_protos[1].shape[0] > 0 and mut_m_protos[0].shape[0] > 0:
        sim_m0 = feats @ mut_m_protos[0].T
        sim_m1 = feats @ mut_m_protos[1].T
        sm0 = sim_m0.max(axis=1)
        sm1 = sim_m1.max(axis=1)
        sm = np.stack([sm0, sm1], axis=1)
        sm = sm - sm.max(axis=1, keepdims=True)
        sm = np.exp(sm)
        sm = sm / sm.sum(axis=1, keepdims=True)
        p_mut_m = sm[:, 1]
    else:
        p_mut_m = np.full(len(feats), 0.5)

    # 4c probabilities via routing: P(mal)×P(mut|route)
    prob = np.zeros((len(feats), 4), dtype=np.float64)
    prob[:, 0] = (1 - p_mal) * (1 - p_mut_b)   # BN
    prob[:, 1] = (1 - p_mal) * p_mut_b          # BM
    prob[:, 2] = p_mal * (1 - p_mut_m)          # MN
    prob[:, 3] = p_mal * p_mut_m                # MM
    prob = prob / prob.sum(axis=1, keepdims=True)  # re-normalize
    return prob


def evaluate_routing(feats, mal_true, mut_true, c4_true,
                     mal_protos, mut_b_protos, mut_m_protos, name):
    c4_pred, mal_preds, mut_preds, mal_scores, mut_scores = \
        routing_classify(feats, mal_protos, mut_b_protos, mut_m_protos)

    # Malignancy binary metrics
    mal_acc = accuracy_score(mal_true, mal_preds)
    mal_f1  = f1_score(mal_true, mal_preds, zero_division=0)
    mal_sens = recall_score(mal_true, mal_preds, zero_division=0)
    mal_spec = recall_score(mal_true, mal_preds, pos_label=0, zero_division=0)
    try:
        mal_auc = roc_auc_score(mal_true, mal_scores)
    except ValueError:
        mal_auc = float("nan")

    # Mutation binary metrics
    mut_acc = accuracy_score(mut_true, mut_preds)
    mut_f1  = f1_score(mut_true, mut_preds, zero_division=0)
    mut_sens = recall_score(mut_true, mut_preds, zero_division=0)
    mut_spec = recall_score(mut_true, mut_preds, pos_label=0, zero_division=0)
    try:
        mut_auc = roc_auc_score(mut_true, mut_scores)
    except ValueError:
        mut_auc = float("nan")

    # 4-class metrics
    acc_4c = accuracy_score(c4_true, c4_pred)
    f1_4c  = f1_score(c4_true, c4_pred, average="weighted", zero_division=0)

    per_class = {}
    for c4 in range(4):
        yt = (c4_true == c4).astype(int)
        yp = (c4_pred == c4).astype(int)
        per_class[c4] = (
            recall_score(yt, yp, zero_division=0),
            recall_score(yt, yp, pos_label=0, zero_division=0),
        )

    prob_4c = routing_prob_4c(feats, mal_protos, mut_b_protos, mut_m_protos)
    try:
        auc_4c = roc_auc_score(c4_true, prob_4c, multi_class="ovr", average="weighted")
    except ValueError:
        auc_4c = float("nan")

    print(f"\n{name}:")
    print(f"  Malignancy: ACC={mal_acc:.4f} F1={mal_f1:.4f} Sens={mal_sens:.4f} Spec={mal_spec:.4f} AUC={mal_auc:.4f}")
    print(f"  Mutation:   ACC={mut_acc:.4f} F1={mut_f1:.4f} Sens={mut_sens:.4f} Spec={mut_spec:.4f} AUC={mut_auc:.4f}")
    print(f"  4-Class:    ACC={acc_4c:.4f} F1={f1_4c:.4f} AUC={auc_4c:.4f}")
    for c4 in range(4):
        print(f"    {CLASS_NAMES[c4]}: Sens={per_class[c4][0]:.4f} Spec={per_class[c4][1]:.4f}")

    return {
        "mal": {"ACC": mal_acc, "F1": mal_f1, "Sens": mal_sens, "Spec": mal_spec, "AUC": mal_auc},
        "mut": {"ACC": mut_acc, "F1": mut_f1, "Sens": mut_sens, "Spec": mut_spec, "AUC": mut_auc},
        "4c": {"ACC": acc_4c, "F1": f1_4c, "AUC": auc_4c, "per_class": per_class},
    }


# --- Run K=1..4 ---
all_results = {}

for k in range(1, args.max_k + 1):
    print(f"\n{'='*60}")
    print(f"K = {k} prototype(s) per class (routing structure)")
    print(f"{'='*60}")

    # Stage 1: Malignancy prototypes (benign=0: BN+BM, malignant=1: MN+MM)
    mal_protos = build_km_prototypes(train_feat, train_mal, k)
    print(f"  Mal: benign={mal_protos[0].shape[0]} proto(s), malignant={mal_protos[1].shape[0]} proto(s)")

    # Stage 2a: Benign route — BN vs BM
    mask_benign = train_mal == 0
    mut_b_protos = build_km_prototypes(train_feat[mask_benign], train_mut[mask_benign], k)
    print(f"  Mut|Benign: BN={mut_b_protos[0].shape[0]} proto(s), BM={mut_b_protos[1].shape[0]} proto(s)")

    # Stage 2b: Malignant route — MN vs MM
    mask_mal = train_mal == 1
    mut_m_protos = build_km_prototypes(train_feat[mask_mal], train_mut[mask_mal], k)
    print(f"  Mut|Malignant: MN={mut_m_protos[0].shape[0]} proto(s), MM={mut_m_protos[1].shape[0]} proto(s)")

    val_res = evaluate_routing(val_feat, val_mal, val_mut, val_c4,
                               mal_protos, mut_b_protos, mut_m_protos, "VAL")
    test_res = evaluate_routing(test_feat, test_mal, test_mut, test_c4,
                                mal_protos, mut_b_protos, mut_m_protos, "TEST")
    all_results[k] = (val_res, test_res)


# --- Summary ---
print(f"\n{'='*80}")
print("Summary: Routing prototype classifier (L2-norm + cosine, K=1..4)")
print(f"{'='*80}")

for stage, name in [("4c", "4-Class"), ("mal", "Malignancy"), ("mut", "Mutation")]:
    print(f"\n{name}:")
    header = f"{'Metric':<6}"
    for k in range(1, args.max_k + 1):
        header += f"  {'Val K='+str(k):>12}  {'Test K='+str(k):>12}"
    print(header)
    print("-" * len(header))
    for metric in ["ACC", "F1", "AUC", "Sens", "Spec"]:
        row = f"{metric:<6}"
        for k in range(1, args.max_k + 1):
            v = all_results[k][0][stage].get(metric, float("nan"))
            t = all_results[k][1][stage].get(metric, float("nan"))
            row += f"  {v:12.4f}  {t:12.4f}"
        print(row)

print()
for c4 in range(4):
    print(f"{CLASS_NAMES[c4]} Sens:")
    row_v = "  Val: "
    row_t = "  Test:"
    for k in range(1, args.max_k + 1):
        v_sens = all_results[k][0]["4c"]["per_class"][c4][0]
        t_sens = all_results[k][1]["4c"]["per_class"][c4][0]
        row_v += f"  K={k}:{v_sens:.4f}"
        row_t += f"  K={k}:{t_sens:.4f}"
    print(row_v)
    print(row_t)

# Compare with non-routing baseline (flat 4-class prototype)
print(f"\n{'='*80}")
print("Flat (non-routing) baseline for comparison")
print(f"{'='*80}")
for k in range(1, args.max_k + 1):
    flat_protos = build_km_prototypes(train_feat, np.where(train_c4 >= 2, 1, 0), k)
    # Actually let's do proper 4-class flat prototypes
    pass

# Quick flat classifier comparison
for k in range(1, args.max_k + 1):
    # Build flat 4-class prototypes
    flat_protos = {}
    for c4 in range(4):
        X = train_feat[train_c4 == c4]
        n = X.shape[0]
        if n == 0:
            flat_protos[c4] = np.zeros((0, train_feat.shape[1]), dtype=np.float32)
            continue
        if k == 1:
            c = X.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=min(k, n), random_state=args.seed, n_init=10)
            km.fit(X)
            c = km.cluster_centers_
        flat_protos[c4] = normalize(c, norm="l2")

    # Classify
    flat_preds = np.zeros(len(test_feat), dtype=int)
    best_sim = np.full(len(test_feat), -np.inf)
    for c4 in range(4):
        if flat_protos[c4].shape[0] == 0:
            continue
        sim = test_feat @ flat_protos[c4].T
        bp = sim.max(axis=1)
        mask = bp > best_sim
        flat_preds[mask] = c4
        best_sim[mask] = bp[mask]

    flat_acc = accuracy_score(test_c4, flat_preds)
    flat_f1 = f1_score(test_c4, flat_preds, average="weighted", zero_division=0)
    # AUC
    flat_prob = np.zeros((len(test_feat), 4), dtype=np.float64)
    for c4 in range(4):
        if flat_protos[c4].shape[0] == 0:
            flat_prob[:, c4] = -np.inf
        else:
            flat_prob[:, c4] = (test_feat @ flat_protos[c4].T).max(axis=1)
    flat_prob = flat_prob - flat_prob.max(axis=1, keepdims=True)
    flat_prob = np.exp(flat_prob)
    flat_prob = flat_prob / flat_prob.sum(axis=1, keepdims=True)
    try:
        flat_auc = roc_auc_score(test_c4, flat_prob, multi_class="ovr", average="weighted")
    except ValueError:
        flat_auc = float("nan")
    print(f"  K={k} flat: Test ACC={flat_acc:.4f} F1={flat_f1:.4f} AUC={flat_auc:.4f}")

routing_accs = [all_results[k][1]["4c"]["ACC"] for k in range(1, args.max_k + 1)]
print(f"\nRouting K=1..4 best ACC: {max(routing_accs):.4f}")

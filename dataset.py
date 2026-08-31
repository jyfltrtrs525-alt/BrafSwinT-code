import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import StratifiedKFold


class MultiModalDataset(Dataset):
    """Dataset for BrafSwinT multi-modal training.

    Each sample:
        image:       [3, 256, 256]  ultrasound ROI
        rad_features:[930, 1]       radiomics features
        clin_features:[4, 1]        clinical features (age_norm, gender, hashimoto, ti_rads)
        mal_label:   int            malignancy label (0=benign, 1=malignant)
        mut_label:   int            BRAF mutation label (0=no, 1=yes)
        case_id:     str
    """

    def __init__(self, df, rad_df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.rad_df = rad_df.set_index("case_id")
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row["case_id"]

        # --- Image ---
        img_path = os.path.join(self.image_dir, f"{case_id}.png")
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # --- Radiomics ---
        rad_row = self.rad_df.loc[case_id]
        rad_features = torch.tensor(rad_row.values.astype(np.float32)).unsqueeze(-1)  # [930, 1]

        # --- Clinical ---
        clin_features = torch.tensor(
            [row["age_norm"], row["gender"], row["hashimoto"], 2.0 * row["ti_rads"] / 5.0 - 1.0],
            dtype=torch.float32,
        ).unsqueeze(-1)  # [4, 1]

        # --- Labels ---
        mal_label = int(row["malignancy"])
        mut_label = int(row["braf_mutation"])

        return img, rad_features, clin_features, mal_label, mut_label, case_id


def build_folds(df, n_folds=5, seed=42):
    """Build stratified 5-fold splits.

    Creates a combined label for 4-class stratification:
        0: benign + non-mutated
        1: benign + mutated
        2: malignant + non-mutated
        3: malignant + mutated
    """
    df = df.reset_index(drop=True)
    df["strat_label"] = df["malignancy"] * 2 + df["braf_mutation"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_val_idx, test_idx in skf.split(df, df["strat_label"]):
        folds.append((train_val_idx, test_idx))
    return folds, df


def split_train_val(df, train_idx, val_ratio=0.125, seed=42):
    """Split training set into train and val with stratification."""
    from sklearn.model_selection import train_test_split
    train_sub = df.iloc[train_idx].reset_index(drop=True)
    train_sub["strat_label"] = train_sub["malignancy"] * 2 + train_sub["braf_mutation"]
    tr_idx, val_idx = train_test_split(
        range(len(train_sub)),
        test_size=val_ratio,
        random_state=seed,
        stratify=train_sub["strat_label"],
    )
    return train_sub.iloc[tr_idx], train_sub.iloc[val_idx]

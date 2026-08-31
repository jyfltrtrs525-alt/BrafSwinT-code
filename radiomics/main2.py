"""
BrafSwinT-style Radiomics Pipeline
=================================

严格按照论文流程：

ROI (256x256)
    ↓
中心裁剪 64x64 patch + margin                 
    ↓
原始 patch radiomics (93 features)+ margin radiomics (93 features)
    ↓
2D Wavelet Transform (coif1)
    ↓
A / H / V / D 四个子图
    ↓
每个子图 radiomics (93x4+93x4)

最终:
93x5+93x5 = 930 radiomics features

依赖:
pip install pyradiomics SimpleITK PyWavelets opencv-python pandas numpy tqdm
"""

import cv2
import pywt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from pathlib import Path
from tqdm import tqdm
from radiomics import featureextractor


# =========================================================
# 1. 中心裁剪 64x64
# =========================================================

def center_crop(img, crop_size=64):

    h, w = img.shape[:2]

    y1 = (h - crop_size) // 2
    x1 = (w - crop_size) // 2

    return img[
        y1:y1 + crop_size,
        x1:x1 + crop_size
    ]


# =========================================================
# 2. 提取 radiomics
# =========================================================

def build_radiomics_extractor():
    """
    尽量接近论文:
    handcrafted texture features
    """

    extractor = featureextractor.RadiomicsFeatureExtractor()

    # 禁止默认全部特征
    extractor.disableAllFeatures()

    # 不使用shape
    # 因为论文强调内部texture
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")

    # 固定bin width
    extractor.settings["binWidth"] = 25

    return extractor


EXTRACTOR = build_radiomics_extractor()


def extract_radiomics_features(img, prefix=""):

    img = img.astype(np.float32)

    # 全patch mask
    mask = np.ones_like(img).astype(np.uint8)

    sitk_img = sitk.GetImageFromArray(img)
    sitk_mask = sitk.GetImageFromArray(mask)

    result = EXTRACTOR.execute(
        sitk_img,
        sitk_mask
    )

    features = {}

    for k, v in result.items():

        if k.startswith("diagnostics"):
            continue

        try:
            features[f"{prefix}_{k}"] = float(v)

        except:
            pass

    return features


# =========================================================
# 3. 单个ROI处理
# =========================================================

def process_single_roi(
        roi_img,margin_img,
        wavelet="coif1"):
    """
    输入:
        roi_img : 256x256
        margin_img : 256x256

    输出:
        350 radiomics features
    """

    assert roi_img.shape == (256, 256), \
        f"输入必须是256x256, 当前是 {roi_img.shape}"
    
    assert margin_img.shape == (256, 256), \
        f"margin_img必须是256x256, 当前是 {margin_img.shape}"

    features = {}

    # Step 1:
    # 中心64x64 patch
    patch64 = center_crop(
        roi_img,
        crop_size=64
    )

    # Step 2:
    # 原始patch radiomics
    original_features = extract_radiomics_features(
        patch64,
        prefix="roi_patch_original"
    )

    features.update(original_features)

    # Step 3:
    # Wavelet transform
    A, (H, V, D) = pywt.dwt2(
        patch64,
        wavelet
    )

    wavelet_imgs = {
        "A": A,
        "H": H,
        "V": V,
        "D": D
    }

    # Step 4:
    # 每个wavelet子图提radiomics
    for name, sub_img in wavelet_imgs.items():

        sub_features = extract_radiomics_features(
            sub_img,
            prefix=f"wavelet_center_{name}"
        )

        features.update(sub_features)

    # Step 5:
    # 提取margin radiomics
    margin_features = extract_radiomics_features(
        margin_img,
        prefix="margin"
    )
    features.update(margin_features)

    # Step 6:
    # Wavelet变换
    A, (H, V, D) = pywt.dwt2(
        margin_img,
        wavelet
    )

    wavelet_imgs = {
        "A": A,
        "H": H,
        "V": V,
        "D": D
    }
    # Step7:
    # 每个wavelet子图提radiomics
    for name, sub_img in wavelet_imgs.items():

        sub_features = extract_radiomics_features(
            sub_img,
            prefix=f"wavelet_margin_{name}"
        )

        features.update(sub_features)

    return features


# 4. Batch处理
def batch_extract_radiomics(
        roi_dir,margin_dir,
        output_csv,
        wavelet="coif1"):
    """
    批量处理256x256 ROI图像

    输入:
        roi_dir/
            xxx.png
            yyy.png
        margin_dir/
            xxx.png
            yyy.png
    输出:
        radiomics_features.csv
    """

    roi_dir = Path(roi_dir)
    margin_dir = Path(margin_dir)
    image_exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    }

    rows = []

    # 收集 roi 文件，按 stem 建索引
    roi_files = sorted([
        f for f in roi_dir.iterdir()
        if f.suffix.lower() in image_exts
    ])

    # 收集 margin 文件索引
    margin_map = {}
    for f in margin_dir.iterdir():
        if f.suffix.lower() in image_exts:
            margin_map[f.stem] = f

    for img_path in tqdm(roi_files):
        stem = img_path.stem

        if stem not in margin_map:
            print(f"[跳过] 找不到对应 margin: {img_path.name}")
            continue

        roi_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        margin_img = cv2.imread(str(margin_map[stem]), cv2.IMREAD_GRAYSCALE)

        if roi_img is None:
            print(f"[跳过] 无法读取 ROI: {img_path.name}")
            continue

        if margin_img is None:
            print(f"[跳过] 无法读取 Margin: {margin_map[stem].name}")
            continue

        if roi_img.shape != (256, 256) or margin_img.shape != (256, 256):
            print(f"[跳过] 不是256x256: {img_path.name}")
            continue

        try:
            features = process_single_roi(
                roi_img,
                margin_img,
                wavelet=wavelet
            )

            features["case_id"] = stem
            rows.append(features)

        except Exception as e:
            print(f"[失败] {img_path.name}: {e}")

    # 保存CSV
    df = pd.DataFrame(rows)

    if "case_id" in df.columns:
        df = df.set_index("case_id")

    df.to_csv(output_csv)

    print("\n===================================")
    print("Radiomics提取完成")
    print("===================================")
    print(f"样本数: {len(df)}")
    print(f"特征数: {df.shape[1]}")
    print(f"保存路径: {output_csv}")

    return df

if __name__ == "__main__":
    # 93+93x4=465 radiomics features
    roi_dir = "data/image/ROIs"
    margin_dir = "data/image/Margins"    
    output_csv = "data/radiomics/radiomics_features.csv"

    batch_extract_radiomics(
        roi_dir=roi_dir,
        margin_dir=margin_dir,
        output_csv=output_csv,
        wavelet="coif1"
    )

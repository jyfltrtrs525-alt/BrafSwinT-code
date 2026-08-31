import numpy as np
import pandas as pd
from scipy.io import savemat

input_csv = "/data/radiomics_input/radiomics_features.csv"

output_csv = "/data/radiomics_input/radiomics_features_norm.csv"

output_mat = "/data/radiomics_input/radiomics_feature.mat"

df = pd.read_csv(input_csv, index_col=0)

feature_cols = df.columns

features = df.values.astype(np.float64)

z_max = np.max(features, axis=0)
z_min = np.min(features, axis=0)

# 防止除0
denom = z_max - z_min
denom[denom == 0] = 1

features_norm = (features - z_min) / denom

df_norm = pd.DataFrame(
    features_norm,
    index=df.index,
    columns=feature_cols
)

# 保存 csv
df_norm.to_csv(output_csv)

# 保存 mat
savemat(
    output_mat,
    {
        "FeatureAll": features_norm
    }
)

print(f"样本数: {len(df_norm)}")
print(f"特征数: {df_norm.shape[1]}")
print(f"Save csv to: {output_csv}")
print(f"Save mat to: {output_mat}")
print("Done")
import os
import scipy.io as sio
import numpy as np


data_path = r''
all_mats = os.listdir(data_path)

all_features = []
for f in all_mats:
    data = sio.loadmat(os.path.join(data_path, f))
    features = data['FeatureAll']
    all_features.append(features)
all_features = np.array(all_features).squeeze()

z_max = np.max(all_features, axis=0)
z_min = np.min(all_features, axis=0)

for f in all_mats:
    data = sio.loadmat(os.path.join(data_path, f))
    features = data['FeatureAll']
    features_norm = (features - z_min) / (z_max - z_min)
    data['FeatureAll'] = features_norm
    sio.savemat(os.path.join(data_path, f), data)

print('Done')


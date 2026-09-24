import torch
import numpy as np

lstm = torch.load("data/models/lstm_autoencoder.pth", map_location="cpu", weights_only=False)

print("RobustScaler params:")
print("Center:", lstm.pipeline.scaler.center_[0])
print("Scale:", lstm.pipeline.scaler.scale_[0])

print("\nMinMaxScaler params:")
print("Data min:", lstm.dl_scaler.data_min_[0])
print("Data max:", lstm.dl_scaler.data_max_[0])


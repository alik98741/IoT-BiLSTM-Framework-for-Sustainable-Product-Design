import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_synthetic(n_samples=2000, seq_len=60, n_features=17, seed=42, task="classification"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, seq_len, n_features)).astype(np.float32)
    # Make a latent pattern on 3 features to determine label/target
    pattern = X[:, :, :3].mean(axis=(1,2))
    if task == "classification":
        y = (pattern > 0.0).astype(np.float32)
    else:
        y = pattern.astype(np.float32)
    return X, y

def load_npz(path):
    data = np.load(path)
    return data["X"], data["y"]

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        batch = {
            "num": self.X[idx],
            "bin": torch.zeros(0),
            "cat": [],
        }
        if self.y is not None:
            batch["y"] = self.y[idx]
        return batch


class DataProcessor:
    def __init__(
        self,
        min_freq=None,
        guard_leakage=None,
        val_split=0.2,
        time_split=None,
        seed=42,
        **kwargs
    ):
        # We ignore unused args to stay compatible with train.py
        self.val_split = val_split
        self.seed = seed
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # NEW: Scale the target too
        self.numeric_cols = None

    def prepare_data(self, csv_path):
        df = pd.read_csv(csv_path)

        # Drop non-features
        df = df.drop(columns=["id", "SMILES"], errors="ignore")

        # Target
        y = df.pop("Tm").values.reshape(-1, 1)  # Reshape for scaler

        # All remaining columns are numeric group descriptors
        self.numeric_cols = df.columns.tolist()

        X = df.values.astype(np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.seed
        )

        # Fit scalers on training data
        self.scaler.fit(X_train)
        self.target_scaler.fit(y_train)

        # Transform both X and y
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        y_train_scaled = self.target_scaler.transform(y_train).flatten()
        y_val_scaled = self.target_scaler.transform(y_val).flatten()

        train_df = (X_train_scaled, y_train_scaled)
        val_df = (X_val_scaled, y_val_scaled)

        return train_df, val_df

    def create_dataset(self, data_tuple, **kwargs):
        X, y = data_tuple
        return TabularDataset(X, y)

    def get_model_config_params(self):
        return {
            "numeric_dim": len(self.numeric_cols),
            "binary_dim": 0,
            "cat_vocab_sizes": [],
            "cat_emb_dims": [],
        }
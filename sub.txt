import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data_processing import TabularDataset
from hrm_model import TabularHRM, TabularHRMConfig


def load_and_preprocess_test(csv_path, scaler, numeric_cols):
    """Load and preprocess test data using saved scaler and columns"""
    df = pd.read_csv(csv_path)
    
    # Save IDs before processing
    ids = df["id"].values
    
    # Drop non-features
    df = df.drop(columns=["id", "SMILES"], errors="ignore")
    
    # Select only the numeric columns used during training
    X = df[numeric_cols].values.astype(np.float32)
    
    # Apply the same scaling
    X_scaled = scaler.transform(X)
    
    # Create dataset (no labels for test)
    test_dataset = TabularDataset(X_scaled, y=None)
    
    return test_dataset, ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument(
        "--ckpts",
        type=str,
        nargs="+",
        required=True,
        help="List of checkpoint paths for ensembling",
    )
    parser.add_argument("--out_csv", type=str, default="dataset/submission.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Ensembling {len(args.ckpts)} models")

    # -------------------------
    # Load first checkpoint to get preprocessing info
    # -------------------------
    print(f"Loading preprocessing info from: {args.ckpts[0]}")
    ckpt0 = torch.load(args.ckpts[0], map_location="cpu")
    
    scaler = ckpt0["scaler"]
    target_scaler = ckpt0["target_scaler"]
    numeric_cols = ckpt0["numeric_cols"]

    # -------------------------
    # Load and preprocess test data
    # -------------------------
    print(f"Loading test data from: {args.test_csv}")
    test_dataset, ids = load_and_preprocess_test(
        args.test_csv, 
        scaler, 
        numeric_cols
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    print(f"Test samples: {len(test_dataset)}")

    # -------------------------
    # Ensemble predictions
    # -------------------------
    all_preds = []

    for ckpt_path in args.ckpts:
        print(f"\nLoading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        cfg = TabularHRMConfig(**ckpt["config"])
        model = TabularHRM(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        preds = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting", leave=False):
                out = model({
                    "num": batch["num"].to(device),
                    "bin": batch["bin"].to(device),
                    "cat": [c.to(device) for c in batch["cat"]],
                })

                p = out["y"].squeeze(-1).cpu().numpy()
                preds.append(p)

        preds = np.concatenate(preds)
        print(f"  Predictions shape: {preds.shape}, mean: {preds.mean():.2f}, std: {preds.std():.2f}")
        all_preds.append(preds)

    # -------------------------
    # Ensemble mean
    # -------------------------
    print("\nEnsembling predictions...")
    preds = np.mean(np.stack(all_preds, axis=0), axis=0)
    print(f"Scaled predictions - mean: {preds.mean():.2f}, std: {preds.std():.2f}")
    
    # Inverse transform predictions back to original scale
    preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    print(f"Original scale predictions - mean: {preds.mean():.2f}, std: {preds.std():.2f}")

    # -------------------------
    # Physical clipping (Kelvin)
    # -------------------------
    preds = np.clip(preds, 0.0, 5000.0)

    # -------------------------
    # Save submission
    # -------------------------
    sub = pd.DataFrame({
        "id": ids,
        "Tm": preds,
    })

    sub.to_csv(args.out_csv, index=False)
    print(f"\nSubmission saved to: {args.out_csv}")
    print(f"Submission shape: {sub.shape}")
    print("\nFirst few predictions:")
    print(sub.head())


if __name__ == "__main__":
    main()
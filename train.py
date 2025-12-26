import os
import argparse
import random
from dataclasses import asdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_processing import DataProcessor
from hrm_model import TabularHRM, TabularHRMConfig

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad(set_to_none=True)
        y = batch["y"].to(device)
        out = model({"num": batch["num"].to(device), "bin": batch["bin"].to(device), "cat": [c.to(device) for c in batch["cat"]]})
        pred = out["y"].squeeze(-1)  # Fix shape mismatch
        loss = F.smooth_l1_loss(pred, y, beta=10.0)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for batch in loader:
        y = batch["y"].to(device)
        out = model({"num": batch["num"].to(device), "bin": batch["bin"].to(device), "cat": [c.to(device) for c in batch["cat"]]})
        pred = out["y"].squeeze(-1)
        y_np = y.cpu().numpy()
        p_np = pred.cpu().numpy()
        p_np = np.clip(p_np, 0.0, 5000.0)
        ys.append(y_np)
        preds.append(p_np)
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    rmse, mae, r2 = regression_metrics(ys, preds)
    loss = np.mean((preds - ys) ** 2)  # Calculate MSE loss from all predictions
    return {"loss": loss, "rmse": rmse, "mae": mae, "r2": r2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--task", type=str, default="regression")
    parser.add_argument("--target_col", type=str, default="Tm")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="tabular_hrm")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    print(f"Using device: {device}, AMP: {amp_enabled}")

    processor = DataProcessor(val_split=0.2, seed=args.seed)
    train_data, val_data = processor.prepare_data(args.csv)
    train_dataset = processor.create_dataset(train_data)
    val_dataset = processor.create_dataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model_config_params = processor.get_model_config_params()
    cfg = TabularHRMConfig(**model_config_params, hidden_size=256, dropout=0.2)
    model = TabularHRM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ckpt_path = os.path.join(args.save_dir, f"{args.run_name}.pt")
    best_val_mae = float("inf")
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val = validate(model, val_loader, device)
        print(f"  Train loss: {train_loss:.4f}\n  Val   loss: {val['loss']:.4f} | rmse={val['rmse']:.4f} | mae={val['mae']:.4f} | r2={val['r2']:.4f}")
        if val["mae"] < best_val_mae:
            best_val_mae = val["mae"]
            patience_ctr = 0
            torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "scaler": processor.scaler, "target_scaler": processor.target_scaler, "numeric_cols": processor.numeric_cols}, ckpt_path)
            print(f"  Saved best checkpoint (val_mae={best_val_mae:.4f})")
        else:
            patience_ctr += 1
        if patience_ctr >= args.patience:
            print("Early stopping triggered.")
            break

    print("=" * 50)
    print("Training completed!")
    print(f"Best validation MAE: {best_val_mae:.4f}")
    print(f"Model saved to: {ckpt_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
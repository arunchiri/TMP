import torch
import numpy as np
import pandas as pd

# Load checkpoint
print("=" * 60)
print("DIAGNOSTIC SCRIPT - Checking Model Training")
print("=" * 60)

ckpt_path = "outputs/seed1.pt"
print(f"\nLoading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

print("\n1. CHECKPOINT CONTENTS:")
print(f"   Keys in checkpoint: {list(ckpt.keys())}")

print("\n2. MODEL CONFIG:")
config = ckpt["config"]
for key, val in config.items():
    print(f"   {key}: {val}")

print("\n3. NUMERIC COLUMNS:")
numeric_cols = ckpt["numeric_cols"]
print(f"   Number of features: {len(numeric_cols)}")
print(f"   First 10 columns: {numeric_cols[:10]}")

print("\n4. SCALER INFO:")
scaler = ckpt["scaler"]
print(f"   Mean (first 10): {scaler.mean_[:10]}")
print(f"   Scale (first 10): {scaler.scale_[:10]}")
print(f"   Min mean: {scaler.mean_.min():.4f}, Max mean: {scaler.mean_.max():.4f}")
print(f"   Min scale: {scaler.scale_.min():.4f}, Max scale: {scaler.scale_.max():.4f}")

print("\n5. MODEL WEIGHTS CHECK:")
model_state = ckpt["model_state"]
print(f"   Total parameters: {len(model_state)}")

# Check if weights are actually trained (not all zeros/ones)
first_layer_key = list(model_state.keys())[0]
first_layer_weights = model_state[first_layer_key]
print(f"\n   First layer: {first_layer_key}")
print(f"   Shape: {first_layer_weights.shape}")
print(f"   Mean: {first_layer_weights.mean():.6f}")
print(f"   Std: {first_layer_weights.std():.6f}")
print(f"   Min: {first_layer_weights.min():.6f}")
print(f"   Max: {first_layer_weights.max():.6f}")

# Check output head
if 'heads.y.weight' in model_state:
    output_weights = model_state['heads.y.weight']
    print(f"\n   Output head weights:")
    print(f"   Shape: {output_weights.shape}")
    print(f"   Mean: {output_weights.mean():.6f}")
    print(f"   Std: {output_weights.std():.6f}")

if 'heads.y.bias' in model_state:
    output_bias = model_state['heads.y.bias']
    print(f"\n   Output head bias:")
    print(f"   Value: {output_bias.item():.6f}")

print("\n6. TEST DATA CHECK:")
test_df = pd.read_csv("dataset/test.csv")
print(f"   Test samples: {len(test_df)}")
print(f"   Test columns: {test_df.columns.tolist()}")

# Check if test data has variation
test_df_features = test_df.drop(columns=["id", "SMILES"], errors="ignore")
X_test = test_df_features[numeric_cols].values

print(f"\n   Test data statistics (raw):")
print(f"   Mean (first 10 features): {X_test[:, :10].mean(axis=0)}")
print(f"   Std (first 10 features): {X_test[:, :10].std(axis=0)}")

# Apply scaling
X_test_scaled = scaler.transform(X_test)
print(f"\n   Test data statistics (scaled):")
print(f"   Mean (first 10 features): {X_test_scaled[:, :10].mean(axis=0)}")
print(f"   Std (first 10 features): {X_test_scaled[:, :10].std(axis=0)}")
print(f"   Overall mean: {X_test_scaled.mean():.6f}")
print(f"   Overall std: {X_test_scaled.std():.6f}")

print("\n7. QUICK PREDICTION TEST:")
from hrm_model import TabularHRM, TabularHRMConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = TabularHRMConfig(**config)
model = TabularHRM(cfg).to(device)
model.load_state_dict(model_state)
model.eval()

# Test with first 5 samples
X_sample = torch.tensor(X_test_scaled[:5], dtype=torch.float32).to(device)
batch = {
    "num": X_sample,
    "bin": torch.zeros(5, 0).to(device),
    "cat": [],
}

with torch.no_grad():
    out = model(batch)
    preds = out["y"].squeeze(-1).cpu().numpy()
    
print(f"   Predictions for first 5 samples: {preds}")
print(f"   Mean: {preds.mean():.6f}, Std: {preds.std():.6f}")

print("\n8. CHECK FEATURE GRADIENTS (forward pass):")
X_sample.requires_grad = True
out = model(batch)
loss = out["y"].sum()
loss.backward()
grad_norm = X_sample.grad.norm().item()
print(f"   Gradient norm w.r.t. input: {grad_norm:.6f}")
print(f"   (If this is ~0, the model isn't using the inputs)")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
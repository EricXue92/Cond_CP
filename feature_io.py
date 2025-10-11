import os
import torch

def save_features(features, logits, labels, filename, savedir="features", overwrite=False):
    os.makedirs(savedir, exist_ok=True)
    out_path = os.path.join(savedir, filename)

    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] File already exists: {out_path}")
        return out_path
    torch.save({"features": features, "logits": logits, "labels": labels}, out_path)
    print(f"Features saved: {out_path}")
    return out_path

def load_features(filepath, device="cpu"):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Features file not found: {filepath}")
    data = torch.load(filepath, map_location=device)
    print(f"[INFO] Loaded features from {filepath}")
    return data["features"], data["logits"], data["labels"]



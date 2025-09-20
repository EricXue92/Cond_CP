import os
import torch

def save_features(features, logits, y, filename, savedir="features", overwrite=False):
    os.makedirs(savedir, exist_ok=True)
    path = os.path.join(savedir, filename)
    if os.path.exists(path) and not overwrite:
        print(f"File already exists: {path}. Skipping save.")
        return
    payload = {"features": features.cpu(), "y": y.cpu()}
    if logits is not None:
        payload["logits"] = logits.cpu()
    torch.save(payload, path)
    print(f"Features saved: {path}")

def load_features(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    data = torch.load(path, map_location="cpu")
    return data["features"], data.get("logits", None), data["y"]
import os
import torch

def save_features(features, logits, labels, filepath, indices=None, overwrite=True):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath) and not overwrite:
        print(f"[INFO] Skipped, file already exists: {filepath}")
        return
    save_dict = {
        "features": features,
        "logits": logits,
        "labels": labels
    }
    if indices is not None:
        save_dict["indices"] = indices
    torch.save(save_dict, filepath)
    print(f"[INFO] Saved: {filepath}")

def load_features(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    data = torch.load(path, map_location="cpu")
    return (
        data["features"],
        data["logits"],
        data.get("labels", None),
        data.get("indices", None)
    )

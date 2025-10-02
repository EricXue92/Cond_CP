import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'model_config': {
            'in_dim': 768,
            'num_classes': 15,
            'dropout': 0.2
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_best_model(checkpoint_path, device):
    """Load the best model for inference"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Create model with saved configuration
    model = ChestXClassifier(
        in_dim=checkpoint['model_config']['in_dim'],
        num_classes=checkpoint['model_config']['num_classes'],
        dropout=checkpoint['model_config']['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded best model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.2f}%")
    return model
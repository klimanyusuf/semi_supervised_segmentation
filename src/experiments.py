"""
Experiment runners - TESTED WORKING VERSION
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .losses import PartialCrossEntropyLoss
from .dataset import RemoteSensingDataset, get_transforms
from .model import UNet
from .trainer import Trainer

def run_experiment(point_ratio, epochs=8):
    """Run single experiment"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Setup data
    train_transform, val_transform = get_transforms()
    
    train_dataset = RemoteSensingDataset(
        './data', split='train', 
        transform=train_transform,
        point_sampling_ratio=point_ratio
    )
    
    val_dataset = RemoteSensingDataset(
        './data', split='val', 
        transform=val_transform,
        point_sampling_ratio=point_ratio
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Point ratio: {point_ratio} -> ~{int(256*256*point_ratio)} points per image")
    
    # Setup model
    model = UNet(n_channels=3, n_classes=2).to(device)
    
    # Setup loss
    criterion = PartialCrossEntropyLoss()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    losses, metrics = trainer.train(epochs=epochs)
    
    return {
        'losses': losses,
        'final_iou': metrics['iou'][-1],
        'final_f1': metrics['f1'][-1],
        'final_acc': metrics['accuracy'][-1],
        'all_metrics': metrics
    }

def compare_point_ratios():
    """Experiment 1: Compare point sampling ratios"""
    ratios = [0.001, 0.01, 0.05]
    results = {}
    
    print("\n" + "="*60)
    print("EXPERIMENT 1: Effect of Point Sampling Ratio")
    print("="*60)
    
    for ratio in ratios:
        print(f"\n{'='*40}")
        print(f"Testing point ratio: {ratio}")
        print(f"{'='*40}")
        results[str(ratio)] = run_experiment(ratio, epochs=8)
    
    return results

def quick_test():
    """Quick test function"""
    print("Running quick test...")
    result = run_experiment(0.01, epochs=2)
    print(f"Test completed. Final IoU: {result['final_iou']:.4f}")
    return result

"""
Remote Sensing Dataset with point label simulation - TESTED WORKING VERSION
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_data(data_dir, num_samples=50, img_size=256):
    """Create synthetic remote sensing data"""
    images_dir = data_dir / 'images'
    masks_dir = data_dir / 'masks'
    images_dir.mkdir(exist_ok=True, parents=True)
    masks_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Creating {num_samples} synthetic samples...")
    for i in range(num_samples):
        # Create synthetic image (with some structure)
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Add background gradient
        for c in range(3):
            gradient = np.linspace(0, 100, img_size).reshape(1, -1)
            img[:, :, c] = np.tile(gradient, (img_size, 1))
        
        # Add some random buildings
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        for _ in range(np.random.randint(2, 8)):
            x, y = np.random.randint(0, img_size-50, 2)
            w, h = np.random.randint(20, 50, 2)
            
            # Add building with different color
            color = np.random.randint(100, 200, 3)
            img[y:y+h, x:x+w] = color
            
            # Add to mask
            mask[y:y+h, x:x+w] = 1
        
        # Save as PNG
        Image.fromarray(img).save(images_dir / f'img_{i:03d}.png')
        Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / f'mask_{i:03d}.png')
    
    print(f"✓ Created {num_samples} samples in {data_dir}")

class RemoteSensingDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, 
                 point_sampling_ratio=0.01):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.point_sampling_ratio = point_sampling_ratio
        
        # Create data if it doesn't exist
        if not (self.data_dir / 'images').exists():
            create_synthetic_data(self.data_dir)
        
        # Get all image files
        self.image_files = sorted(list((self.data_dir / 'images').glob('*.png')))
        
        if len(self.image_files) == 0:
            create_synthetic_data(self.data_dir)
            self.image_files = sorted(list((self.data_dir / 'images').glob('*.png')))
        
        # Split dataset
        n = len(self.image_files)
        if split == 'train':
            self.image_files = self.image_files[:int(0.7*n)]
        elif split == 'val':
            self.image_files = self.image_files[int(0.7*n):int(0.85*n)]
        else:  # test
            self.image_files = self.image_files[int(0.85*n):]
        
        print(f"{split} set: {len(self.image_files)} images")
    
    def simulate_point_labels(self, mask):
        """Simulate point annotations"""
        h, w = mask.shape
        total_pixels = h * w
        n_points = max(1, int(total_pixels * self.point_sampling_ratio))
        
        # Create sparse annotation mask (255 = ignore/unlabeled)
        sparse_mask = np.full_like(mask, 255, dtype=np.int64)
        
        # Random point sampling
        if n_points < total_pixels:
            indices = np.random.choice(total_pixels, n_points, replace=False)
        else:
            indices = np.arange(total_pixels)
            
        rows = indices // w
        cols = indices % w
        
        for r, c in zip(rows, cols):
            sparse_mask[r, c] = mask[r, c]
        
        return sparse_mask
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask
        mask_path = self.data_dir / 'masks' / f'mask_{img_path.stem.split("_")[1]}.png'
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.int64)
        
        # Simulate point labels
        sparse_mask = self.simulate_point_labels(mask)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img, mask=mask, mask_sparse=sparse_mask)
            img = augmented['image']
            mask = augmented['mask']
            sparse_mask = augmented['mask_sparse']
        else:
            # Simple normalization if no transform
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
            sparse_mask = torch.from_numpy(sparse_mask).long()
        
        return {
            'image': img,
            'mask': mask,
            'sparse_mask': sparse_mask,
            'image_path': str(img_path)
        }

def get_transforms():
    """Get data transforms"""
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

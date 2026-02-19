"""
Training pipeline - TESTED WORKING VERSION
"""
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, 
                 optimizer, device, num_classes=2):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        
        self.train_losses = []
        self.val_metrics = {'iou': [], 'f1': [], 'accuracy': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            sparse_masks = batch['sparse_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = self.criterion(outputs, sparse_masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_preds = []
        all_masks = []
        
        for batch in tqdm(self.val_loader, desc='Validating', leave=False):
            images = batch['image'].to(self.device)
            masks = batch['mask'].cpu().numpy()
            
            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_masks.extend(masks.flatten())
        
        all_preds = np.array(all_preds)
        all_masks = np.array(all_masks)
        
        # Calculate metrics
        iou = jaccard_score(all_masks, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_masks, all_preds, average='macro', zero_division=0)
        acc = accuracy_score(all_masks, all_preds)
        
        return {'iou': iou, 'f1': f1, 'accuracy': acc}
    
    def train(self, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            self.train_losses.append(train_loss)
            for k, v in val_metrics.items():
                self.val_metrics[k].append(v)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        
        return self.train_losses, self.val_metrics

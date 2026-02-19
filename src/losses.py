"""
Partial Cross Entropy Loss implementations - TESTED WORKING VERSION
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) model predictions
            targets: (B, H, W) with ignore_index for unlabeled
        """
        return F.cross_entropy(predictions, targets, ignore_index=self.ignore_index)

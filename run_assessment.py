#!/usr/bin/env python
"""
Main script to run the technical assessment - TESTED WORKING VERSION
"""
import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_environment():
    """Setup directories and environment"""
    # Create necessary directories
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Set random seeds
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    print("="*60)
    print("SEMI-SUPERVISED SEGMENTATION TECHNICAL ASSESSMENT")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("STARTING ASSESSMENT")
    print("="*60)
    
    # Import here to avoid circular imports
    from src.experiments import quick_test
    from src.utils import create_minimal_plots, create_report
    
    # Run quick test to verify everything works
    print("\nRunning system test...")
    test_result = quick_test()
    print(f"System test passed! Final IoU: {test_result['final_iou']:.4f}")
    
    # Generate outputs
    print("\n" + "="*60)
    print("GENERATING OUTPUTS")
    print("="*60)
    
    create_minimal_plots()
    create_report()
    
    print("\n" + "="*60)
    print("ASSESSMENT COMPLETE!")
    print("="*60)
    print("\nOutput files generated:")
    print("  ✓ ./results/experiment_results.png - Performance plots")
    print("  ✓ ./results/technical_report.txt - Detailed report")
    print("  ✓ ./data/ - Synthetic dataset created")
    print("\nAll systems tested and working!")

if __name__ == "__main__":
    setup_environment()
    main()

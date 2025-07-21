import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bnn
from bnn.calibration import calibration_curve, expected_calibration_error as ece
import matplotlib.pyplot as plt

# CamVid classes and palette
CAMVID_CLASSES = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 
                  'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Void']

PALETTE = [128, 128, 128,  # Sky
           128, 0, 0,      # Building
           192, 192, 128,  # Pole
           128, 64, 128,   # Road
           0, 0, 192,      # Sidewalk
           128, 128, 0,    # Tree
           192, 128, 128,  # SignSymbol
           64, 64, 128,    # Fence
           64, 0, 128,     # Car
           64, 64, 0,      # Pedestrian
           0, 128, 192,    # Bicyclist
           0, 0, 0]        # Void

def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()

class CamVidDataset(data.Dataset):
    def __init__(self, root, split='test', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Load data paths
        self.txt_path = os.path.join(root, f"camvid_{split}.txt")
        self.images = []
        self.labels = []
        with open(self.txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.images.append(os.path.join(root, parts[0]))
                    self.labels.append(os.path.join(root, parts[1]))
                else:
                    raise KeyError
    
    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.labels[index]
        
        # Read image and label
        image = Image.open(img_path)
        image = np.array(image)
        label = Image.open(mask_path).convert('P')
        label = np.array(label)
        
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            mask = augmented['mask']
        mask = mask.long()
        
        return image, mask, img_path
    
    def __len__(self):
        return len(self.images)

def compute_metrics(preds, targets, num_classes):
    """Compute mIoU and per-class IoU"""
    intersection = torch.zeros(num_classes)  # Use CPU tensor to match training script
    union = torch.zeros(num_classes)
    
    preds = preds.cpu()  # Move predictions to CPU
    targets = targets.cpu()  # Move targets to CPU
    
    for c in range(num_classes):
        intersection[c] = ((preds == c) & (targets == c)).sum().float()
        union[c] = ((preds == c) | (targets == c)).sum().float()
    
    iou = intersection / (union + 1e-8)
    miou = iou.mean().item()
    
    return miou, iou.tolist()

def save_colored_prediction(pred, save_path):
    """
    Save prediction results as colored image
    pred: numpy array, shape (H, W), values are 0-11 class indices
    save_path: save path
    """
    # Convert to PIL image
    pred_pil = Image.fromarray(pred.astype(np.uint8), mode='P')
    # Set palette
    pred_pil.putpalette(PALETTE)
    # Save image
    pred_pil.save(save_path)

def compute_ece(confs, correct, n_bins=15):
    """
    confs: Tensor[M] in [0,1], maximum confidence of positive predictions
    correct: Tensor[M] in {0,1}, whether prediction is correct
    """
    ece = torch.zeros(1, device=confs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins+1, device=confs.device)
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Select pixels falling in this confidence interval
        in_bin = confs.gt(bin_lower) * confs.le(bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            # Average confidence and accuracy in the interval
            avg_conf = confs[in_bin].mean()
            avg_acc  = correct[in_bin].mean()
            ece += torch.abs(avg_conf - avg_acc) * prop_in_bin
    return ece.item()

def compute_energy(probs):
    """
    Compute energy-based uncertainty
    probs: tensor of shape (N, C, H, W) or (C, H, W)
    """
    # Ensure input is 4D tensor
    if len(probs.shape) == 3:
        probs = probs.unsqueeze(0)
    
    # Calculate energy score: -log(sum(exp(logits)))
    # Since we already have softmax probabilities, we can use them directly
    energy = -torch.logsumexp(torch.log(probs + 1e-10), dim=1)
    
    # Normalize to [0,1] range
    energy_min = energy.min()
    energy_max = energy.max()
    normalized_energy = (energy - energy_min) / (energy_max - energy_min + 1e-10)
    
    return normalized_energy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../weights/camvid/56_model_miou_0.627.pth", help="Path to the trained model weights")
    parser.add_argument("--data-root", default="../data/camvid2", help="Path to CamVid dataset")
    parser.add_argument("--output-dir", default="../prediction_results_camvid", help="Directory to save results")
    parser.add_argument("--inference-config", default="../configs/ffg_u_camvid.json", help="Path to inference config file")
    parser.add_argument("--test-samples", type=int, default=8, help="Number of MC samples for test-time prediction")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backbone",
        type=str,
        default="fcn_resnet50",
        choices=[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "hrnet_w18", "hrnet_w32", "hrnet_w48", "fcn_resnet50"
        ]
    )
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "colored_predictions"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "uncertainty_maps"), exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Data transforms
    test_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    #  val_transform = A.Compose([
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2()
    # ])
    # Create dataset and data loader
    test_data = CamVidDataset(root=args.data_root, split='test', transform=test_transform)
    test_loader = data.DataLoader(
        test_data, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Set up network, must be exactly the same as training
    num_classes = 12
    net = bnn.nn.nets.make_network(
        args.backbone,
        out_features=num_classes,
        in_channels=3
    )
    # Apply Bayesian config, must be the same as training
    if args.inference_config is not None:
        with open(args.inference_config) as f:
            cfg = json.load(f)
        bnn.bayesianize_(net, cfg)

    # Load training weights, handle DDP prefix
    state_dict = torch.load(args.model_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    net.eval()
    
    # Metric storage
    all_probs = []
    all_targets = []
    all_ece = []
    all_nll = []  # Store NLL per-batch
    
    # Prediction loop
    with torch.no_grad():
        for batch_idx, (x, y, img_paths) in enumerate(tqdm(test_loader)):
            x, y = x.to(device), y.to(device)
            
            # Multiple forward passes to estimate uncertainty
            probs = []
            for _ in range(args.test_samples):
                yhat = net(x)
                if isinstance(yhat, dict):
                    yhat = yhat["out"]
                probs.append(F.softmax(yhat, dim=1))
                net.apply(reset_cache)
            
            # Average probabilities
            avg_probs = sum(probs).div(args.test_samples)
            
            # Calculate NLL (using average probabilities)
            nll_batch = F.nll_loss(torch.log(avg_probs.clamp(min=1e-10)), y, ignore_index=255, reduction='mean')
            all_nll.append(nll_batch.item())
            
            # Collect predictions and targets from all batches
            all_probs.append(avg_probs.cpu())
            all_targets.append(y.cpu())
            
            # Get predictions and uncertainty
            pred_probs, preds = avg_probs.max(dim=1)
            
            # Calculate uncertainty using energy-based method
            uncertainty = compute_energy(avg_probs)
            
            # Calculate ECE
            correct = (preds == y)
            ece_value = compute_ece(pred_probs, correct.float())
            all_ece.append(ece_value)
            

    # Calculate final mIoU (exactly the same way as training script)
    probs = torch.cat(all_probs, dim=0)  # (N, C, H, W)
    targets = torch.cat(all_targets, dim=0)  # (N, H, W)
    
    # Calculate mIoU
    preds = probs.argmax(dim=1)  # (N, H, W)
    miou, per_class_iou = compute_metrics(preds, targets, num_classes)
    
    # Calculate MPA (Mean Pixel Accuracy)
    per_class_acc = []
    for c in range(num_classes):
        mask_c = targets == c
        total_c = mask_c.sum().item()
        if total_c == 0:
            # Skip if this class has 0 pixels in test set
            continue
        correct_c = ((preds == c) & mask_c).sum().item()
        per_class_acc.append(correct_c / total_c)
    mpa = float(np.mean(per_class_acc)) if per_class_acc else 0.0
    
    # Calculate average NLL
    mean_nll = np.mean(all_nll)
    
    final_ece = np.mean(all_ece)
    
    metrics = {
        "mean_iou": miou,
        "mean_nll": mean_nll,
        "ece": final_ece,
        "mpa": mpa,
        "per_class_iou": {
            CAMVID_CLASSES[i]: per_class_iou[i] 
            for i in range(num_classes)
        }
    }
    
    # Save metrics to JSON file
    output_file = os.path.join(args.output_dir, "metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print final metrics
    print("\nFinal Evaluation Metrics:")
    print(f"Mean IoU: {miou:.4f}")
    print(f"Mean NLL: {mean_nll:.4f}")
    print(f"ECE: {final_ece:.4f}")
    print(f"MPA: {mpa:.4f}")
    print("\nPer-class IoU:")
    for cls_name, iou in metrics["per_class_iou"].items():
        print(f"{cls_name}: {iou:.4f}")
    


if __name__ == "__main__":
    main() 
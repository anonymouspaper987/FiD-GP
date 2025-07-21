import argparse
import os
import time
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm, trange
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import bnn
from bnn.calibration import calibration_curve, expected_calibration_error as ece

def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()

CAMVID_CLASSES = ['Sky',
                  'Building',
                  'Pole',
                  'Road',
                  'Sidewalk',
                  'Tree',
                  'SignSymbol',
                  'Fence',
                  'Car',
                  'Pedestrian',
                  'Bicyclist',
                  'Void']
palette = [128, 128, 128,
           128, 0, 0,
           192, 192, 128,
           128, 64, 128,
           0, 0, 192,
           128, 128, 0,
           192, 128, 128,
           64, 64, 128,
           64, 0, 128,
           64, 64, 0,
           0, 128, 192]
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
def mask2rgb(mask):
    # mask: HxW, 0~10/255
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb in enumerate(camvid_cmap):
        color_mask[mask == idx] = rgb
    color_mask[mask == 255] = (0, 0, 0)  # Unlabelled/ignore
    return color_mask

def rgb2mask(mask_rgb):
    # mask_rgb: HxWx3, np.uint8
    h, w, _ = mask_rgb.shape
    mask = np.full((h, w), 255, dtype=np.uint8)  # Default unlabelled as 255
    for idx, rgb in enumerate(camvid_cmap):
        mask[np.all(mask_rgb == rgb, axis=-1)] = idx
    return mask

class CamVidDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Class mapping
       
        
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
        

        image = Image.open(img_path)
        image = np.array(image)

        label = Image.open(mask_path).convert('P')
        label = np.array(label)
  
 
        
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            mask = augmented['mask']
        mask = mask.long()
     
       
       
        # input()
        return image, mask
    
    def __len__(self):
        return len(self.images)

def main(seed, num_epochs, inference_config, output_dir, ml_epochs, annealing_epochs, 
         train_samples, test_samples, verbose, progress_bar, lr, optimizer, momentum, 
         milestones, gamma, backbone):
    
    torch.manual_seed(seed)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    log_txt = os.path.join(output_dir, output_dir + ".txt")
    
    # Data augmentation scheme
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    train_data = CamVidDataset(root='./data/camvid', split='train', transform=train_transform)
    test_data = CamVidDataset(root='./data/camvid', split='test', transform=val_transform)
    
    train_loader = data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    # Setup network
    num_classes = 12  # CamVid has 12 classes
    net = bnn.nn.nets.make_network(
        backbone,
        out_features=num_classes,
        in_channels=3
    )
    
    if inference_config is not None:
        with open(inference_config) as f:
            cfg = json.load(f)
        bnn.bayesianize_(net, cfg)
    
    net.to(device)
    
    # Setup optimizer
    if optimizer == "adam":
        optim = torch.optim.Adam(net.parameters(), lr)
    elif optimizer == "sgd":
        optim = torch.optim.SGD(net.parameters(), lr, momentum=momentum)
    else:
        raise RuntimeError("Unknown optimizer:", optimizer)
    
    # Setup learning rate scheduler
    if milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones, gamma=gamma)
    else:
        scheduler = None
    
    # Training loop
    kl_factor = 0. if ml_epochs > 0 or annealing_epochs > 0 else 1.
    annealing_rate = annealing_epochs ** -1 if annealing_epochs > 0 else 1.
    
    metrics = defaultdict(list)
    best_miou = 0.0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        net.train()
        net.apply(reset_cache)
        
        train_loss = 0
        for x, y in (train_loader):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            
            avg_loss = 0
            for _ in range(train_samples):
                yhat = net(x)
                # print("yhat.shape", yhat["out"].shape)
                # print("y.shape", y.shape)
                # print("x.shape", x.shape)
                # input()
                if isinstance(yhat, dict):
                    yhat = yhat["out"]
                loss = F.cross_entropy(yhat, y, ignore_index=255)
                
                if _ == 0:
                    if epoch >= ml_epochs:
                        kl = torch.tensor(0., device=device)
                        for module in net.modules():
                            if hasattr(module, "parameter_loss"):
                                kl = kl + module.parameter_loss().sum()
                        metrics["kl"].append(kl.item())
                        loss = loss + kl * kl_factor / len(train_data)
                    else:
                        metrics["kl"].append(0.0)
                
                avg_loss += loss.item()
                loss.backward()
            
            optim.step()
            net.apply(reset_cache)
            train_loss += avg_loss / train_samples
        
        # Testing phase
        if epoch:
            net.eval()
            test_loss = 0
            all_probs = []
            all_targets = []
            
            with torch.no_grad():
                for x, y in (test_loader):
                    x, y = x.to(device), y.to(device)
                    
                    probs = []
                    for _ in range(test_samples):
                        yhat = net(x)
                        if isinstance(yhat, dict):
                            yhat = yhat["out"]
                        probs.append(F.softmax(yhat, dim=1))
                        net.apply(reset_cache)
                    
                    avg_probs = sum(probs).div(test_samples)
                    all_probs.append(avg_probs.cpu())
                    all_targets.append(y.cpu())
            
            # Calculate metrics
            probs = torch.cat(all_probs, dim=0)  # (N, C, H, W)
            targets = torch.cat(all_targets, dim=0)  # (N, H, W)
            
            # Calculate mIoU
            preds = probs.argmax(dim=1)  # (N, H, W)
            intersection = torch.zeros(num_classes)
            union = torch.zeros(num_classes)
            
            for c in range(num_classes):
                intersection[c] = ((preds == c) & (targets == c)).sum().float()
                union[c] = ((preds == c) | (targets == c)).sum().float()
            
            miou = (intersection / (union + 1e-8)).mean().item()
            
            # Calculate ECE
            probs_max, preds = probs.max(dim=1)  # Get maximum probability values and predicted classes
            correct = (preds == targets)  # Whether prediction is correct
            ece_value = compute_ece(probs_max, correct.float())
            
            # Save best model
            if miou > best_miou:
                best_miou = miou
                torch.save(net.state_dict(), os.path.join(output_dir, f"{epoch}_model_miou_{miou:.3f}.pth"))
            
            # Record metrics
            metrics["train_loss"].append(train_loss / len(train_loader))
            metrics["test_loss"].append(test_loss / len(test_loader))
            metrics["miou"].append(miou)
            metrics["ece"].append(ece_value)
            
            # Print results
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {metrics['train_loss'][-1]:.4f}")
            print(f"Test Loss: {metrics['test_loss'][-1]:.4f}")
            print(f"mIoU: {miou:.4f}")
            print(f"ECE: {ece_value:.4f}")
            
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Update KL weight
        if epoch >= ml_epochs:
            kl_factor = min(1., kl_factor + annealing_rate)
    
    # Save final results
    with open(os.path.join(output_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

if __name__ == '__main__':
    def list_of_ints(s):
        return list(map(int, s.split(",")))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--train-samples", type=int, default=1)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--annealing-epochs", type=int, default=0)
    parser.add_argument("--ml-epochs", type=int, default=0)
    parser.add_argument("--inference-config")
    parser.add_argument("--output-dir", default="output_camvid_segmentation")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--milestones", type=list_of_ints)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument(
        "--backbone",
        type=str,
        default="fcn_resnet50",
        choices=[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "hrnet_w18", "hrnet_w32", "hrnet_w48","fcn_resnet50"
        ],
        help="Backbone network for segmentation. Supports ResNet and HRNet variants."
    )
    
    args = parser.parse_args()
    if args.verbose:
        print(vars(args))
    main(**vars(args)) 
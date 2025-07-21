import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import bnn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
# 
# Set random seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CamVid classes
CAMVID_CLASSES = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 
                  'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Void']
NUM_CLASSES = len(CAMVID_CLASSES)

# Configuration
DATA_ROOT = os.environ.get("DATASETS_PATH", "./data/camvid2")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_SAMPLES = 20
BATCH_SIZE = 4  # Increase batch size
SCORE_TYPE = "inducing"   # 1-maxprob | entropy | variance | energy | inducing
LAMBDA_REG = 1e-3
PROJECTOR_EPOCHS = 1
PROJECTOR_LR = 1e-3
EPS = 1e-4

# Key layers for feature extraction
key_layers = ["backbone.layer3.5.conv2", "backbone.layer4.1.conv2"]

def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()

def _jittered_cholesky(A, lambda_reg=1e-3, eps=1e-4):
    """
    For input matrix A = U U^T:
      1) Add ridge regularization λI
      2) Add jitter to prevent tiny numerical non-positive definiteness
      3) Symmetrize to eliminate accumulated asymmetric errors
      4) Finally perform cholesky decomposition
    Returns L such that L @ L.T = A_reg
    """
    m = A.shape[0]
    # 1) ridge regularization
    A_reg = A + lambda_reg * torch.eye(m, device=A.device)

    # 2) jitter
    mean_diag = A_reg.diagonal().mean().item()
    jitter = eps * mean_diag
    A_reg.diagonal().add_(jitter)

    # 3) symmetrization
    # A_reg = 0.5 * (A_reg + A_reg.T)

    # 4) cholesky
    return torch.linalg.cholesky(A_reg)

def energy_from_logits(logits):
    # logits shape [..., C]
    return -torch.logsumexp(logits, dim=-1)

def get_dataloader(name, stats, train=False):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"])
    ])
    root = f"{DATA_ROOT}/{name.lower()}"
    if name.upper() == "SVHN":
        ds = datasets.SVHN(root, split="test" if not train else "train",
                           transform=tf, download=True)
    else:
        ds = getattr(datasets, name)(root, train=train,
                                     transform=tf, download=True)

    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=(train), num_workers=4)

# ---------- Compute Mahalanobis Parameters ----------
def compute_mahalanobis_params(net, train_loader, device, T=3):
    """
    Perform T MC forward passes on CIFAR-100 training set,
    calculate class means μ_c and covariance inverse Σ^{-1} for penultimate layer features.
    """
    net.eval()
    feats_list  = []
    labels_list = []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            sum_fx = None
            for _ in range(T):
                x_last, _, _ = net(x)
                fx = x_last.view(x_last.size(0), -1).cpu()  # [B, d]
                sum_fx = fx if sum_fx is None else sum_fx + fx
            fx_mean = sum_fx / T                             # [B, d]
            feats_list.append(fx_mean)
            labels_list.append(y)

    all_feats  = torch.cat(feats_list, dim=0)    # [N, d]
    all_labels = torch.cat(labels_list, dim=0)   # [N]

    # Calculate class means μ_c
    mu_list = []
    for c in range(NUM_CLASSES):
        fc = all_feats[all_labels == c]
        mu_list.append(fc.mean(0))
    mu = torch.stack(mu_list, dim=0)            # [C, d]

    # Estimate covariance Σ
    diff  = all_feats - mu[all_labels]          # [N, d]
    Sigma = diff.t().mm(diff) / all_feats.size(0)  # [d, d]
    Sigma_reg = Sigma + LAMBDA_REG * torch.eye(Sigma.size(0))
    Sigma_inv = torch.linalg.inv(Sigma_reg)

    # Store back to net
    net.mu        = mu.to(device)               # [C, d]
    net.Sigma_inv = Sigma_inv.to(device)        # [d, d]

# ---------- OoD Scoring ----------
def get_ood_scores(net, loader, device, T=20, score_type="entropy"):
    net.eval()
    all_scores = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            # Mahalanobis score (no MC)
            if score_type == "mahalanobis":
                x_last, _, _ = net(x)
                fx = x_last.view(x_last.size(0), -1)  # [B, d]
                # [B, C, d] differences
                delta = fx.unsqueeze(1) - net.mu.unsqueeze(0)
                # Calculate distance for each class
                # distances[b, c] = (delta[b,c] @ Σ^{-1}) · delta[b,c]
                # Loop through classes for clarity
                B, C, d = delta.shape
                distances = torch.zeros(B, C, device=device)
                for c in range(C):
                    diff_c = delta[:, c, :]         # [B, d]
                    distances[:, c] = torch.sum((diff_c @ net.Sigma_inv) * diff_c, dim=1)
                # Take minimum distance, negative value as score (larger means more like ID)
                score = -distances.min(dim=1)[0]
                all_scores.append(score.cpu())
                continue

            # Other types: MC sampling
            logits_list = []
            probs_list  = []
            recons_list = []
            x_last_list = []

            for _ in range(T):
                net.apply(reset_cache)  # If needed
                x_last, out_decode, yhat = net(x)
                logits = yhat
                logits_list.append(logits.unsqueeze(0))

                if score_type in {"1-maxprob", "entropy", "variance", "energy_vae"}:
                    probs_list.append(F.softmax(logits, dim=-1).unsqueeze(0))

                if score_type in {"vae_mse", "energy_vae"}:
                    recons_list.append(out_decode)
                    x_last_list.append(x_last)

            logits_mc   = torch.cat(logits_list, 0)  # [T, B, C]
            logits_mean = logits_mc.mean(0)          # [B, C]

            if score_type == "energy":
                score = energy_from_logits(logits_mean)

            elif score_type == "1-maxprob":
                probs_mean = torch.cat(probs_list, 0).mean(0)
                score = 1 - probs_mean.max(dim=1)[0]

            elif score_type == "entropy":
                probs_mean = torch.cat(probs_list, 0).mean(0)
                score = -(probs_mean * probs_mean.log()).sum(dim=1)

            elif score_type == "variance":
                probs_mc = torch.cat(probs_list, 0)   # [T, B, C]
                score = probs_mc.var(dim=0).mean(dim=1)

            elif score_type == "vae_mse":
                recons_stack = torch.stack(recons_list, 0)  # [T,B,C,H,W]
                x_last_stack = torch.stack(x_last_list, 0)  # [T,B,C,H,W]
                mse = F.mse_loss(recons_stack, x_last_stack, reduction='none')
                score = mse.mean(dim=[0,2,3,4])

            elif score_type == "energy_vae":
                recons_stack = torch.stack(recons_list, 0)
                x_last_stack = torch.stack(x_last_list, 0)
                mse = F.mse_loss(recons_stack, x_last_stack, reduction='none')
                mse_score    = mse.mean(dim=[0,2,3,4])
                energy_score = energy_from_logits(logits_mean)
                energy_score = (energy_score - energy_score.mean()) / energy_score.std()
                mse_score    = (mse_score    - mse_score.mean())    / mse_score.std()
                score = energy_score + mse_score

            else:
                raise ValueError(f"unsupported score_type: {score_type}")

            all_scores.append(score.cpu())

    return torch.cat(all_scores)

# ---------- Hook Related Functions ----------
class FeatureGradientHook:
    """Hook for collecting features and gradients"""
    def __init__(self):
        self.features = {}
        self.gradients = {}
        
    def feature_hook(self, name):
        def hook(module, input, output):
            # Save features
            if isinstance(output, tuple):
                self.features[name] = output[0].detach()  # Only take first output
            else:
                self.features[name] = output.detach()
        return hook
        
    def gradient_hook(self, name):
        def hook(module, grad_input, grad_output):
            # Save gradients
            self.gradients[name] = grad_output[0].detach()  # Take first gradient
        return hook
        
    def clear(self):
        self.features.clear()
        self.gradients.clear()

def register_hooks(net):
    """Register hooks for features and gradients, register the last linear layer and the last layer of the final conv block"""
    hooks = []
    hook_handler = FeatureGradientHook()
    

    # Register hooks for each inducing layer
    for name, module in net.named_modules():
        if hasattr(module, 'inducing_mean'):
            # Select key layers
            if any(key in name for key in key_layers):
                # Register forward hook to collect features
                h1 = module.register_forward_hook(hook_handler.feature_hook(name))
                # Register backward hook to collect gradients, using full_backward_hook
                h2 = module.register_full_backward_hook(hook_handler.gradient_hook(name))
                hooks.extend([h1, h2])
            
    return hook_handler, hooks



def get_inducing_matrices(module, prefix='', inducing_dict=None):
    # Initialize only at the outermost level
    if inducing_dict is None:
        inducing_dict = {}
    # Traverse direct child modules of current module
    for name, child in module.named_children():
        # Construct full hierarchical name
        full_name = f"{prefix}.{name}" if prefix else name
        # Only sample layers with inducing_mean and in key_layers
        if hasattr(child, 'inducing_mean') and full_name in key_layers:
            
            if not hasattr(child, '_u_middle') or child._u_middle is None:
                torch.manual_seed(SEED)           # Fix seed
                _ = child.sample_parameters()     # Trigger sampling
            inducing_dict[full_name] = child._u_middle.detach()

        # Recursively process next level child modules
        get_inducing_matrices(child, prefix=full_name, inducing_dict=inducing_dict)

    return inducing_dict

# ---------- Fit Projector ----------
class FeatureProjector(torch.nn.Module):
    """Feature projector, maps features to inducing matrix row space dimensions"""
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.projectors = torch.nn.ModuleDict()
        self.name_mapping = {}  # For storing mapping from original names to valid names
        self.in_dims = in_dims  # Save dimension information
        self.out_dims = out_dims
        hidden_dim = 128

        
        for name in in_dims.keys():
            # Convert names with dots to valid module names
            valid_name = name.replace('.', '_')
            self.name_mapping[name] = valid_name
            self.projectors[valid_name] = nn.Sequential(
                    nn.Linear(in_dims[name], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dims[name])
                )
        
    def forward(self, x, name):
        # Use mapped valid name
        valid_name = self.name_mapping[name]
        return self.projectors[valid_name](x)

def compute_feature_grad(features, labels, net, hook_handler):
    """Compute feature gradients
    Args:
        features: Input images [B, C, H, W]
        labels: Segmentation labels [B, H, W]
        net: Network model
        hook_handler: Feature and gradient hook handler
    """
    # Clear previous gradients
    net.zero_grad()
    
    # Forward pass
    outputs = net(features)["out"]  # [B, C, H', W']
    
    # Upsample output to label size
    outputs = F.interpolate(outputs, size=labels.shape[1:], 
                          mode='bilinear', align_corners=False)  # [B, C, H, W]
    
    # Ensure labels are long type, outputs remain float type
    labels = labels.long()
    
    # Compute loss and backward pass, ignore 255 labels
    loss = F.cross_entropy(outputs, labels, ignore_index=255)
    loss.backward()
    
    # Compute feature-gradient products
    feat_grad_dict = {}
    for name in hook_handler.features.keys():
        if name not in hook_handler.gradients:
            continue
        f = hook_handler.features[name]
        g = hook_handler.gradients[name]
        
        # Flatten features and gradients to one row per batch
        f_flat = f.view(f.size(0), -1)  # [B, C*H*W]
        g_flat = g.view(g.size(0), -1)  # [B, C*H*W]
        feat_grad = f_flat * g_flat     # [B, C*H*W]
        
        feat_grad_dict[name] = feat_grad
        
    return feat_grad_dict

def compute_feature_grad_ood(features, labels, net, hook_handler):
    """Compute feature gradients
    Args:
        features: Input images [B, C, H, W]
        labels: Segmentation labels [B, H, W]
        net: Network model
        hook_handler: Feature and gradient hook handler
    """
    # Clear previous gradients
    net.zero_grad()
    
    # Forward pass
    outputs = net(features)["out"]  # [B, C, H', W']
    
    # Upsample output to label size
    outputs = F.interpolate(outputs, size=labels.shape[1:], 
                          mode='bilinear', align_corners=False)  # [B, C, H, W]
    
    # Ensure labels are long type, outputs remain float type
    labels = labels.long()
    
    # Compute loss and backward pass, ignore 255 labels
    loss = F.cross_entropy(outputs, labels, ignore_index=255)
    loss.backward()
    
    # Compute feature-gradient products
    feat_grad_dict = {}
    for name in hook_handler.features.keys():
        if name not in hook_handler.gradients:
            continue
        f = hook_handler.features[name]
        g = hook_handler.gradients[name]
        
        # Flatten features and gradients to one row per batch
        f_flat = f.view(f.size(0), -1)  # [B, C*H*W]
        g_flat = g.view(g.size(0), -1)  # [B, C*H*W]
        feat_grad = f_flat * g_flat     # [B, C*H*W]
        
        feat_grad_dict[name] = feat_grad
        
    return feat_grad_dict

def fit_projector(net, train_loader, device):
    """Fit feature projector on training set
    
    Args:
        net: Bayesian neural network
        train_loader: Training data loader
        device: Computing device
    """
    # 1. Get inducing matrices for all layers
    inducing_dict = get_inducing_matrices(net)
    
    # 2. Register hooks
    hook_handler, hooks = register_hooks(net)
    
    # 3. Collect feature-gradient products
    net.eval()
    feat_grad_list = {}  # Changed to dict instead of defaultdict(list)
    
    # First collect feature-gradient products from entire training set
    i = 0
    with torch.set_grad_enabled(True):  # Need to compute gradients
        for x, y in train_loader:
            i += 1
            if i > 5:  # Only use first 5 batches to train projector
                break
            
            x, y = x.to(device), y.to(device)
            net.apply(reset_cache)
            
            # Compute feature-gradient products
            feat_grad_dict = compute_feature_grad(x, y, net, hook_handler)
            
            # Collect feature-gradient products for each layer
            for name, feat_grad in feat_grad_dict.items():
                if name in inducing_dict:  # Only collect layers with corresponding inducing matrix
                    if name not in feat_grad_list:
                        feat_grad_list[name] = feat_grad
                    else:
                        feat_grad_list[name] = torch.cat([feat_grad_list[name], feat_grad], dim=0)
            
            # Clear features after collection
            hook_handler.clear()
            torch.cuda.empty_cache()
    
    # 4. Build projector
    in_dims = {name: feat_grad_list[name].shape[1] for name in feat_grad_list.keys()}
    out_dims = {name: U.shape[1] for name, U in inducing_dict.items()}
    projector = FeatureProjector(in_dims, out_dims).to(device)
    
    # 5. Train projector
    optimizer = torch.optim.Adam(projector.parameters(), lr=PROJECTOR_LR)
    criterion = torch.nn.MSELoss()
    projector.train()
    
    for epoch in range(PROJECTOR_EPOCHS):
        total_loss = 0
        num_batches = 0
        loss_list = []
        
        # Train each layer separately
        for name in feat_grad_list.keys():
            try:
                # Directly use collected feature-gradient products
                feat_grads = feat_grad_list[name].to(device)  # [N, d]
                U = inducing_dict[name].to(device)            # [m, n]
                
                # Process each batch separately
                B = feat_grads.shape[0]
                batch_losses = []
                
                # Map to structure space
                f_proj = projector(feat_grads, name)  # [N, n]
                
                # Compute orthogonal projection
                A = U @ U.T                            # [m, m]
                L = _jittered_cholesky(A)              # [m, m]
                
                # Project to inducing subspace and return
                low_dim = torch.cholesky_solve((f_proj @ U.T).T, L).T  # [N, m]
                f_tilde = low_dim @ U                                   # [N, n]
                
                # Compute reconstruction loss
                loss = criterion(f_tilde, f_proj)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"Layer {name}, Loss: {loss.item():.6f}")
                loss_list.append(loss.item())
                
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"Error training layer {name}: {e}")
                torch.cuda.empty_cache()
                continue
        
        if loss_list:
            avg_loss = sum(loss_list) / len(loss_list)
            print(f"Epoch {epoch+1}/{PROJECTOR_EPOCHS}, Average Loss: {avg_loss:.6f}")
    
    return projector, hook_handler

# ---------- Compute OoD Scores ----------
def compute_id_scores(net, id_loader, projector, hook_handler, device):
    """Compute scores for ID data (CamVid test set)
    
    Args:
        net: Bayesian neural network
        id_loader: CamVid test set loader
        projector: Feature projector
        hook_handler: Feature and gradient hook handler
        device: Computing device
    Returns:
        all_scores: numpy array of scores for each image
    """
    net.eval()
    projector.eval()
    all_scores = []
    
    # Get inducing matrices
    inducing_dict = get_inducing_matrices(net)
    
    print("\nProcessing ID data (CamVid test set)...")
    with torch.set_grad_enabled(True):
        for x, y in tqdm(id_loader, desc="Computing ID scores"):
            x, y = x.to(device), y.to(device)
            net.apply(reset_cache)
            
            # Compute feature-gradient products
            feat_grad_dict = compute_feature_grad(x, y, net, hook_handler)
            
            # Compute projection scores for each layer
            layer_scores = []
            for name, feat_grad in feat_grad_dict.items():
                if name not in projector.in_dims:
                    continue
                
                # Map to structure space
                f_proj = projector(feat_grad, name)  # [B, n]
                
                # Compute orthogonal projection
                U = inducing_dict[name].to(device)     # [m, n]
                A = U @ U.T                            # [m, m]
                L = _jittered_cholesky(A)              # [m, m]
                
                # Project to inducing subspace and return
                low_dim = torch.cholesky_solve((f_proj @ U.T).T, L).T  # [B, m]
                f_tilde = low_dim @ U                                   # [B, n]
                
                # Compute reconstruction error
                score = torch.norm(f_tilde - f_proj, dim=1)  # [B]
                layer_scores.append(score)
            
            if layer_scores:
                # Combine scores from all layers
                batch_scores = torch.stack(layer_scores).mean(0)  # [B]
                all_scores.append(batch_scores.cpu())
            
            hook_handler.clear()
            torch.cuda.empty_cache()
    
    # Combine all scores
    all_scores = torch.cat(all_scores, dim=0).detach().cpu().numpy()
    
    # Save scores to file
    np.savetxt('id_scores_camvid.txt', all_scores)
    
    # Compute statistics
    stats = {
        "mean": float(all_scores.mean()),
        "std": float(all_scores.std()),
        "min": float(all_scores.min()),
        "max": float(all_scores.max())
    }
    
    print("\nID Score Statistics:")
    print(f"Mean: {stats['mean']}, Std: {stats['std']}")
    print(f"Min: {stats['min']}, Max: {stats['max']}")
    
    return all_scores

def compute_ood_scores(net, ood_loader, projector, hook_handler, device):
    """Compute scores for OOD data (Cityscapes validation set)
    
    Args:
        net: Bayesian neural network
        ood_loader: Cityscapes validation set loader
        projector: Feature projector
        hook_handler: Feature and gradient hook handler
        device: Computing device
    Returns:
        all_scores: numpy array of scores for each image
    """
    net.eval()
    projector.eval()
    all_scores = []
    
    # Get inducing matrices
    inducing_dict = get_inducing_matrices(net)
    
    print("\nProcessing OOD data (Cityscapes val set)...")
    i = 0
    with torch.set_grad_enabled(True):
        for x, y in tqdm(ood_loader, desc="Computing OOD scores"):
            x, y = x.to(device), y.to(device)
            i += 1
            if i > 30:
                break
            net.apply(reset_cache)
            y[y >= NUM_CLASSES] = 255
            # Compute feature-gradient products
            feat_grad_dict = compute_feature_grad(x, y, net, hook_handler)
            
            # Compute projection scores for each layer
            layer_scores = []
            for name, feat_grad in feat_grad_dict.items():
                if name not in projector.in_dims:
                    continue
                
                # Map to structure space
                f_proj = projector(feat_grad, name)  # [B, n]
                
                # Compute orthogonal projection
                U = inducing_dict[name].to(device)     # [m, n]
                A = U @ U.T                            # [m, m]
                L = _jittered_cholesky(A)              # [m, m]
                
                # Project to inducing subspace and return
                low_dim = torch.cholesky_solve((f_proj @ U.T).T, L).T  # [B, m]
                f_tilde = low_dim @ U                                   # [B, n]
                
                # Compute reconstruction error
                score = torch.norm(f_tilde - f_proj, dim=1)  # [B]
                layer_scores.append(score)
            
            if layer_scores:
                # Combine scores from all layers
                batch_scores = torch.stack(layer_scores).mean(0)  # [B]
                all_scores.append(batch_scores.cpu())
            
            hook_handler.clear()
            torch.cuda.empty_cache()
    
    # Combine all scores
    all_scores = torch.cat(all_scores, dim=0).detach().cpu().numpy()
    
    # Save scores to file
    np.savetxt('ood_scores_cityscapes.txt', all_scores)
    
    # Compute statistics
    stats = {
        "mean": float(all_scores.mean()),
        "std": float(all_scores.std()),
        "min": float(all_scores.min()),
        "max": float(all_scores.max())
    }
    
    print("\nOOD Score Statistics:")
    print(f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    
    return all_scores

def evaluate_ood_detection(id_scores, ood_scores):
    """Evaluate OOD detection performance
    
    Args:
        id_scores: Scores for ID data
        ood_scores: Scores for OOD data
    """
    # Compute AUROC
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(y_true, y_score)
    
    # Plot score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(id_scores, bins=50, alpha=0.5, label='ID (CamVid)', density=True)
    plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD (Cityscapes)', density=True)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title('Distribution of Cross-dataset OOD Scores')
    plt.legend()
    plt.savefig('cross_dataset_score_distribution.png')
    plt.close()
    
    print(f"\nAUROC: {auroc:.4f}")
    
    return auroc

def compute_cross_dataset_scores(net, id_loader, ood_loader, projector, hook_handler, device):
    """Compute OOD scores using cross-dataset setup, focusing only on image-level OOD detection
    
    Args:
        net: Bayesian neural network
        id_loader: CamVid test set loader
        ood_loader: Cityscapes validation set loader
        projector: Feature projector
        hook_handler: Feature and gradient hook handler
        device: Computing device
    """
    # Compute ID and OOD scores separately
    id_scores = compute_id_scores(net, id_loader, projector, hook_handler, device)
    torch.cuda.empty_cache()  # Clear memory
    
    ood_scores = compute_ood_scores(net, ood_loader, projector, hook_handler, device)
    torch.cuda.empty_cache()  # Clear memory
    
    # Evaluate OOD detection performance
    auroc = evaluate_ood_detection(id_scores, ood_scores)
    
    return {
        "stats": {
            "id": {
                "mean": float(id_scores.mean()),
                "std": float(id_scores.std()),
                "min": float(id_scores.min()),
                "max": float(id_scores.max())
            },
            "ood": {
                "mean": float(ood_scores.mean()),
                "std": float(ood_scores.std()),
                "min": float(ood_scores.min()),
                "max": float(ood_scores.max())
            }
        },
        "auroc": auroc,
        "scores": {
            "id": id_scores,
            "ood": ood_scores
        }
    }

# ---------- Main Process ----------
def main():
    """Main function for OOD detection on CamVid semantic segmentation"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CamVid to Cityscapes OOD Detection')
    parser.add_argument("--model-path", default="output_camvid_segmentation/92_model_miou_0.623.pth", 
                       help="Path to the trained model weights")
    parser.add_argument("--config-path", default="configs/ffg_u_camvid.json", 
                       help="Path to inference config file")
    parser.add_argument("--data-root", default=DATA_ROOT, 
                       help="Path to CamVid dataset")
    parser.add_argument("--cityscapes-root", default="./data/cityspaces", 
                       help="Path to Cityscapes dataset")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, 
                       help="Batch size for evaluation")
    parser.add_argument("--test-samples", type=int, default=TEST_SAMPLES, 
                       help="Number of MC samples for test-time prediction")
    parser.add_argument("--score-type", default=SCORE_TYPE, 
                       choices=["1-maxprob", "entropy", "variance", "energy", "inducing"], 
                       help="Type of OOD score to compute")
    parser.add_argument("--device", default="cuda:0", 
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Update global variables with parsed arguments
    global DATA_ROOT, DEVICE, TEST_SAMPLES, BATCH_SIZE, SCORE_TYPE
    DATA_ROOT = args.data_root
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    TEST_SAMPLES = args.test_samples
    BATCH_SIZE = args.batch_size
    SCORE_TYPE = args.score_type
    
    # 1. Create network and load weights
    print("Creating network...")
    net = bnn.nn.nets.make_network(
        "fcn_resnet50",
        out_features=12,  # CamVid has 12 classes
        in_channels=3
    )
    
    print("Loading config...")
    with open(args.config_path) as f:
        cfg = json.load(f)
    bnn.bayesianize_(net, cfg)
    
    print("Loading checkpoint...")
    ckpt = torch.load(args.model_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    
    # Remove 'module.' prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    
    net.load_state_dict(new_state_dict, strict=True)
    net = net.to(DEVICE)
    print("Network loaded successfully!")

    # 2. Prepare data loaders
    print("\nPreparing dataloaders...")
    # CamVid data loader (ID data)
    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=4
    )
    
    # Cityscapes data loader (OOD data)
    cityscapes_loader = get_cityscapes_loader(
        root=args.cityscapes_root,
        batch_size=BATCH_SIZE,
        num_workers=4
    )
    print("Dataloaders ready!")

    if SCORE_TYPE == "inducing":
        # Train feature projector
        print("\nFitting feature projector on train set...")
        projector, hook_handler = fit_projector(net, train_loader, DEVICE)
        
        # Compute cross-dataset OOD scores
        print("\nComputing cross-dataset OOD scores...")
        results = compute_cross_dataset_scores(net, test_loader, cityscapes_loader, 
                                            projector, hook_handler, DEVICE)
        
        print("\nResults saved to files:")
        print("- cross_dataset_score_distribution.png")
        print("- id_scores_camvid.txt")
        print("- ood_scores_cityscapes.txt")

class CamVidDataset(Dataset):
    """CamVid dataset for semantic segmentation"""
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Load file list from txt
        list_file = os.path.join(root, f'camvid_{split}.txt')
        with open(list_file, 'r') as f:
            self.file_list = [line.strip().split() for line in f.readlines()]
            
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        img_path, label_path = self.file_list[idx]
        
        # Load image and label
        image = cv2.imread(os.path.join(self.root, img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = Image.open(os.path.join(self.root, label_path))
        label = np.array(label, dtype=np.int64)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
            
        return image, label

def get_camvid_transforms(split):
    """Get transforms for CamVid dataset"""
    if split == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def get_cityscapes_transforms():
    """Get transforms for Cityscapes dataset"""
    return A.Compose([
        A.Resize(height=960, width=720),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_dataloaders(batch_size=4, num_workers=4):
    """Get CamVid dataloaders for train and test"""
    # Create datasets
    train_dataset = CamVidDataset(
        root=DATA_ROOT,
        split='train',
        transform=get_camvid_transforms('train')
    )
    
    test_dataset = CamVidDataset(
        root=DATA_ROOT,
        split='test',
        transform=get_camvid_transforms('test')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_cityscapes_loader(root='./data/cityspaces', batch_size=4, num_workers=4):
    """Get Cityscapes dataloader for OOD detection"""
    dataset = CityscapesDataset(
        root=root,
        split='val',
        transform=get_cityscapes_transforms()
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader

class CityscapesDataset(torch.utils.data.Dataset):
    """Cityscapes dataset class"""
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Data paths
        self.images_dir = os.path.join(root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(root, 'gtFine', split)
        
        # Get all image files
        self.images = []
        self.targets = []
        
        if os.path.exists(self.images_dir):
            for city in os.listdir(self.images_dir):
                city_img_dir = os.path.join(self.images_dir, city)
                city_gt_dir = os.path.join(self.targets_dir, city)
                
                if os.path.isdir(city_img_dir):
                    for filename in os.listdir(city_img_dir):
                        if filename.endswith('_leftImg8bit.png'):
                            img_path = os.path.join(city_img_dir, filename)
                            gt_filename = filename.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                            gt_path = os.path.join(city_gt_dir, gt_filename)
                            
                            if os.path.exists(gt_path):
                                self.images.append(img_path)
                                self.targets.append(gt_path)
        
        print(f"Found {len(self.images)} images in {split} split")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        target_path = self.targets[idx]
        
        # Load image and label
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=target)
            image = transformed['image']
            target = transformed['mask']
        
        return image, target

if __name__ == "__main__":
    main()

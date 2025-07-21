import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import bnn
from utils import get_network, get_dataset

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# -------------------------------------------------
CIFAR100_STATS   = {"mean": (0.50707516, 0.48654887, 0.44091784),
                    "std":  (0.26733429, 0.25643846, 0.27615047)}
DATA_ROOT        = os.environ.get("DATASETS_PATH", "./data")
CHECKPOINT_PATH  = "output_resnet18_cifar100_context_128_no_james/epoch: 135-0.763-0.036"
CONFIG_PATH      = "configs/ffg_u_cifar100.json"
NUM_CLASSES      = 100
DEVICE           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_SAMPLES     = 20
BATCH_SIZE       = 103
SCORE_TYPE       = "inducing"   # 1-maxprob | entropy | variance | energy | vae_mse | energy_vae | mahalanobis | inducing
LAMBDA_REG       = 1e-3
PROJECTOR_EPOCHS = 1
PROJECTOR_LR     = 1e-3
# -------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
feat_grad_list = defaultdict(list)
EPS = 1e-4
key_layers = ["layer2.1.conv2", "layer4.1.conv2"]
def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()

def _jittered_cholesky(A, lambda_reg=1e-3, eps=1e-4):
   
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

def compute_mahalanobis_params(net, train_loader, device, T=3):
    """
    Compute Mahalanobis parameters on CIFAR-100 training set with T MC forward passes,
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
                delta = fx.unsqueeze(1) - net.mu.unsqueeze(0)
                B, C, d = delta.shape
                distances = torch.zeros(B, C, device=device)
                for c in range(C):
                    diff_c = delta[:, c, :]         # [B, d]
                    distances[:, c] = torch.sum((diff_c @ net.Sigma_inv) * diff_c, dim=1)
                score = -distances.min(dim=1)[0]
                all_scores.append(score.cpu())
                continue

            # Other types: MC sampling
            logits_list = []
            probs_list  = []
            recons_list = []
            x_last_list = []

            for _ in range(T):
                net.apply(reset_cache) 
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

class FeatureGradientHook:
    def __init__(self):
        self.features = {}
        self.gradients = {}
        
    def feature_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.features[name] = output[0].detach()  
            else:
                self.features[name] = output.detach()
        return hook
        
    def gradient_hook(self, name):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()  
        return hook
        
    def clear(self):
        self.features.clear()
        self.gradients.clear()

def register_hooks(net):
    hooks = []
    hook_handler = FeatureGradientHook()
    

    for name, module in net.named_modules():
        if hasattr(module, 'inducing_mean'):
            if any(key in name for key in key_layers):
                h1 = module.register_forward_hook(hook_handler.feature_hook(name))
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
                torch.manual_seed(SEED)          # Fix seed
                _ = child.sample_parameters()    # Trigger sampling
            inducing_dict[full_name] = child._u_middle.detach()

        # Recursively process next level child modules
        get_inducing_matrices(child, prefix=full_name, inducing_dict=inducing_dict)

    return inducing_dict

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
    net.zero_grad()
    
    outputs = net(features)
    logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs

    pseudo_labels = logits.detach().argmax(dim=1)

    ce_loss = F.cross_entropy(logits, pseudo_labels)
    
    kl = torch.tensor(0., device=features.device)
    for module in net.modules():
        if hasattr(module, "parameter_loss"):
            kl = kl + module.parameter_loss().sum()
    
    loss = ce_loss + kl / len(features) 
    loss.backward()
    
    feat_grad_dict = {}
    for name in hook_handler.features.keys():
        if name not in hook_handler.gradients:
            continue
        f = hook_handler.features[name]
        g = hook_handler.gradients[name]
        
        f_flat = f.view(f.size(0), -1)
        g_flat = g.view(g.size(0), -1)
        feat_grad = f_flat * g_flat
        
        feat_grad_dict[name] = feat_grad
        
    return feat_grad_dict

def fit_projector(net, train_loader, device):
    """Fit feature projector on training set"""
    # 1. Get inducing matrices for all layers
    inducing_dict = get_inducing_matrices(net)
    
    # 2. Register hooks
    hook_handler, hooks = register_hooks(net)
    
    # 3. Collect feature-gradient products
    net.eval()
    feat_grad_list = {}  # Changed to dict instead of defaultdict(list)
    
    i = 0
    with torch.set_grad_enabled(True):  # Need to compute gradients
        for x, y in train_loader:
            i += 1
            if i > 50:
                break
            x, y = x.to(device), y.to(device)
            net.apply(reset_cache)
            
            # Compute feature-gradient products
            feat_grad_dict = compute_feature_grad(x, y, net, hook_handler)
            
            # Collect feature-gradient products for each layer
            for name, feat_grad in feat_grad_dict.items():
                if name in inducing_dict:  # Only collect layers with corresponding inducing matrix
                    if name not in feat_grad_list.keys():
                        feat_grad_list[name] = feat_grad
                    else:
                        feat_grad_list[name] = torch.cat([feat_grad_list[name], feat_grad], dim=0)
            
            hook_handler.clear()
    
    print("\nTesting feat_grad_list shapes:")
    for name in feat_grad_list.keys():
        print(f"{name}: {feat_grad_list[name].shape}")
    
    # 4. Build projector
    in_dims = {name: feat_grad_list[name].shape[1] for name in feat_grad_list.keys()}
    out_dims = {name: U.shape[1] for name, U in inducing_dict.items()}
    print("in_dims", in_dims)
    print("out_dims", out_dims)
    
    projector = FeatureProjector(in_dims, out_dims).to(device) 
    
    # 5. Train projector
    optimizer = torch.optim.Adam(projector.parameters(), lr=PROJECTOR_LR)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300], gamma=0.1)
    projector.train()
    
    for epoch in range(PROJECTOR_EPOCHS):
        total_loss = 0
        num_batches = 0
        loss_list = []
        
        # Train each layer separately
        for name in feat_grad_list.keys():         
            feat_grads = feat_grad_list[name].to(device)
            
            U = inducing_dict[name].to(device)

            # Map to structure space
            f_proj = projector(feat_grads, name)   # [B, n]

            U = inducing_dict[name].to(device)     # [m, n]
            U_t = U.T                              # [n, m]
            A = U @ U_t                            # [m, m]

            L = _jittered_cholesky(A)

            # Correct projection: [B, m] low-dimensional representation
            low_dim_proj = torch.cholesky_solve((f_proj @ U_t).T, L).T

            # Restore to original space
            f_tilde = low_dim_proj @ U

            loss = criterion(f_tilde, f_proj)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loss_list.append(loss.item())
            num_batches += 1
            
        scheduler.step()
        avg_loss = sum(loss_list)/len(loss_list)
    
    return projector, hook_handler

def compute_inducing_scores(net, loader, projector, hook_handler, device):
    """Compute OoD scores using inducing matrix projection"""
    net.eval()
    projector.eval()
    all_scores = []
    
    inducing_dict = get_inducing_matrices(net)
    
    with torch.set_grad_enabled(True):  # Need to compute gradients
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            net.apply(reset_cache)
            
            # Compute feature-gradient products
            feat_grad_dict = compute_feature_grad(x, y, net, hook_handler)
            
            # Compute projection scores for each layer
            layer_scores = []
            for name, feat_grad in feat_grad_dict.items():
                if name not in inducing_dict:
                    continue
                    
                # Map to structure space
                f_proj = projector(feat_grad, name)  # [B, n]
                
                # Compute orthogonal projection
                # f_proj: [B, n]
                U = inducing_dict[name].to(device)      # [m, n]
                U_t = U.T                               # [n, m]
                A = U @ U_t                             # [m, m]

                # Stable inverse
                L = _jittered_cholesky(A)               # [m, m]

                # 1. Project to inducing subspace [B, m]
                low_dim_proj = torch.cholesky_solve((f_proj @ U_t).T, L).T  # [B, m]

                # 2. Return to original space [B, n]
                f_tilde = low_dim_proj @ U              # [B, n]

                # 3. Compute residual
                score = torch.norm(f_proj - f_tilde, dim=1)  # [B]

                layer_scores.append(score)     
            
            if not layer_scores:  # If layer_scores is empty
                print("Warning: No scores computed for this batch")
                continue
                
            # Take mean of all layer scores
            mean_score = torch.stack(layer_scores).mean(0)
            all_scores.append(mean_score.cpu())
            
            # Clear features after use
            hook_handler.clear()
            
    if not all_scores:  # If all_scores is empty
        raise RuntimeError("No scores were computed for any batch")
        
    return torch.cat(all_scores)

def main():

    net = bnn.nn.nets.make_network(
        "resnet18", kernel_size=3,
        remove_maxpool=True, out_features=NUM_CLASSES
    )
    
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    bnn.bayesianize_(net, cfg)
    net = net.cuda()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("module.", "")   # remove `module.`
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict, strict=True)

    # net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu")["model_state_dict"], strict=True)
    net.to(DEVICE)
    # print(net)
    # input()

    # train_loader = get_dataloader("CIFAR100", CIFAR100_STATS, train=True)
    # id_loader = get_dataloader("CIFAR100", CIFAR100_STATS, train=False)
    # # ood_loader = get_dataloader("SVHN", CIFAR100_STATS, train=False)
    # ood_loader = get_dataloader("CIFAR10", CIFAR100_STATS, train=False)

    train_data, test_data, num_classes = get_dataset("CIFAR100")
    ood_train_data, ood_test_data, _ = get_dataset("CIFAR10")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    id_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    ood_loader = DataLoader(ood_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    if SCORE_TYPE == "inducing":
        projector_path = "feature.pth"
        
        if os.path.exists(projector_path):
            print("Loading saved feature projector...")
            checkpoint = torch.load(projector_path)
            projector = FeatureProjector(checkpoint['in_dims'], checkpoint['out_dims']).to(DEVICE)
            projector.load_state_dict(checkpoint['state_dict'])
            hook_handler, _ = register_hooks(net)
        else:
            print("Fitting feature projector on train set...")
            projector, hook_handler = fit_projector(net, train_loader, DEVICE)
            
            # print("Saving feature projector...")
            # torch.save({
            #     'state_dict': projector.state_dict(),
            #     'in_dims': projector.in_dims,
            #     'out_dims': projector.out_dims
            # }, projector_path)
        
        print("Computing inducing-based OoD scores...")
        id_scores = compute_inducing_scores(net, id_loader, projector, hook_handler, DEVICE)
        ood_scores = compute_inducing_scores(net, ood_loader, projector, hook_handler, DEVICE)
        
        print("\nScore Statistics:")
        print(f"ID scores - Mean: {id_scores.mean()}, Std: {id_scores.std()}")
        print(f"ID scores - Min: {id_scores.min()}, Max: {id_scores.max()}")
        print(f"OOD scores - Mean: {ood_scores.mean()}, Std: {ood_scores.std()}")
        print(f"OOD scores - Min: {ood_scores.min()}, Max: {ood_scores.max()}")

    else:
        if SCORE_TYPE == "mahalanobis":
            print("Computing Mahalanobis parameters on train set...")
            compute_mahalanobis_params(net, train_loader, DEVICE)
            
        print(f"[{SCORE_TYPE}] MC samples = {TEST_SAMPLES}")
        id_scores = get_ood_scores(net, id_loader, DEVICE, TEST_SAMPLES, SCORE_TYPE)
        ood_scores = get_ood_scores(net, ood_loader, DEVICE, TEST_SAMPLES, SCORE_TYPE)

    id_scores = id_scores.detach().cpu().numpy()
    ood_scores = ood_scores.detach().cpu().numpy()
    
    id_filename = f'id_scores_{SCORE_TYPE}.txt'
    ood_filename = f'ood_scores_{SCORE_TYPE}.txt'
    
    np.savetxt(id_filename, id_scores)
    np.savetxt(ood_filename, ood_scores)
    
    print(f"ID scores saved to: {id_filename}")
    print(f"OoD scores saved to: {ood_filename}")
    
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(y_true, y_score)
    print(f"\nAUROC ({SCORE_TYPE}): {auroc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.hist(id_scores, bins=50, alpha=0.5, label='ID (CIFAR100)', density=True)
    plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD (CIFAR10)', density=True)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title('Distribution of ID and OOD Scores')
    plt.legend()
    plt.savefig('score_distribution.png')
    plt.close()

if __name__ == "__main__":
    main()

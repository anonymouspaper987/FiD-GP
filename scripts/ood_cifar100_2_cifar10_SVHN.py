import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from utils import get_dataset

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
# Remove global random seed setting, only set during inference
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
        # print(full_name)
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
    """Compute feature gradients"""
    # Clear previous gradients
    net.zero_grad()
    
    # Forward pass
    outputs = net(features)
    # If network forward returns multiple values (e.g., (features, recon, logits)), take the last as logits
    logits = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs

    # Use model's own predictions (pseudo-labels)
    pseudo_labels = logits.detach().argmax(dim=1)

    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(logits, pseudo_labels)
    
    # Compute KL divergence loss (similar to training phase)
    kl = torch.tensor(0., device=features.device)
    for module in net.modules():
        if hasattr(module, "parameter_loss"):
            kl = kl + module.parameter_loss().sum()
    
    # Total loss = cross-entropy + KL divergence (using same weights as training phase)
    loss = ce_loss + kl / len(features)  # Normalize by batch size
    loss.backward()
    
    # 计算特征-梯度乘积
    feat_grad_dict = {}
    for name in hook_handler.features.keys():
        if name not in hook_handler.gradients:
            continue
        f = hook_handler.features[name]
        g = hook_handler.gradients[name]
        
        # 将特征和梯度展平并相乘
        f_flat = f.view(f.size(0), -1)
        g_flat = g.view(g.size(0), -1)
        feat_grad = f_flat * g_flat
        
        feat_grad_dict[name] = feat_grad
        
    return feat_grad_dict

def fit_projector(net, train_loader, device):
    """在训练集上拟合特征映射器"""
    # 1. 获取所有层的inducing matrices
    inducing_dict = get_inducing_matrices(net)
    # print("len inducing_dict", len(inducing_dict))
    
    # 2. 注册hooks
    hook_handler, hooks = register_hooks(net)
    
    # 3. 收集特征-梯度乘积
    net.eval()
    feat_grad_list = {}  # 改为字典而不是defaultdict(list)
    
    # 先收集整个训练集的特征-梯度乘积
    i = 0
    with torch.set_grad_enabled(True):  # 需要计算梯度
        for x, y in train_loader:
            i += 1
            if i > 50:
                break
            # break
            x, y = x.to(device), y.to(device)
            net.apply(reset_cache)
            
            # 计算特征-梯度乘积
            feat_grad_dict = compute_feature_grad(x, y, net, hook_handler)
            # print(f"feat_grad_dict: {feat_grad_dict.keys()}")
            # 收集每层的特征-梯度乘积
            for name, feat_grad in feat_grad_dict.items():
                if name in inducing_dict:  # 只收集有对应inducing matrix的层
                    # print(f"feat_grad {name}: {feat_grad.shape}")
                    if name not in feat_grad_list.keys():
                        feat_grad_list[name] = feat_grad
                    else:
                        feat_grad_list[name] = torch.cat([feat_grad_list[name], feat_grad], dim=0)
            
            # 在收集完特征后再清空
            hook_handler.clear()
    
    # 测试feat_grad_list的形状
    print("\nTesting feat_grad_list shapes:")
    for name in feat_grad_list.keys():
        print(f"{name}: {feat_grad_list[name].shape}")
    # input()
    # 4. 构建映射器
    in_dims = {name: feat_grad_list[name].shape[1] for name in feat_grad_list.keys()}
    out_dims = {name: U.shape[1] for name, U in inducing_dict.items()}
    print("in_dims", in_dims)
    print("out_dims", out_dims)
    # input()
    projector = FeatureProjector(in_dims, out_dims).to(device) 
    
    # 5. 训练映射器
    optimizer = torch.optim.Adam(projector.parameters(), lr=PROJECTOR_LR)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300], gamma=0.1)
    projector.train()
    for epoch in range(PROJECTOR_EPOCHS):
        total_loss = 0
        num_batches = 0
        loss_list = []
        # 对每个层分别训练
        for name in feat_grad_list.keys():         
            # 直接使用收集好的特征-梯度乘积
            feat_grads = feat_grad_list[name].to(device)
            # print(f"\nEpoch {epoch+1}, Layer {name}:")
            # print(f"Feature shape: {feat_grads.shape}")
            
            U = inducing_dict[name].to(device)

            # 前向传播
            f_proj = projector(feat_grads, name)
            
            # U: [M, N], F: [B, N], lambda_: scalar
            # UUt = U @ U.T                              # [M, M]
            # UUt_reg = UUt + LAMBDA_REG*torch.eye(UUt.shape[0], device=U.device)
            # L = torch.linalg.cholesky(UUt_reg)         # [M, M]

            # # 构造投影算子 P  = U^T (U U^T + λI)^(-1) U
            # # 先解 (U U^T + λI) X = U  →  X = (U U^T + λI)^(-1) U
            # X = torch.cholesky_solve(U, L)             # [M, N]
            # P = U.T @ X                                 # [N, N]
            # # print("P",P.shape) # 128,128
            # # 将 F 投影到行空间
            # f_tilde = f_proj @ P                              # [B, N]
            f_proj = projector(feat_grads, name)   # [B, n]

            U = inducing_dict[name].to(device)     # [m, n]
            U_t = U.T                              # [n, m]
            A = U @ U_t                            # [m, m]

            L = _jittered_cholesky(A)

            # 正确投影：[B, m] 低维表示
            low_dim_proj = torch.cholesky_solve((f_proj @ U_t).T, L).T

            # 还原到原始空间
            f_tilde = low_dim_proj @ U

            loss = criterion(f_tilde, f_proj)
            
            # 反向传播
            # print("name",loss.item())
            # input()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loss_list.append(loss.item())
            num_batches += 1
            
            # print(f"Loss: {loss.item():.6f}")
        scheduler.step()
        avg_loss = sum(loss_list)/len(loss_list)
        # print(f"\nEpoch {epoch+1}/{PROJECTOR_EPOCHS}, Average Loss: {avg_loss}")
    # input()
    
    return projector, hook_handler

# ---------- 计算OoD分数 ----------
def compute_inducing_scores(net, loader, projector, hook_handler, device):
    """使用inducing matrix投影计算OoD分数"""
    net.eval()
    projector.eval()
    all_scores = []
    
    # 获取inducing matrices
    inducing_dict = get_inducing_matrices(net)
    # print("len inducing_dict", len(inducing_dict))
    with torch.set_grad_enabled(True):  # 需要计算梯度
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            net.apply(reset_cache)
            
            # 计算特征-梯度乘积
            feat_grad_dict = compute_feature_grad(x, y, net, hook_handler)
            
            # 计算每层的投影分数
            layer_scores = []
            # print("len feat_grad_dict", len(feat_grad_dict))
            for name, feat_grad in feat_grad_dict.items():
                if name not in inducing_dict:
                    continue
                    
               
                # 映射到结构空间
                f_proj = projector(feat_grad, name)  # [B, n]
                
                # 计算正交投影
                # f_proj: [B, n]
                U = inducing_dict[name].to(device)      # [m, n]
                U_t = U.T                               # [n, m]
                A = U @ U_t                             # [m, m]

                # 稳定求逆
                L = _jittered_cholesky(A)               # [m, m]

                # 1. 投影到 inducing 子空间 [B, m]
                low_dim_proj = torch.cholesky_solve((f_proj @ U_t).T, L).T  # [B, m]

                # 2. 返回到原空间 [B, n]
                f_tilde = low_dim_proj @ U              # [B, n]

                # 3. 计算残差
                score = torch.norm(f_proj - f_tilde, dim=1)  # [B]

                layer_scores.append(score)     
            
            
            if not layer_scores:  # 如果layer_scores为空
                print("Warning: No scores computed for this batch")
                continue
                
            # 取所有层分数的均值
            mean_score = torch.stack(layer_scores).mean(0)
            all_scores.append(mean_score.cpu())
            
            # 在使用完特征后再清空
            hook_handler.clear()
            
    if not all_scores:  # 如果all_scores为空
        raise RuntimeError("No scores were computed for any batch")
        
    return torch.cat(all_scores)

# ---------- 主流程 ----------
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CIFAR-100 to CIFAR-10/SVHN OOD Detection')
    parser.add_argument("--model-path", default="output_resnet18_cifar100_context_128_no_james/epoch: 135-0.763-0.036", 
                       help="Path to the trained model weights")
    parser.add_argument("--config-path", default="configs/ffg_u_cifar100.json", 
                       help="Path to inference config file")
    parser.add_argument("--data-root", default=DATA_ROOT, 
                       help="Path to datasets")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, 
                       help="Batch size for evaluation")
    parser.add_argument("--test-samples", type=int, default=TEST_SAMPLES, 
                       help="Number of MC samples for test-time prediction")
    parser.add_argument("--score-type", default=SCORE_TYPE, 
                       choices=["1-maxprob", "entropy", "variance", "energy", "vae_mse", "energy_vae", "mahalanobis", "inducing"], 
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
    
    # 1. 构建网络并加载权重
    net = bnn.nn.nets.make_network(
        "resnet18", kernel_size=3,
        remove_maxpool=True, out_features=NUM_CLASSES
    )
    
    with open(args.config_path) as f:
        cfg = json.load(f)
    bnn.bayesianize_(net, cfg)
    net = net.cuda()
    ckpt = torch.load(args.model_path, map_location="cpu")

    # 2. 去除 module. 前缀
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("module.", "")   # remove `module.`
        new_state_dict[name] = v

    # 3. 用 clean 过的 state_dict 加载
    net.load_state_dict(new_state_dict, strict=True)
    net.to(DEVICE)

    # 2. 准备数据
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
            # 如果存在保存的映射器，直接加载
            print("Loading saved feature projector...")
            checkpoint = torch.load(projector_path)
            projector = FeatureProjector(checkpoint['in_dims'], checkpoint['out_dims']).to(DEVICE)
            projector.load_state_dict(checkpoint['state_dict'])
            hook_handler, _ = register_hooks(net)
        else:
            # 3. 在训练集上拟合映射器
            print("Fitting feature projector on train set...")
            projector, hook_handler = fit_projector(net, train_loader, DEVICE)
            
            # 保存映射器
            # print("Saving feature projector...")
            # torch.save({
            #     'state_dict': projector.state_dict(),
            #     'in_dims': projector.in_dims,
            #     'out_dims': projector.out_dims
            # }, projector_path)
        
        # 4. 计算ID和OoD分数
        print("Computing inducing-based OoD scores...")
        id_scores = compute_inducing_scores(net, id_loader, projector, hook_handler, DEVICE)
        ood_scores = compute_inducing_scores(net, ood_loader, projector, hook_handler, DEVICE)
        
        # 添加调试信息
        print("\nScore Statistics:")
        print(f"ID scores - Mean: {id_scores.mean()}, Std: {id_scores.std()}")
        print(f"ID scores - Min: {id_scores.min()}, Max: {id_scores.max()}")
        print(f"OOD scores - Mean: {ood_scores.mean()}, Std: {ood_scores.std()}")
        print(f"OOD scores - Min: {ood_scores.min()}, Max: {ood_scores.max()}")

    else:
        # 使用其他OoD检测方法
        if SCORE_TYPE == "mahalanobis":
            print("Computing Mahalanobis parameters on train set...")
            compute_mahalanobis_params(net, train_loader, DEVICE)
            
        print(f"[{SCORE_TYPE}] MC samples = {TEST_SAMPLES}")
        id_scores = get_ood_scores(net, id_loader, DEVICE, TEST_SAMPLES, SCORE_TYPE)
        ood_scores = get_ood_scores(net, ood_loader, DEVICE, TEST_SAMPLES, SCORE_TYPE)

    # 5. 评估AUROC
    # 确保转换为numpy之前detach
    id_scores = id_scores.detach().cpu().numpy()
    ood_scores = ood_scores.detach().cpu().numpy()
    
    # 保存ID和OoD分数到不同的txt文件
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

    # 添加分数分布的可视化
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

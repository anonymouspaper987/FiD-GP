import argparse
from collections import defaultdict
import json
import os
import pickle
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.utils.data as data
import torchvision
from torchvision import datasets
import torchvision.transforms as tf
import bnn
from bnn.calibration import calibration_curve, expected_calibration_error as ece

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

ROOT = os.environ.get("DATASETS_PATH", "./data")
NUM_BINS = 100

def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class MMCE(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, probs, targets):
        # Convert targets to one-hot
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=probs.size(1)).float()
        
        # Calculate confidence and accuracy
        confidence = torch.max(probs, dim=1)[0]
        accuracy = (probs.argmax(dim=1) == targets).float()
        
        # Calculate MMCE
        n = probs.size(0)
        mmce = 0.0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Gaussian kernel
                    kernel = torch.exp(-0.5 * ((confidence[i] - confidence[j]) ** 2))
                    mmce += kernel * (accuracy[i] - confidence[i]) * (accuracy[j] - confidence[j])
        
        return mmce / (n * (n-1))

class Prefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self._preload()
    
    def _preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.next_input is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        input, target = self.next_input, self.next_target
        self._preload()
        return input, target

def main():
    def list_of_ints(s):
        return list(map(int, s.split(",")))

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--train-samples", type=int, default=1)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--annealing-epochs", type=int, default=50)
    parser.add_argument("--ml-epochs", type=int, default=100)
    parser.add_argument("--inference-config",  default="configs/ffg_u_imagenet.json")
    parser.add_argument("--output-dir", default="output_resnet50_imagenet")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--milestones", type=list_of_ints, default = [100, 180])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--resnet", type=int, default=18, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--local_rank', dest='local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', dest='local_rank', type=int, default=-1)
    parser.add_argument("--dist-url", default='tcp://127.0.0.1:23456', type=str)
    parser.add_argument("--dist-backend", default='nccl', type=str)
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--multiprocessing-distributed", action="store_true")
    parser.add_argument("--batch-size", type=int, default=110)
    parser.add_argument("--test-batch-size", type=int, default=110)
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--focal-loss", action="store_true", help="Use focal loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling factor")
    parser.add_argument("--mmce-loss", action="store_true", help="Use MMCE loss")
    parser.add_argument("--mmce-weight", type=float, default=0.1, help="Weight for MMCE loss")
    # Data loading optimization parameters
    parser.add_argument("--num-workers", type=int, default=16, help="Number of data loading workers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument("--use-prefetcher", action="store_true", help="Use data prefetcher")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    #  Performance optimization: disable strict determinism, allow cuDNN to find the fastest algorithm automatically
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True  # Enable TF32 (Ampere+)

    ngpus_per_node = torch.cuda.device_count()
    args.distributed = ngpus_per_node > 1
    main_worker(args.local_rank, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")
    
    # Performance optimization: fix distributed training initialization
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=ngpus_per_node,
            rank=args.local_rank
        )
        print(f"Distributed training initialized: rank {args.local_rank}/{ngpus_per_node}")
    
    device = torch.device(f"cuda:{args.gpu}")
    
    # Only create output directory in the main process
    if not args.distributed or dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_txt = os.path.join(args.output_dir, args.output_dir + ".txt")
        snapshot_sd_path = os.path.join(args.output_dir, "snapshot_sd.pt")
        snapshot_optim_path = os.path.join(args.output_dir, "snapshot_optim.sd")
        metrics_path = os.path.join(args.output_dir, "metrics.pkl")
    else:
        log_txt = None
        snapshot_sd_path = None
        snapshot_optim_path = None
        metrics_path = None

    # ImageNet data loading
    root = "data/imagenet"
    if not os.path.exists(os.path.join(root, "train")) or not os.path.exists(os.path.join(root, "val")):
        print(f"Error: ImageNet dataset not found in {root}")
        print("Please ensure the following directories exist:")
        print(f"- {os.path.join(root, 'train')}")
        print(f"- {os.path.join(root, 'val')}")
        return

    if not args.distributed or dist.get_rank() == 0:
        print(f"Loading ImageNet dataset from: {root}")
        print(f"Using batch size: {args.batch_size} per GPU")
        print(f"Performance optimizations: Workers={args.num_workers}, Prefetcher={args.use_prefetcher}")

    train_tf = tf.Compose([
        tf.RandomResizedCrop(224),
        tf.RandomHorizontalFlip(),
        tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = tf.Compose([
        tf.Resize(256),
        tf.CenterCrop(224),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.ImageFolder(root=f'{root}/train', transform=train_tf)
    test_data = datasets.ImageFolder(root=f'{root}/val', transform=val_tf)

    train_sampler = DistributedSampler(train_data) if args.distributed else None
    test_sampler = DistributedSampler(test_data, shuffle=False) if args.distributed else None
    
    # Performance optimization: increase num_workers and add prefetcher
    train_loader = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep worker processes
        prefetch_factor=args.prefetch_factor  # Prefetch factor
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor
    )
    
    acc_best = 0.0
    num_classes = 1000
    net = bnn.nn.nets.make_network_with_pretrain(architecture = f"resnet{args.resnet}", out_features = num_classes)
    
    if args.inference_config is not None:
        with open(args.inference_config) as f:
            cfg = json.load(f)
        bnn.bayesianize_(net, cfg)

    net.to(device)
    
    # Performance optimization: optimize DDP configuration
    if args.distributed:
        net = DDP(
            net,
            device_ids=[args.gpu],
            find_unused_parameters=False,  # If all parameters are used
            gradient_as_bucket_view=True,  # Memory optimization
            broadcast_buffers=False  # If BN synchronization is not needed
        )
        
    if args.optimizer == "adam":
        optim = torch.optim.Adam(net.parameters(), args.lr)
    elif args.optimizer == "sgd":
        optim = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum)
    else:
        raise RuntimeError("Unknown optimizer:", args.optimizer)
    
    metrics = defaultdict(list)
   
    if args.milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, args.milestones, gamma=args.gamma)
    else:
        scheduler = None
    kl_factor = 0. if args.ml_epochs > 0 or args.annealing_epochs > 0 else 1.
    annealing_rate = args.annealing_epochs ** -1 if args.annealing_epochs > 0 else 1.
    smoothing = 0.02
    
    # Progress bar optimization: only show progress bar in main process
    show_progress = not args.distributed or dist.get_rank() == 0
    
    # Initialize losses
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    if args.mmce_loss:
        mmce_criterion = MMCE(device)
    
    # Progress bar optimization: create main progress bar
    if show_progress:
        epoch_pbar = trange(0, args.num_epochs, desc="Training Progress", 
                           unit="epoch", position=0, leave=True)
    else:
        epoch_pbar = range(1, args.num_epochs)
    
    for i in epoch_pbar:
        if train_sampler is not None:
            train_sampler.set_epoch(i)
        epoch_start_time = time.time()
        net.train()
        net.apply(reset_cache)
        
        # Progress bar optimization: create batch progress bar
        if show_progress:
            batch_pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {i+1}/{args.num_epochs}",
                unit="batch",
                position=1,
                leave=False,
                ncols=100
            )
        
        # Performance optimization: use prefetcher
        if args.use_prefetcher:
            batch_iter = Prefetcher(train_loader, device)
        else:
            batch_iter = iter(train_loader)
        
        epoch_loss = 0.0
        epoch_kl = 0.0
        batch_count = 0
        
        for j, (x, y) in enumerate(batch_iter):
            # Performance optimization: non-blocking data transfer
            if not args.use_prefetcher:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            
            # Performance optimization: more efficient gradient zeroing
            optim.zero_grad(set_to_none=True)
            avg_nll = 0.
            batch_kl = 0.0
            
            for k in range(args.train_samples):
                yhat = net(x)
                yhat = yhat / args.temperature
                
                if args.focal_loss:
                    loss = criterion(yhat, y)
                else:
                    log_preds = torch.nn.functional.log_softmax(yhat, dim=-1)
                    y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).float().to(yhat.device)
                    y_smoothed = y_onehot * (1 - args.label_smoothing) + args.label_smoothing / num_classes
                    nll = -(y_smoothed * log_preds).sum(dim=-1).mean() / args.train_samples
                    loss = nll
                
                
                if k == 0:
                    if i >= args.ml_epochs:
                        kl = torch.tensor(0., device=device)
                        for module in net.modules():
                            if hasattr(module, "parameter_loss"):
                                kl = kl + module.parameter_loss().sum()
                        batch_kl = kl.item()
                        loss = loss + kl * kl_factor / len(train_data)
                    else:
                        batch_kl = 0.0
                
                avg_nll += loss.item()
                loss.backward(retain_graph=args.train_samples > 1)
            
            optim.step()
            net.apply(reset_cache)
            metrics["nll"].append(avg_nll)
            metrics["kl"].append(batch_kl)
            
            epoch_loss += avg_nll
            epoch_kl += batch_kl
            batch_count += 1
            
            # Progress bar optimization: update batch progress bar
            if show_progress:
                avg_loss = epoch_loss / batch_count
                avg_kl = epoch_kl / batch_count
                current_lr = optim.param_groups[0]['lr']
                
                batch_pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'KL': f'{avg_kl:.4f}',
                    'LR': f'{current_lr:.2e}'
                })
                batch_pbar.update(1)
        
        # Close batch progress bar
        if show_progress:
            batch_pbar.close()
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        avg_epoch_loss = epoch_loss / batch_count
        avg_epoch_kl = epoch_kl / batch_count
        
        if not args.distributed or dist.get_rank() == 0:
            print(f"Epoch {i+1}/{args.num_epochs} - Time: {epoch_time:.2f}s - Loss: {avg_epoch_loss:.4f} - KL: {avg_epoch_kl:.4f}")
            if i == 0:
                print(f"--> Estimated time per epoch: {epoch_time:.2f} seconds. Validation will run every epoch.")
        
        if scheduler is not None:
            scheduler.step()
            
        # Run validation every epoch
        if show_progress:
            print(f"\n{'='*50}")
            print(f"Starting validation at epoch {i+1}")
            print(f"{'='*50}")
            
            val_pbar = tqdm(
                total=len(test_loader),
                desc="Validation",
                unit="batch",
                position=1,
                leave=False,
                ncols=100
            )
        
        net.eval()
        all_probs = []
        all_targets = []
        val_batch_count = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                probs = []
                for _ in range(args.test_samples):
                    probs.append(net(x).softmax(-1))
                    net.apply(reset_cache)
                avg_probs = sum(probs).div(args.test_samples)
                all_probs.append(avg_probs)
                all_targets.append(y.to(device))
                val_batch_count += 1
                
                if show_progress:
                    val_pbar.update(1)
        
        if show_progress:
            val_pbar.close()
        
        probs = torch.cat(all_probs, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Synchronize results in distributed environment
        if args.distributed:
            # Gather results from all processes
            probs_list = [torch.zeros_like(probs) for _ in range(dist.get_world_size())]
            targets_list = [torch.zeros_like(targets) for _ in range(dist.get_world_size())]
            dist.all_gather(probs_list, probs)
            dist.all_gather(targets_list, targets)
            
            # Use results from main process
        if not args.distributed or dist.get_rank() == 0:
            if args.distributed:
                probs = torch.cat(probs_list, dim=0)
                targets = torch.cat(targets_list, dim=0)
            # Move to CPU for computation
        
            accuracy = (probs.argmax(-1) == targets).float().mean().item()
            p, f, w = calibration_curve(probs.cpu().numpy(), targets.cpu().numpy(), NUM_BINS)
            ece_value = ece(p, f, w)
            
            if show_progress:
                epoch_pbar.set_postfix({
                    'Acc': f'{accuracy:.4f}',
                    'ECE': f'{ece_value:.4f}',
                    'Best': f'{acc_best:.4f}'
                })
            
            print(f"Validation at Epoch {i+1} - Test accuracy: {accuracy:.4f}, ECE: {ece_value:.4f}")
                
            if accuracy > acc_best:
                acc_best = accuracy
                torch.save(net.state_dict(), os.path.join(args.output_dir, f"epoch: {i}-{accuracy:.3f}-{ece_value:.3f}"))
                if show_progress:
                    print(f"🎉 New best accuracy: {accuracy:.4f} (previous: {acc_best:.4f})")
            
            with open(log_txt, "a") as f:
                f.write(f"epoch {i}, acc: {accuracy}, ece: {ece_value} \n")
            if args.verbose:
                print(f"Epoch {i} -- Accuracy: {100 * accuracy:.2f}")
            if args.output_dir is not None:
                torch.save(net.state_dict(), snapshot_sd_path)
                torch.save(optim.state_dict(), snapshot_optim_path)
                with open(metrics_path, "wb") as fn:
                    pickle.dump(metrics, fn)
                        
        if i >= args.ml_epochs:
            kl_factor = min(1., kl_factor + annealing_rate)
    
    # Close main progress bar
    if show_progress:
        epoch_pbar.close()
        print(f"\n{'='*50}")
        print(f"Training completed! Best accuracy: {acc_best:.4f}")
        print(f"{'='*50}")

if __name__ == '__main__':
    main()

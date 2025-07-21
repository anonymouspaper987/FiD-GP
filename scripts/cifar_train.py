import argparse
from collections import defaultdict
import json
import os
import pickle
import time
from optimizers import LARS, LAMB
import numpy as np

from tqdm import tqdm, trange
# add
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.distributions as dist
import torch.utils.data as data

import torchvision
import torchvision.transforms as tf


import bnn
from bnn.calibration import calibration_curve, expected_calibration_error as ece
from bnn.nn.mixins.variational.inducing_simple2 import InducingMixin


STATS = {
    "CIFAR10": {"mean": (0.49139968, 0.48215841, 0.44653091), "std": (0.24703223, 0.24348513, 0.26158784)},
    "CIFAR100": {"mean": (0.50707516, 0.48654887, 0.44091784), "std": (0.26733429, 0.25643846, 0.27615047)}
}
ROOT = os.environ.get("DATASETS_PATH", "./data")
NUM_BINS = 100


def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()


def main(seed, num_epochs, inference_config, output_dir, ml_epochs, annealing_epochs, train_samples, test_samples,
         verbose, progress_bar, lr, cifar, optimizer, momentum, milestones, gamma, resnet):
    torch.manual_seed(seed)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    log_txt = os.path.join(output_dir, output_dir + ".txt")
    # set up data loaders
    dataset_name = f"CIFAR{cifar}"
    dataset_cls = getattr(torchvision.datasets, dataset_name)
    root = f"{ROOT}/{dataset_name.lower()}"
    print(f"Loading dataset {dataset_cls} from {root}")
    aug_tf = [tf.RandomCrop(32, padding=4, padding_mode="reflect"), tf.RandomHorizontalFlip()]
    norm_tf = [tf.ToTensor(), tf.Normalize(**STATS[dataset_name])]
    train_data = dataset_cls(root, train=True, transform=tf.Compose(aug_tf + norm_tf), download=True)
    test_data = dataset_cls(root, train=False, transform=tf.Compose(norm_tf), download=True)

    train_loader = data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = data.DataLoader(test_data, batch_size=200)
    acc_best = 0.0
    num_classes = cifar
    # set up net and optimizer
    net = bnn.nn.nets.make_network(f"resnet{resnet}", kernel_size=3, remove_maxpool=True, out_features=cifar)
    if inference_config is not None:
        with open(inference_config) as f:
            cfg = json.load(f)
        bnn.bayesianize_(net, cfg)
 
    net.to(device)

    if optimizer == "adam":
        optim = torch.optim.Adam(net.parameters(), lr)
    elif optimizer == "sgd":
        optim = torch.optim.SGD(net.parameters(), lr, momentum=momentum)
    else:
        raise RuntimeError("Unknown optimizer:", optimizer)

    # set up dict for tracking losses and load state dicts if applicable
    metrics = defaultdict(list)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        snapshot_sd_path = os.path.join(output_dir, "snapshot_sd.pt")
        snapshot_optim_path = os.path.join(output_dir, "snapshot_optim.sd")
        metrics_path = os.path.join(output_dir, "metrics.pkl")
       
    else:
        snapshot_sd_path = None
        snapshot_optim_path = None
        metrics_path = None
        
    last_epoch = len(metrics["acc"]) - 1
    
    if milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones, gamma=gamma, last_epoch=last_epoch)
    else:
        scheduler = None

    kl_factor = 0. if ml_epochs > 0 or annealing_epochs > 0 else 1.
    annealing_rate = annealing_epochs ** -1 if annealing_epochs > 0 else 1.
    smoothing = 0.05
    epoch_iter = trange(last_epoch + 1, num_epochs, desc="Epochs") if progress_bar else range(last_epoch + 1, num_epochs)
    for i in epoch_iter:
        net.train()
        net.apply(reset_cache)
        batch_iter = tqdm(iter(train_loader), desc="Batches") if progress_bar else iter(train_loader)
        
        for j, (x, y) in enumerate(batch_iter):
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            avg_nll = 0.
            for k in range(train_samples):
                yhat = net(x)
                log_preds = torch.nn.functional.log_softmax(yhat, dim=-1)
                y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).float().to(yhat.device)
                y_smoothed = y_onehot * (1 - smoothing) + smoothing / num_classes
                nll = -(y_smoothed * log_preds).sum(dim=-1).mean() / train_samples
                if k == 0:
                    kl = torch.tensor(0., device=device)
                    for module in net.modules():
                        if hasattr(module, "parameter_loss"):
                            kl = kl + module.parameter_loss().sum()
                    metrics["kl"].append(kl.item())
                    loss = nll + kl * kl_factor / len(train_data)
                else:
                    loss = nll
                    
                avg_nll += nll.item()
                loss.backward(retain_graph=train_samples > 1)

            optim.step()
            
            net.apply(reset_cache)
            
            metrics["nll"].append(avg_nll)

        if scheduler is not None:
            scheduler.step()

        net.eval()
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                # Multiple samples per batch
                probs = []
                for _ in range(test_samples):
                    probs.append(net(x).softmax(-1))
                    net.apply(reset_cache)
                # Compute average probability
                avg_probs = sum(probs).div(test_samples)
                # Move results to CPU and store
                all_probs.append(avg_probs.cpu())
                all_targets.append(y)
        

        probs = torch.cat(all_probs, dim=0)
        targets = torch.cat(all_targets, dim=0)

  
        accuracy = (probs.argmax(-1) == targets).float().mean().item()
        # print(f"epoch {i}, Test accuracy: {accuracy:.4f}")

        # Compute ECE
        p, f, w = calibration_curve(probs.numpy(), targets.numpy(), NUM_BINS)
        ece_value = ece(p, f, w)
        # print(f"Expected Calibration Error: {ece_value:.4f}")

        if accuracy > acc_best: 
            acc_best = accuracy
            torch.save(net.state_dict(), os.path.join(output_dir, f"epoch: {i}-{accuracy:.3f}-{ece_value:.3f}"))
        with open(log_txt, "a") as f:
            f.write(f"epoch {i}, acc: {accuracy}, ece: {ece_value} \n")

        
        print(f"Epoch {i} -- Accuracy: {100 * accuracy:.4f}")
        print(f"Expected Calibration Error: {ece_value:.4f}")

        if output_dir is not None:
            torch.save(net.state_dict(), snapshot_sd_path)
            torch.save(optim.state_dict(), snapshot_optim_path)
            with open(metrics_path, "wb") as fn:
                pickle.dump(metrics, fn)

    print(f"Final test accuracy: {100 * accuracy:.2f}")
    
    if output_dir is not None:
        bin_width = NUM_BINS ** -1
        bin_centers = np.linspace(bin_width / 2, 1 - bin_width / 2, NUM_BINS)

        plt.figure(figsize=(5, 5))
        plt.plot([0, 100], [0, 100], color="black", linestyle="dashed", alpha=0.5)
        plt.plot(100 * p[w > 0], 100 * f[w > 0], marker="o", markersize=8)
        plt.bar(100 * bin_centers[w > 0], 100 * w[w > 0], width=100 * bin_width, alpha=0.5)
        plt.xlabel("Mean probability predicted")
        plt.ylabel("Empirical accuracy")
        plt.title(f"Calibration curve (Accuracy={100 * accuracy:.2f})")
        plt.savefig(os.path.join(output_dir, "calibration.png"), bbox_inches="tight")


if __name__ == '__main__':
    def list_of_ints(s):
        return list(map(int, s.split(",")))

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--train-samples", type=int, default=1)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--annealing-epochs", type=int, default=50)
    parser.add_argument("--ml-epochs", type=int, default=100)
    parser.add_argument("--inference-config", default="configs/ffg_u_cifar100.json")
    parser.add_argument("--output-dir", default="output_resnet18_cifar100_context_128_no_james")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cifar", type=int, default=10, choices=[10, 100])
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--milestones", type=list_of_ints, default="100")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--resnet", type=int, default=18, choices=[18, 34, 50, 101, 152])

    args = parser.parse_args()
    if args.verbose:
        print(vars(args))
    main(**vars(args))

import argparse
import json
import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as tf

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bnn.calibration import calibration_curve, expected_calibration_error as ece

STATS = {
    "CIFAR10": {"mean": (0.49139968, 0.48215841, 0.44653091), "std": (0.24703223, 0.24348513, 0.26158784)},
    "CIFAR100": {"mean": (0.50707516, 0.48654887, 0.44091784), "std": (0.26733429, 0.25643846, 0.27615047)}
}
ROOT = os.environ.get("DATASETS_PATH", "./data")
NUM_BINS = 100

def load_model(model_path, config_path, cifar=100, resnet=18):
    import bnn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = bnn.nn.nets.make_network(f"resnet{resnet}", kernel_size=3, remove_maxpool=True, out_features=cifar)
    with open(config_path) as f:
        cfg = json.load(f)
    bnn.bayesianize_(net, cfg)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net, device

def get_test_loader(cifar=100, batch_size=200):
    dataset_name = f"CIFAR{cifar}"
    dataset_cls = getattr(torchvision.datasets, dataset_name)
    norm_tf = [tf.ToTensor(), tf.Normalize(**STATS[dataset_name])]
    test_data = dataset_cls(f"{ROOT}/{dataset_name.lower()}", train=False, transform=tf.Compose(norm_tf), download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader

def evaluate(net, device, test_loader, test_samples=8):
    all_probs = []
    all_targets = []
    nll_total = 0.0
    n_samples = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            probs = []
            nll_batch = 0.0
            for _ in range(test_samples):
                logits = net(x)
                prob = torch.softmax(logits, dim=-1)
                probs.append(prob)
                nll_batch += torch.nn.functional.nll_loss(torch.log(prob), y, reduction='sum').item()
            avg_probs = sum(probs) / test_samples
            all_probs.append(avg_probs.cpu())
            all_targets.append(y.cpu())
            nll_total += nll_batch / test_samples
            n_samples += y.size(0)
    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    accuracy = (probs.argmax(-1) == targets).mean()
    p, f, w = calibration_curve(probs, targets, NUM_BINS)
    ece_value = ece(p, f, w)
    nll = nll_total / n_samples
    return accuracy, nll, ece_value

def main():
    parser = argparse.ArgumentParser(description='CIFAR Prediction with FiD-GP')
    parser.add_argument("--model-path", default="../weights/cifar100/epoch-135-0.763-0.036.pth", 
                       help="Path to the trained model weights")
    parser.add_argument("--config-path", default="../configs/ffg_u_cifar100.json", 
                       help="Path to inference config file")
    parser.add_argument("--cifar", type=int, default=100, choices=[10, 100], 
                       help="CIFAR dataset version (10 or 100)")
    parser.add_argument("--resnet", type=int, default=18, choices=[18, 34, 50, 101, 152], 
                       help="ResNet architecture")
    parser.add_argument("--test-samples", type=int, default=8, 
                       help="Number of MC samples for test-time prediction")
    parser.add_argument("--batch-size", type=int, default=200, 
                       help="Batch size for evaluation")
    parser.add_argument("--device", default="cuda:0", 
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    net, device = load_model(args.model_path, args.config_path, cifar=args.cifar, resnet=args.resnet)
    test_loader = get_test_loader(cifar=args.cifar, batch_size=args.batch_size)
    acc, nll, ece_value = evaluate(net, device, test_loader, test_samples=args.test_samples)
    
    print(f"Model: {args.model_path}")
    print(f"Config: {args.config_path}")
    print(f"Dataset: CIFAR-{args.cifar}")
    print(f"Architecture: ResNet-{args.resnet}")
    print(f"Test Samples: {args.test_samples}")
    print(f"Accuracy: {acc:.4f}")
    print(f"NLL: {nll:.4f}")
    print(f"ECE: {ece_value:.4f}")

if __name__ == '__main__':
    main() 
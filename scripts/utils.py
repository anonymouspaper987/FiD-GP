import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision import datasets, transforms

#from resnet_official import resnet18, resnet34, resnet50, resnet101, resnet152
import torchvision
import torch.nn as nn
import random
#from AutoAugment.autoaugment import CIFAR10Policy
#from fast_autoaugment.FastAutoAugment.augmentations import *
# from fast_autoaugment.FastAutoAugment.archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet

# class Augmentation(object):
#     def __init__(self, policies):
#         self.policies = policies

#     def __call__(self, img):
#         for _ in range(1):
#             policy = random.choice(self.policies)
#             for name, pr, level in policy:
#                 if random.random() > pr:
#                     continue
#                 img = apply_augment(img, name, level)
#         return img

def convert_bn_to_syncbn(model):
    """Convert all BatchNorm layers to SyncBatchNorm."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.__class__ = nn.SyncBatchNorm
    return model



def make_network(architecture: str, *args, **kwargs):
    if architecture == "fcn":
        return FCN(**kwargs)
    elif architecture == "cnn":
        return CNN(**kwargs)
    elif architecture.startswith("resnet"):
        net = getattr(torchvision.models, architecture)(num_classes=kwargs["out_features"])
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
            stride = kwargs.get("stride", 1)
            padding = kwargs.get("padding", kernel_size // 2)
            in_channels = kwargs.get("in_channels", 3)
            bias = net.conv1.bias is not None
            net.conv1 = nn.Conv2d(in_channels, net.conv1.out_channels, kernel_size, stride, padding, bias=bias)
        if kwargs.get("remove_maxpool", False):
            net.maxpool = nn.Identity()
        if "bn_layer" in kwargs:
            # Convert BatchNorm to specified type
            for module in net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.__class__ = kwargs["bn_layer"]
        return net
    else:
        raise ValueError("Unrecognized network architecture:", architecture)

def get_dataset(name):
    if name == 'MNIST':
        return datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), \
                datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), 10
    elif name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

        return trainset, testset, 10
    elif name == 'CIFAR100':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #CIFAR10Policy(),
            #Augmentation(autoaug_paper_cifar10()),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            #Cutout(1, 8)
        ])
        cifar100_train = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        cifar100_test = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)

        return cifar100_train, cifar100_test, 100

    elif name == 'ImageNet':
        pass
    else:
        raise Exception("Unkown dataset:", name)


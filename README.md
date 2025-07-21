# Flow-Induced Diagonal Gaussian Processes (FiD-GP)

This repository contains the official implementation of the paper "Flow-Induced Diagonal Gaussian Processes" for uncertainty quantification in deep neural networks.

## Uncertainty Visualization Results

Our method effectively captures epistemic uncertainty in semantic segmentation tasks. The uncertainty maps below demonstrate how FiD-GP identifies regions of high uncertainty, particularly at object boundaries and ambiguous areas.

### Uncertainty Visualization Results

<table>
<tr>
<td align="center" style="width: 40%;">
<img src="gif/camvid_uncertainty_maps.gif" style="width: 350px; height: 250px; object-fit: cover;" alt="CamVid Uncertainty Maps">
<br>
<strong>CamVid Dataset</strong>
<br>
<em>Uncertainty visualization for CamVid test images. Bright white outlines indicate high uncertainty regions, typically occurring at object boundaries and ambiguous areas.</em>
</td>
<td align="center" style="width: 60%;">
<img src="gif/lindau_uncertainty_maps.gif" style="width: 500px; height: 250px; object-fit: cover;" alt="Lindau Uncertainty Maps">
<br>
<strong>Lindau Driving Sequence</strong>
<br>
<em>Uncertainty maps for the Lindau driving sequence. Notice how uncertainty is higher at object boundaries (cars, trees, buildings) and road edges, where the model is less confident about pixel classifications.</em>
</td>
</tr>
</table>


## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA (for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd FiD-GP
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install albumentations opencv-python pillow tqdm matplotlib
```

## Usage

### Training


#### CIFAR-100 Classification Training
```bash
python scripts/cifar_train.py \
    --inference-config configs/ffg_u_cifar100.json \
    --num-epochs 200 \
    --ml-epochs 100 \
    --annealing-epochs 50 \
    --lr 1e-3 \
    --milestones 100 \
    --resnet 18 \
    --cifar 100
```

#### ImageNet Classification Training (4 GPUs)
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
    scripts/imagenet_train.py \
    --dist-url tcp://127.0.0.1:29500 \
    --resnet 18 \
    --batch-size 100 \
    --use-prefetcher \
    --num-workers 8 \
    --multiprocessing-distributed \
    --output-dir output_resnet18_imagenet_4gpu
```

#### CamVid Segmentation Training
```bash
python scripts/camvid_segmentation_train.py \
    --inference-config configs/ffg_u_camvid.json \
    --num-epochs 200 \
    --ml-epochs 100 \
    --annealing-epochs 50 \
    --lr 1e-3 \
    --milestones 100 \
    --backbone resnet18
```

### Prediction and Uncertainty Estimation



#### CIFAR-100 Classification Prediction
#### Download the weights of model: [Google Drive](https://drive.google.com/file/d/1rw0RWz1L0MI9cLX3ZaLFqkQjf-7jNRHu/view?usp=drive_link) 
```bash
python scripts/cifar_prediction.py \
    --model-path weights/cifar100/epoch-135-0.763-0.036.pth \
    --config-path configs/ffg_u_cifar100.json \
    --cifar 100 \
    --resnet 18 \
    --test-samples 8 \
    --batch-size 200
```

#### CamVid Segmentation Prediction

#### Download the weights of model: [Google Drive](https://drive.google.com/file/d/1fllkwkeCoaJj_ITi_n9jR77GOXF8qBpf/view?usp=drive_link) 


```bash
python scripts/camvid_predict.py \
    --model-path weights/camvid/112_model_miou_0.627.pth \
    --data-root data/camvid2 \
    --output-dir prediction_results_camvid \
    --inference-config configs/ffg_u_camvid.json \
    --test-samples 8 \
    --batch-size 8 \
    --backbone fcn_resnet50
```


### Out-of-Distribution Detection

#### CIFAR-100 to CIFAR-10/SVHN
```bash
python scripts/ood_cifar100_2_cifar10_SVHN.py \
    --model-path output_resnet18_cifar100_context_128_no_james/epoch: 135-0.763-0.036 \
    --config-path configs/ffg_u_cifar100.json \
    --data-root data \
    --batch-size 103 \
    --test-samples 20 \
    --score-type inducing
```

#### CamVid to Cityscapes
```bash
python scripts/ood_camvid_2_cityspaces.py \
    --model-path output_camvid_segmentation/112_model_miou_0.623.pth \
    --config-path configs/ffg_u_camvid.json \
    --data-root data/camvid2 \
    --cityscapes-root data/cityspaces \
    --batch-size 4 \
    --test-samples 20 \
    --score-type inducing
```

## Configuration

The model behavior is controlled by JSON configuration files in the `configs/` directory:

- `ffg_u_camvid.json`: CamVid segmentation configuration
- `ffg_u_cifar100.json`: CIFAR-100 classification configuration
- `ffg_u_imagenet.json`: ImageNet classification configuration

Key configuration parameters:
- `inducing_rows/cols`: Number of inducing variables
- `whitened_u`: Whether to use whitened inducing variables
- `q_inducing`: Inducing variable distribution type
- `learn_lamda`: Whether to learn the lambda parameter
- `prior_sd`: Prior standard deviation

## Results

The uncertainty visualization results demonstrate the effectiveness of FiD-GP in capturing epistemic uncertainty. As shown in the GIFs above, our method successfully identifies regions of high uncertainty, particularly at object boundaries where semantic segmentation is most challenging.

### Model Architecture

The implementation includes:
- **BNN Framework**: Bayesian neural network components in `bnn/`
- **Variational Inference**: Various variational inference methods in `bnn/nn/mixins/variational/`
- **Calibration**: Uncertainty calibration utilities in `bnn/calibration.py`



# AI-Based Image Restoration for Semiconductor Inspection

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

Our solution leverages a Wavelet-guided NAFNet (NAFNetDWT) optimized with a multi-loss objective and Exponential Moving Average (EMA) weight tracking, packaged into an inference engine.

---

## Table of Contents
1. [Project Overview & Methodology](#project-overview--methodology)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Inference Guide (Hackathon Evaluation)](#inference-guide-hackathon-evaluation)
5. [Reproducing the Training Process](#reproducing-the-training-process)
6. [Evaluation & Metrics](#evaluation--metrics)

---

## Project Overview & Methodology

### 1. Architecture: NAFNet + Haar DWT
We utilize a Nonlinear Activation Free Network (NAFNet) as our core backbone. NAFNet achieves restoration efficiency by replacing standard activation functions with SimpleGate mechanisms.
To handle spatial downsampling, the NAFNet encoder/decoder incorporates 2-D Discrete Haar Wavelet Transforms (DWT & IDWT). This allows the model to process high-frequency texture details without destroying spatial structures.

### 2. Data Augmentation (On-the-Fly GPU Physics)
To prevent overfitting and promote generalization to unfamiliar image content, we implemented a two-fold augmentation strategy:
* Spatial Augmentation: 128x128 patch extraction with random horizontal/vertical flips and 90-degree rotations.
* Dynamic GPU Noise Injection: During the training loop, synthetic degradations are applied on the GPU, including dynamic speckle noise, thermal/Gaussian noise, contrast shifting, Gaussian blur, and salt-and-pepper noise to model physical sensor degradation mechanisms.

### 3. Training Strategy (Two-Phase & EMA)
We split training into two distinct phases to balance mathematical pixel fidelity with structural and perceptual quality:
* Phase 1 (Warmup & Fidelity): Initial training optimizes Charbonnier Loss, a smooth L1 variant that recovers baseline structure while remaining robust to outliers.
* Phase 2 (Perceptual Refinement): Training transitions to a composite loss function combining Charbonnier loss, FFT frequency loss, MS-SSIM, full-resolution SSIM, and VGG-LPIPS.
* EMA Weight Tracking: Throughout training, an Exponential Moving Average (EMA) of model weights is maintained (decay = 0.999). The final submitted model utilizes these EMA weights to smooth parameter variance and improve testing generalization.

---

## Repository Structure

```text
KLA/
├── README.md                  # Complete documentation
├── requirements.txt           # Environment dependencies specification
├── inference.py               # Standalone, high-throughput evaluation script
├── train.py                   # Multi-loss EMA training pipeline
├── model.py                   # NAFNetDWT PyTorch architecture
├── dataset.py                 # Dataloaders and patch extractors
├── slice.py                   # Dataset validation split generator
├── npy_to_png.py              # Tool to convert output .npy arrays to .png images
├── weights/                   
│   └── kla_model_final.pth    # Final submitted EMA checkpoint
├── logs/                      
│   ├── Phase1_Log_LPIPS_0.1.csv   # Training loss and metric logs for Phase 1
│   └── Phase2_Log_LPIPS_0.02.csv  # Training loss and metric logs for Phase 2
└── results/                   
    ├── calculate_metrics.py   # PSNR & SSIM evaluation tool
    ├── evaluate_metrics.py    # LPIPS & MS-SSIM evaluation tool
    ├── visualize_test.py      # Output visualization generator (test set)
    └── visualization_metrics.json # Pre-calculated evaluation metrics
```

---

## Environment Setup

The environment relies on standard, cross-platform PyTorch libraries without hardware-locked platform dependencies.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 2. Install required dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

> **A Quick Note on File Paths:**
> The command-line examples provided in the sections below use default folder names (e.g., `test_degraded`, `train_gt`, `test_restored`). Please ensure you adjust the input file names and directory paths in these commands to match your actual local file structure or specific use case when running the scripts.

---

## Inference Guide (Hackathon Evaluation)

The inference.py script runs completely on its own to meet all evaluation requirements. It is built for maximum speed by grouping same-sized images into batches, saving finished files to disk in the background without stalling the GPU and tuning settings upfront for peak hardware performance.

Run inference on a folder of degraded `.npy` arrays using:

```bash
python inference.py \
  --input_dir test_degraded \
  --output_dir test_restored \
  --checkpoint weights/kla_model_final.pth \
  --device auto \
  --batch_size 4
```

* `--device auto`: Automatically detects CUDA (NVIDIA), MPS (Apple Silicon), or CPU.
* `--batch_size 4`: Shape-bucketing allows batched inference across varying input dimensions.

---

## Reproducing the Training Process

To replicate the training pipeline and generate the final `.pth` weights from scratch, use `train.py`.

The script evaluates validation data at the end of every epoch and saves checkpoints to a `checkpoints/` subfolder inside `--output_dir`:

1. `kla_model.pth`: The checkpoint containing weights that achieved the highest validation PSNR.
2. `last_model.pth`: The checkpoint from the final completed epoch.

### Step 0: Dataset Preparation (Validation Split)
Before starting the training process, you must generate the validation dataset. Use the provided `slice.py` script to automatically extract a subset of the training images and move them into dedicated validation folders.

```bash
python slice.py \
  --train_degraded train_degraded \
  --train_gt train_gt \
  --val_degraded val_degraded \
  --val_gt val_gt
```

### Phase 1: PSNR Priority Training (SSIM & LPIPS Weights = 0)
"This phase maximizes baseline PSNR by optimizing primarily for Charbonnier Loss, with a light FFT frequency term. The perceptual and structural losses (SSIM and LPIPS) are explicitly set to 0."

```bash
python train.py \
  --train_degraded train_degraded \
  --train_gt train_gt \
  --val_degraded val_degraded \
  --val_gt val_gt \
  --output_dir phase1 \
  --base_dim 48 \
  --epochs 40 \
  --lr_epochs 40 \
  --lr 2e-4 \
  --batch_size 32 \
  --num_workers 4 \
  --ssim_weight 0.0 \
  --ssim_full_weight 0.0 \
  --fft_weight 0.05 \
  --lpips_weight 0.0 \
  --device auto \
  --no-amp
```

### Phase 2: Perceptual Refinement (EMA & Multi-Loss)
This phase loads the best weights from Phase 1, lowers the learning rate, enables Exponential Moving Average (EMA) tracking, and activates the full suite of perceptual and structural losses (FFT, MS-SSIM, SSIM, and LPIPS) to generate the final submission checkpoint.

```bash
python train.py \
  --train_degraded train_degraded \
  --train_gt train_gt \
  --val_degraded val_degraded \
  --val_gt val_gt \
  --output_dir phase2 \
  --resume phase1/checkpoints/kla_model.pth \
  --base_dim 48 \
  --epochs 30 \
  --batch_size 16 \
  --num_workers 4 \
  --lr 2e-5 \
  --ssim_weight 0.2 \
  --ssim_full_weight 1.0 \
  --fft_weight 0.05 \
  --lpips_weight 0.02 \
  --device auto \
  --no-amp
```

(Note: For our final submission, we utilized the checkpoint from the very last training epoch of Phase 2 to ensure maximum EMA smoothing. We took `last_model.pth` generated inside `phase2/checkpoints/` and renamed it to `kla_model_final.pth`.)

```bash
mkdir -p weights
cp phase2/checkpoints/last_model.pth weights/kla_model_final.pth
```
### Converting Outputs to PNG
The evaluation protocol requires outputting `.npy` arrays. To easily view these restored arrays as standard images, use the provided conversion script:

```bash
python npy_to_png.py \
  --input_dir test_restored \
  --output_dir test_restored_pngs
```

---

## Evaluation & Metrics

Standalone evaluation scripts are provided to compute quantitative metrics on validation pairs:

Calculate PSNR & SSIM:

```bash
python results/calculate_metrics.py --checkpoint weights/kla_model_final.pth --device auto
```

Calculate Perceptual Metrics (MS-SSIM & VGG-LPIPS):

```bash
python results/evaluate_metrics.py --checkpoint weights/kla_model_final.pth --device auto
```

Generate Side-by-Side Visualizations (Test Set):

```bash
python results/visualize_test.py --num_images 10
```




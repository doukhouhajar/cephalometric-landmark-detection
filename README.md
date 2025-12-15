# Automatic Cephalometric Landmark Detection

A deep learning framework for automatic cephalometric landmark detection using heatmap regression with a modified U-Net architecture.

## Overview

This project develops an automatic cephalometric landmark detection framework based on a modified U-Net architecture that performs **heatmap regression** instead of direct coordinate prediction. Given a lateral cephalometric radiograph, the network outputs a set of Gaussian-like probability maps, one per landmark, encoding its most likely location.

### Key Features

- **Modified U-Net with Transformer Encoder**: Swin Transformer encoder captures global craniofacial context
- **Attention Mechanisms**: 
  - CBAM (Channel and Spatial Attention) at bottleneck
  - Attention gates in decoder for landmark-relevant region focusing
- **Multi-Resolution Deep Supervision**: Stable training through supervision at multiple decoder levels
- **Heatmap Regression**: Robust to device variations and image quality differences
- **Comprehensive Evaluation**: Per-landmark metrics and multiple threshold analysis
- **Interactive Demo**: Gradio interface for real-time inference

## Architecture

### LA-UNet with Swin Transformer

```
Input Image (1-channel grayscale → 3-channel)
    ↓
Swin Transformer Encoder (Swin-Tiny)
    ├─ Stage 1: 1/4 resolution
    ├─ Stage 2: 1/8 resolution
    ├─ Stage 3: 1/16 resolution
    └─ Stage 4: 1/32 resolution
    ↓
Channel Projection + Bottleneck (CBAM)
    ↓
Decoder with Attention Gates
    ├─ Multi-resolution outputs (deep supervision)
    └─ Final heatmap prediction
```

### Model Components

1. **Swin Transformer Encoder**: Captures long-range dependencies and global context
2. **CBAM Attention**: Enhances feature representation at bottleneck
3. **Attention Gates**: Focus decoder on relevant spatial regions
4. **Multi-Resolution Supervision**: Supervises intermediate decoder outputs

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the main model (LA-UNet with Swin Transformer):

```bash
python main.py
```

Train baseline U-Net for comparison:

```bash
python train_baseline.py
```

Training configuration can be modified in `configs/training_config.yaml`:

```yaml
training:
  epochs: 80
  batch_size: 2
  learning_rate: 0.0003
  loss:
    use_deep_supervision: true
    deep_supervision_weights: [1.0, 0.5, 0.25, 0.125]
```

### Evaluation and Analysis

Generate comprehensive analysis plots and metrics:

```bash
python analyze_results.py \
    --checkpoint outputs/checkpoints/la_unet_swin_cbam.pth \
    --baseline-checkpoint outputs/checkpoints/baseline_unet.pth \
    --output-dir outputs/analysis
```

This generates:
- Error distribution plots
- Per-landmark MRE and SDR plots
- Model comparison charts
- Training curves
- Sample visualizations

### Interactive Demo

Launch the Gradio interface:

```bash
python gradio_app.py
```

The interface will be available at `http://localhost:7860` with an option to share via public link.

## Evaluation Metrics

- **Mean Radial Error (MRE)**: Average distance between predicted and ground truth landmarks
- **Success Detection Rate (SDR)**: Percentage of landmarks detected within thresholds:
  - SDR @ 2mm: Clinical accuracy threshold
  - SDR @ 2.5mm, 3mm, 4mm: Varying tolerance levels
- **Per-Landmark Analysis**: Individual landmark performance metrics

## Project Structure

```
cephalometric-landmark-detection/
├── configs/
│   ├── model_config.yaml      # Model architecture config
│   └── training_config.yaml   # Training hyperparameters
├── datasets/
│   └── raw/                   # Dataset (CSV + images)
├── outputs/
│   ├── checkpoints/           # Trained model weights
│   ├── logs/                  # Training logs and TensorBoard
│   ├── predictions/           # Prediction outputs
│   └── analysis/              # Analysis plots and metrics
├── src/
│   ├── data/
│   │   ├── data_pipeline.py   # Data loading and augmentation
│   │   └── heatmap_generator.py
│   ├── models/
│   │   ├── la_unet.py         # Main model
│   │   ├── baseline_unet.py   # Baseline for comparison
│   │   ├── swin_encoder.py
│   │   ├── cbam.py
│   │   └── attention_gate.py
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   ├── losses.py          # Loss functions
│   │   └── metrics.py         # Evaluation metrics
│   ├── inference/
│   │   └── inference.py       # Inference utilities
│   └── visualization/
│       └── visualizer.py      # Plotting utilities
├── main.py                    # Main training script
├── train_baseline.py          # Baseline training
├── analyze_results.py         # Analysis script
├── gradio_app.py              # Interactive demo
└── requirements.txt
```

## Key Scientific Contributions

1. **Multi-Resolution Deep Supervision**: Improves training stability and convergence
2. **Transformer-Enhanced U-Net**: Better global context understanding for craniofacial anatomy
3. **Hybrid Attention**: CBAM + Attention Gates for improved feature representation
4. **Comprehensive Evaluation Framework**: Per-landmark analysis and multiple metric tracking

## Results and Visualizations

The analysis script generates publication-ready plots:

1. **Training Curves**: Loss, MRE, and SDR evolution
2. **Error Distributions**: Statistical analysis of prediction errors
3. **Per-Landmark Performance**: Individual landmark difficulty analysis
4. **Model Comparison**: Baseline vs. proposed architecture
5. **Heatmap Visualizations**: Model confidence maps for each landmark
6. **Sample Overlays**: Qualitative results on test images

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cephalometric2024,
  title={Automatic Cephalometric Landmark Detection using Heatmap Regression with Transformer-Enhanced U-Net},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

[Your License]

## Acknowledgments

- Swin Transformer implementation from `timm`
- U-Net architecture inspiration from original paper
- CBAM attention mechanism


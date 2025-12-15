# Usage Guide

## Quick Start

### 1. Training

Train the main model:
```bash
python main.py
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir outputs/logs
```

### 2. Evaluation

After training, generate comprehensive analysis:
```bash
python analyze_results.py \
    --checkpoint outputs/checkpoints/la_unet_swin_cbam.pth \
    --output-dir outputs/analysis
```

Compare with baseline:
```bash
# First train baseline
python train_baseline.py

# Then compare
python analyze_results.py \
    --checkpoint outputs/checkpoints/la_unet_swin_cbam.pth \
    --baseline-checkpoint outputs/checkpoints/baseline_unet.pth \
    --output-dir outputs/analysis
```

### 3. Interactive Demo

Launch Gradio interface:
```bash
python gradio_app.py
```

Access at `http://localhost:7860` or use the shareable link.

## Output Files

### Training Outputs
- `outputs/checkpoints/`: Model weights
- `outputs/logs/training_history.json`: Training metrics
- `outputs/logs/`: TensorBoard logs

### Analysis Outputs
- `outputs/analysis/metrics.json`: Quantitative metrics
- `outputs/analysis/training_curves.png`: Training progression
- `outputs/analysis/error_distribution.png`: Error statistics
- `outputs/analysis/per_landmark_metrics.png`: Per-landmark performance
- `outputs/analysis/model_comparison.png`: Baseline comparison
- `outputs/analysis/sample_*.png`: Visual examples

## Customization

### Adding Landmark Names

Edit `gradio_app.py` or `analyze_results.py` to customize landmark names:
```python
LANDMARK_NAMES = [
    "Nasion", "Sella", "A-point", "B-point", ...
]
```

### Adjusting Deep Supervision Weights

Edit `configs/training_config.yaml`:
```yaml
loss:
  deep_supervision_weights: [1.0, 0.5, 0.25, 0.125]  # Main, aux2, aux3, aux4
```

### Changing Evaluation Thresholds

Modify thresholds in `analyze_results.py` or metrics computation.

## Tips for Report Generation

1. **Use high-resolution plots**: DPI=300 for publication
2. **Include error bars**: Statistical significance tests
3. **Per-landmark analysis**: Highlight difficult landmarks
4. **Ablation studies**: Compare with/without attention, transformer, etc.
5. **Cross-dataset evaluation**: Test on multiple datasets

## Troubleshooting

### Out of Memory
- Reduce batch size in config
- Use gradient accumulation
- Reduce image size

### Poor Performance
- Check data normalization
- Verify heatmap sigma value
- Adjust learning rate schedule
- Increase training epochs

### Missing Dependencies
```bash
pip install -r requirements.txt
```


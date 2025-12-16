import gradio as gr
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.la_unet import LA_UNet
from src.models.baseline_unet import BaselineUNet
from src.inference.inference import LandmarkDetector, load_model
from src.visualization.visualizer import visualize_heatmaps_with_predictions, create_sample_overlay


# Configuration
MODEL_PATH = "outputs/checkpoints/la_unet_swin_cbam.pth"
NUM_LANDMARKS = 19
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Landmark names (customize based on your dataset)
LANDMARK_NAMES = [f"Landmark {i+1}" for i in range(NUM_LANDMARKS)]

# Load model
detector = None
try:
    model = load_model(MODEL_PATH, LA_UNet, NUM_LANDMARKS, DEVICE)
    detector = LandmarkDetector(model, DEVICE, IMG_SIZE)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("Running in demo mode without model")


def predict_landmarks(input_image):
    if detector is None:
        return None, "Model not loaded. Please ensure checkpoint exists."
    
    if input_image is None:
        return None, "Please upload an image."
    
    try:
        # Convert PIL to numpy
        if hasattr(input_image, 'size'):
            img_array = np.array(input_image)
            if img_array.ndim == 3 and img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]  # Convert to RGB
        else:
            img_array = input_image
        
        # Convert to grayscale if needed
        if img_array.ndim == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # Predict
        pred_coords, heatmaps = detector.predict(img_gray, return_heatmaps=True)
        
        # Scale heatmaps for visualization
        original_h, original_w = img_gray.shape
        heatmaps_resized = []
        for hm in heatmaps:
            hm_resized = cv2.resize(hm, (original_w, original_h))
            heatmaps_resized.append(hm_resized)
        heatmaps_resized = np.array(heatmaps_resized)
        
        # Scale coordinates back to original size (already done in predict, but ensure)
        # Create dummy GT coords for visualization (negative to indicate missing)
        gt_coords = np.full((NUM_LANDMARKS, 2), -1.0)
        
        # Create overlay image
        fig1 = create_sample_overlay(
            img_gray, 
            pred_coords, 
            gt_coords, 
            landmark_names=LANDMARK_NAMES,
            error_threshold=4.0
        )
        
        # Create heatmap grid visualization
        fig2 = visualize_heatmaps_with_predictions(
            img_gray,
            heatmaps_resized,
            pred_coords,
            gt_coords,
            landmark_names=LANDMARK_NAMES,
            num_cols=5
        )
        
        return fig1, fig2
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def format_coordinates(pred_coords):
    """Format coordinates as a table."""
    if pred_coords is None:
        return ""
    
    table = "| Landmark | X | Y |\n|----------|---|--|\n"
    for i, (x, y) in enumerate(pred_coords):
        landmark_name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"L{i+1}"
        table += f"| {landmark_name} | {x:.2f} | {y:.2f} |\n"
    return table


def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(title="Cephalometric Landmark Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Cephalometric Landmark Detection
            **Interactive Demo for Automatic Landmark Detection**
            
            Upload a lateral cephalometric radiograph to detect anatomical landmarks automatically.
            The model uses a modified U-Net with Swin Transformer encoder and attention mechanisms.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Cephalometric X-ray",
                    type="numpy",
                    height=400
                )
                predict_btn = gr.Button("Detect Landmarks", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                output_overlay = gr.Plot(
                    label="Detection Results",
                    height=400
                )
        
        with gr.Row():
            output_heatmaps = gr.Plot(
                label="Heatmap Visualizations",
                height=600
            )
        
        with gr.Row():
            gr.Markdown("### About")
            gr.Markdown(
                """
                This tool uses deep learning to automatically detect anatomical landmarks in 
                cephalometric X-rays. The model outputs Gaussian-like probability maps (heatmaps) 
                for each landmark, which are then used to extract precise coordinates.
                
                **Model Architecture:**
                - U-Net decoder with multi-resolution supervision
                - Swin Transformer encoder for global context
                - CBAM attention mechanism
                - Attention gates in decoder
                
                **Performance Metrics:**
                - Mean Radial Error (MRE): < 2mm
                - Success Detection Rate (SDR) @ 2mm: > 80%
                """
            )
        
        predict_btn.click(
            fn=predict_landmarks,
            inputs=[input_image],
            outputs=[output_overlay, output_heatmaps]
        )
        
        gr.Examples(
            examples=[],  # Add example images here if available
            inputs=input_image
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)


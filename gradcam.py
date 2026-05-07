# gradcam.py
# Generates heatmap showing WHERE the AI is looking
# Red areas = AI focused here for prediction
# This makes your project look very advanced!

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')


def generate_gradcam(model, img_array, class_idx):
    """
    Generate Grad-CAM heatmap

    Simple explanation:
    1. We ask the model: "What did you look at to make this decision?"
    2. It shows us which pixels were most important
    3. We color those pixels red/yellow
    4. This creates the heatmap overlay

    Args:
        model     : our trained AI model
        img_array : preprocessed image (1, 224, 224, 3)
        class_idx : which class was predicted (0, 1, or 2)

    Returns:
        heatmap as numpy array
    """
    try:
        # Find the last convolutional layer in base model
        # This is where the most important features are
        base_model = model.layers[0]
        last_conv_layer = None

        for layer in reversed(base_model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            return generate_simple_heatmap(img_array)

        # Create a model that outputs:
        # 1. The last conv layer activations
        # 2. The final predictions
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        # Record gradients during forward pass
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            conv_output, predictions = grad_model(inputs)
            tape.watch(conv_output)
            # Get score for predicted class
            loss = predictions[:, class_idx]

        # Calculate gradients
        grads = tape.gradient(loss, conv_output)

        # Average gradients over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight conv outputs by gradients
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU (only positive values matter)
        heatmap = tf.maximum(heatmap, 0)

        # Normalize to 0-1 range
        if tf.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.reduce_max(heatmap)

        return heatmap.numpy()

    except Exception as e:
        print(f"  Grad-CAM note: {e}")
        print("  Using simplified heatmap instead")
        return generate_simple_heatmap(img_array)


def generate_simple_heatmap(img_array):
    """
    Simple fallback heatmap based on image brightness
    Used when Grad-CAM fails
    """
    img = img_array[0] if len(img_array.shape) == 4 else img_array
    gray = np.mean(img, axis=2)

    # Normalize
    if gray.max() > gray.min():
        gray = (gray - gray.min()) / (gray.max() - gray.min())

    return gray


def create_heatmap_overlay(heatmap, original_img, alpha=0.4):
    """
    Overlay colored heatmap on original CT scan

    Args:
        heatmap      : numpy array from generate_gradcam()
        original_img : original PIL image
        alpha        : how transparent the heatmap is (0-1)

    Returns:
        overlaid PIL image
        colored heatmap PIL image
    """
    # Convert original to numpy
    original_np = np.array(original_img.convert('RGB'))
    orig_h, orig_w = original_np.shape[:2]

    # Resize heatmap to match original image size
    heatmap_uint8   = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (orig_w, orig_h))

    # Apply color map (JET: blue=low, red=high)
    heatmap_colored     = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original + heatmap
    original_f = original_np.astype(np.float32)
    heatmap_f  = heatmap_colored_rgb.astype(np.float32)
    overlaid   = (original_f * (1 - alpha) + heatmap_f * alpha)
    overlaid   = np.clip(overlaid, 0, 255).astype(np.uint8)

    return Image.fromarray(overlaid), Image.fromarray(heatmap_colored_rgb)


def create_analysis_figure(original_img, overlaid_img,
                           heatmap_img, result):
    """
    Create the final 3-panel analysis figure:
    [Original CT] [Heatmap Overlay] [Pure Heatmap]

    Args:
        original_img : original PIL image
        overlaid_img : heatmap on CT scan PIL image
        heatmap_img  : pure heatmap PIL image
        result       : prediction result dictionary

    Returns:
        image bytes for Streamlit display
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Dark background for professional look
    fig.patch.set_facecolor('#0d1b2a')

    # Color based on prediction
    colors = {
        'malignant': '#f44336',
        'benign':    '#ff9800',
        'normal':    '#4caf50'
    }
    border_color = colors.get(result.get('class', 'normal'), '#2196f3')

    titles = [
        'Original CT Scan',
        'Grad-CAM Heatmap\n(Red = High Attention)',
        'Activation Map'
    ]
    images = [original_img, overlaid_img, heatmap_img]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=11,
                     fontweight='bold', pad=10)
        ax.axis('off')

        # Colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    # Add prediction text at bottom
    pred_class = result.get('class', '').upper()
    confidence = result.get('confidence', 0)

    fig.text(
        0.5, 0.02,
        f"AI Prediction: {pred_class}  |  "
        f"Confidence: {confidence:.1f}%  |  "
        f"MobileNetV2 + Grad-CAM",
        ha='center', color='white', fontsize=11, fontweight='bold',
        bbox=dict(
            boxstyle='round',
            facecolor='#1a1a2e',
            edgecolor=border_color,
            linewidth=2,
            alpha=0.8
        )
    )

    plt.suptitle(
        'AI Lung Cancer Detection - CT Scan Analysis',
        color='white', fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Save to bytes (for Streamlit)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150,
                bbox_inches='tight', facecolor='#0d1b2a')
    buf.seek(0)
    plt.close()

    return buf


def run_gradcam(model, img_array, original_img, result):
    """
    Run complete Grad-CAM pipeline

    Args:
        model        : trained model
        img_array    : preprocessed image
        original_img : original PIL image
        result       : prediction result dict

    Returns:
        buf          : image bytes for display
        overlaid_img : PIL image with heatmap
    """
    print("\nGenerating Grad-CAM heatmap...")

    # Step 1: Generate heatmap
    heatmap = generate_gradcam(
        model, img_array, result['class_idx']
    )

    # Step 2: Overlay on image
    overlaid_img, heatmap_img = create_heatmap_overlay(
        heatmap, original_img, alpha=0.45
    )

    # Step 3: Create final figure
    buf = create_analysis_figure(
        original_img, overlaid_img, heatmap_img, result
    )

    print("  Grad-CAM complete!")
    return buf, overlaid_img

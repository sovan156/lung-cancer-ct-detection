# preprocessing.py
# Complete version with all required functions

import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Settings
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 16
NUM_CLASSES = 2


def create_data_generators(train_dir, test_dir):
    """
    Create image data generators for training
    """
    print("\nSetting up image data generators...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print(f"  Training images   : {train_generator.samples}")
    print(f"  Validation images : {val_generator.samples}")
    print(f"  Test images       : {test_generator.samples}")
    print(f"  Classes detected  : {train_generator.class_indices}")
    print(f"  Number of classes : {len(train_generator.class_indices)}")

    return train_generator, val_generator, test_generator


def preprocess_single_image(image_file):
    """
    Prepare ONE uploaded image for prediction

    This function is called by app.py when user uploads CT scan

    Args:
        image_file: uploaded file from Streamlit

    Returns:
        img_batch    : processed image ready for model (1,224,224,3)
        img_original : original PIL image for display
    """
    # Open the image file
    img = Image.open(image_file)

    # Keep a copy of original for display purposes
    img_original = img.copy()

    # Convert to RGB
    # (handles grayscale, RGBA, and other formats)
    img_rgb = img.convert('RGB')

    # Resize to 224x224 pixels
    # This is what MobileNetV2 expects
    img_resized = img_rgb.resize(IMAGE_SIZE)

    # Convert PIL image to numpy array
    img_array = np.array(img_resized, dtype=np.float32)

    # Normalize pixel values from 0-255 to 0.0-1.0
    img_normalized = img_array / 255.0

    # Add batch dimension
    # Model expects (batch_size, height, width, channels)
    # We only have 1 image so: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch, img_original

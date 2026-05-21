# predict.py
# Fixed version - correct prediction display

import numpy as np
import tensorflow as tf
import joblib
from preprocessing import preprocess_single_image

# Info for each class
CLASS_INFO = {
    'normal': {
        'display':     '✅ NO CANCER DETECTED',
        'color':       '#4caf50',
        'risk':        'LOW RISK',
        'action':      'Continue regular annual CT screenings',
        'icon':        '✅',
        'description': (
            "No signs of lung cancer detected in this CT scan. "
            "The lung tissue appears normal with no suspicious "
            "masses or nodules. Regular annual CT screenings are "
            "still recommended, especially for high-risk individuals "
            "such as smokers or those above age 50 with family history."
        )
    },
    'cancer': {
        'display':     '🔴 LUNG CANCER DETECTED',
        'color':       '#f44336',
        'risk':        'HIGH RISK',
        'action':      'URGENT: Consult oncologist immediately',
        'icon':        '🔴',
        'description': (
            "Signs of lung cancer detected in this CT scan. "
            "Cancerous tissue has been identified by the AI model. "
            "IMMEDIATE consultation with an oncologist is strongly "
            "recommended. Early detection and treatment significantly "
            "improves survival rates. Additional tests such as "
            "PET scan and biopsy may be required for confirmation."
        )
    }
}


def load_model(model_path='model/best_model.h5'):
    """Load the trained model from disk"""
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_model_info(info_path='model/model_info.pkl'):
    """Load model information"""
    try:
        info = joblib.load(info_path)
        return info
    except Exception as e:
        print(f"Error loading model info: {e}")
        return None


def predict_ct_scan(model, model_info, image_file):
    """
    Predict cancer from CT scan image

    FIXED VERSION:
    - Correctly maps index to class name
    - Confidence matches the predicted class
    - No mismatch between result and charts
    """

    # Step 1: Preprocess image
    img_array, original_img = preprocess_single_image(image_file)

    # Step 2: Run model
    # Returns array like [0.89, 0.10] for 2 classes
    predictions   = model.predict(img_array, verbose=0)
    probabilities = predictions[0]

    # Step 3: Print for debugging
    print(f"\nRaw predictions: {probabilities}")
    print(f"Class indices: {model_info['class_indices']}")

    # Step 4: Get class mapping
    # class_indices looks like: {'cancer': 0, 'normal': 1}
    class_indices = model_info['class_indices']

    # Step 5: Build probability dict with correct names
    # {'cancer': 0.89, 'normal': 0.10}
    class_probabilities = {}
    for class_name, class_idx in class_indices.items():
        class_probabilities[class_name] = float(probabilities[class_idx]) * 100

    print(f"Class probabilities: {class_probabilities}")

    # Step 6: Find which class has HIGHEST probability
    # This is the predicted class
    predicted_class = max(class_probabilities, key=class_probabilities.get)
    confidence      = class_probabilities[predicted_class]

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.1f}%")

    # Step 7: Get class index for Grad-CAM
    predicted_idx = class_indices[predicted_class]

    # Step 8: Get display info
    info = CLASS_INFO.get(predicted_class, CLASS_INFO['normal'])

    # Step 9: Build result
    result = {
        'predicted_class': predicted_class,
        'class':           predicted_class,
        'class_idx':       predicted_idx,
        'confidence':      confidence,
        'probabilities':   class_probabilities,
        'display_name':    info['display'],
        'color':           info['color'],
        'risk':            info['risk'],
        'action':          info['action'],
        'icon':            info['icon'],
        'description':     info['description'],
        'is_cancer':       predicted_class == 'cancer',
        'needs_attention': predicted_class == 'cancer'
    }

    return result, img_array, original_img

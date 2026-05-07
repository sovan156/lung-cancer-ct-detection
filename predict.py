# predict.py
# Makes predictions on uploaded CT scan images

import numpy as np
import tensorflow as tf
import joblib
from preprocessing import preprocess_single_image

# What each class means to doctors
CLASS_INFO = {
    'normal': {
        'display': '✅ NORMAL',
        'color':   '#4caf50',
        'risk':    'LOW RISK',
        'action':  'Continue regular annual screenings',
        'icon':    '✅',
        'description': (
            "No signs of lung cancer detected in this CT scan. "
            "The lung tissue appears normal with no suspicious masses "
            "or nodules. Regular annual CT screenings are still "
            "recommended especially for high-risk individuals "
            "(smokers, age above 50, family history)."
        )
    },
    'benign': {
        'display': '⚠️ BENIGN TUMOR',
        'color':   '#ff9800',
        'risk':    'MEDIUM RISK',
        'action':  'Schedule follow-up with doctor within 30 days',
        'icon':    '⚠️',
        'description': (
            "A benign (non-cancerous) tumor or nodule has been "
            "detected in this CT scan. Benign tumors do NOT spread "
            "to other parts of the body. However, medical consultation "
            "is strongly recommended for proper monitoring. "
            "Follow-up CT scans may be advised every 3-6 months."
        )
    },
    'malignant': {
        'display': '🔴 MALIGNANT CANCER',
        'color':   '#f44336',
        'risk':    'HIGH RISK',
        'action':  'URGENT: Consult oncologist immediately',
        'icon':    '🔴',
        'description': (
            "Signs of malignant (cancerous) tissue detected in this "
            "CT scan. Malignant tumors can spread to other parts of "
            "the body if not treated. IMMEDIATE consultation with an "
            "oncologist is strongly recommended. Early detection "
            "significantly improves treatment outcomes and survival rates."
        )
    }
}


def load_model(model_path='model/best_model.h5'):
    """Load the trained model from disk"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_model_info(info_path='model/model_info.pkl'):
    """Load model information from disk"""
    try:
        info = joblib.load(info_path)
        return info
    except Exception as e:
        print(f"Error loading model info: {e}")
        return None


def predict_ct_scan(model, model_info, image_file):
    """
    Predict cancer from CT scan image

    Steps:
    1. Preprocess the image
    2. Run through AI model
    3. Get probabilities for each class
    4. Return detailed result

    Args:
        model      : loaded Keras model
        model_info : model metadata
        image_file : uploaded file from Streamlit

    Returns:
        result       : dict with all prediction details
        img_array    : processed image (for Grad-CAM)
        original_img : original PIL image (for display)
    """
    # Step 1: Preprocess
    img_array, original_img = preprocess_single_image(image_file)

    # Step 2: Predict
    # Returns array like [0.1, 0.2, 0.7] for 3 classes
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]  # Remove batch dimension

    # Step 3: Find which class has highest probability
    predicted_idx = int(np.argmax(probabilities))

    # Step 4: Convert index to class name
    # class_indices looks like: {'benign': 0, 'malignant': 1, 'normal': 2}
    idx_to_class = {
        v: k for k, v in model_info['class_indices'].items()
    }
    predicted_class = idx_to_class[predicted_idx]

    # Confidence = probability of predicted class * 100
    confidence = float(probabilities[predicted_idx]) * 100

    # All class probabilities as percentages
    class_probabilities = {}
    for idx, class_name in idx_to_class.items():
        class_probabilities[class_name] = float(probabilities[idx]) * 100

    # Get class info
    info = CLASS_INFO.get(predicted_class, CLASS_INFO['normal'])

    # Build complete result dictionary
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
        'is_cancer':       predicted_class == 'malignant',
        'needs_attention': predicted_class in ['malignant', 'benign']
    }

    return result, img_array, original_img

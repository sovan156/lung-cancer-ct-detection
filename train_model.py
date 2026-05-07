# train_model.py
# Trains deep learning model on CT scan images
# 2 Classes: normal vs cancer

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from preprocessing import (
    create_data_generators,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_CLASSES
)


def build_model(num_classes):
    """
    Build AI model using MobileNetV2 Transfer Learning

    Args:
        num_classes: number of output classes (2)
    """
    print(f"\nBuilding AI model for {num_classes} classes...")

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze pretrained layers
    base_model.trainable = False
    print(f"  Base model layers: {len(base_model.layers)} (frozen)")

    # Build model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        # Output layer - num_classes neurons (2 for us)
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"  Output classes: {num_classes}")
    print("  Model built successfully!")
    model.summary()

    return model


def get_callbacks():
    """Training callbacks"""
    os.makedirs('model', exist_ok=True)

    return [
        callbacks.ModelCheckpoint(
            filepath='model/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]


def plot_training_history(history):
    """Save accuracy and loss charts"""
    print("\nCreating charts...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Training History', fontsize=14, fontweight='bold')

    epochs = range(len(history.history['accuracy']))

    # Accuracy
    axes[0].plot(epochs, history.history['accuracy'],
                 label='Train', color='#2196F3', linewidth=2)
    axes[0].plot(epochs, history.history['val_accuracy'],
                 label='Validation', color='#4CAF50', linewidth=2)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(epochs, history.history['loss'],
                 label='Train', color='#F44336', linewidth=2)
    axes[1].plot(epochs, history.history['val_loss'],
                 label='Validation', color='#FF9800', linewidth=2)
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs('model', exist_ok=True)
    plt.savefig('model/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: model/training_history.png")


def plot_confusion_matrix_chart(y_true, y_pred, class_names):
    """Save confusion matrix chart"""
    print("Creating confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        annot_kws={"size": 16, "weight": "bold"}
    )
    plt.title('Confusion Matrix', fontsize=13, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: model/confusion_matrix.png")


def main():
    print("\n" + "="*65)
    print("  AI LUNG CANCER CT SCAN - MODEL TRAINING")
    print("="*65)

    # ── Check GPU ──────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU found! Training will be fast!")
    else:
        print("\nNo GPU - Using CPU")
        print("Training will take 15-30 minutes. Be patient!")

    # ── Check dataset ──────────────────────────────────────
    train_dir = 'dataset/train'
    test_dir  = 'dataset/test'

    if not os.path.exists(train_dir):
        print(f"\nERROR: {train_dir} not found!")
        print("Run setup_dataset.py first!")
        return

    # Count images
    print("\nImage counts:")
    for split in ['train', 'test']:
        for cls in ['normal', 'cancer']:
            path = f'dataset/{split}/{cls}'
            if os.path.exists(path):
                imgs = [f for f in os.listdir(path)
                        if f.lower().endswith(('.jpg','.jpeg','.png'))]
                print(f"  {split}/{cls}: {len(imgs)} images")
            else:
                print(f"  {split}/{cls}: NOT FOUND ❌")

    # ── Create data generators ─────────────────────────────
    print("\nLoading images...")
    train_gen, val_gen, test_gen = create_data_generators(
        train_dir, test_dir
    )

    # Get actual number of classes from data
    actual_classes = len(train_gen.class_indices)
    print(f"\nActual classes found: {actual_classes}")
    print(f"Class mapping: {train_gen.class_indices}")

    # ── Build model with ACTUAL number of classes ──────────
    # This is the KEY FIX - use actual_classes not NUM_CLASSES
    model = build_model(num_classes=actual_classes)

    # ── Train ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("STARTING TRAINING...")
    print("Watch accuracy improve each epoch!")
    print("="*65 + "\n")

    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1
    )

    # ── Evaluate ───────────────────────────────────────────
    print("\n" + "="*65)
    print("EVALUATING ON TEST DATA...")
    print("="*65)

    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = test_gen.classes
    class_names  = list(test_gen.class_indices.keys())

    accuracy     = accuracy_score(y_true, y_pred)
    report       = classification_report(
                       y_true, y_pred,
                       target_names=class_names
                   )

    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"\nClassification Report:\n{report}")

    # ── Save charts ────────────────────────────────────────
    print("\nSaving charts...")
    plot_training_history(history)
    plot_confusion_matrix_chart(y_true, y_pred, class_names)

    # ── Save model info ────────────────────────────────────
    model_info = {
        'accuracy':              float(accuracy),
        'classification_report': report,
        'class_indices':         train_gen.class_indices,
        'class_names':           class_names,
        'image_size':            224,
        'model_type':            'MobileNetV2 Transfer Learning',
        'num_classes':           actual_classes
    }

    os.makedirs('model', exist_ok=True)
    joblib.dump(model_info, 'model/model_info.pkl')
    print("  Saved: model/model_info.pkl")

    # ── Done ───────────────────────────────────────────────
    print("\n" + "="*65)
    print("TRAINING COMPLETE!")
    print("="*65)
    print(f"\nFinal Accuracy : {accuracy*100:.2f}%")
    print(f"Model saved    : model/best_model.h5")
    print(f"Info saved     : model/model_info.pkl")
    print(f"\nNow run: streamlit run app.py")
    print("="*65)


if __name__ == "__main__":
    main()

"""
Dog vs Cat Image Classifier
Author: Javier Romero
Description: Build a simple neural network to classify dog and cat images
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

print(f"Full dataset shape - Training: {x_train_full.shape}, Test: {x_test_full.shape}")
print(f"Full dataset labels - Training: {y_train_full.shape}, Test: {y_test_full.shape}")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print("\nCIFAR-10 classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Filter to cats (3) and dogs (5)
cat_class = 3
dog_class = 5

# Training set
train_cat_mask = (y_train_full.flatten() == cat_class)
train_dog_mask = (y_train_full.flatten() == dog_class)
train_mask = train_cat_mask | train_dog_mask

x_train = x_train_full[train_mask]
y_train = y_train_full[train_mask]

# Convert labels: cat=0, dog=1
y_train = np.where(y_train == cat_class, 0, 1)

# Test set
test_cat_mask = (y_test_full.flatten() == cat_class)
test_dog_mask = (y_test_full.flatten() == dog_class)
test_mask = test_cat_mask | test_dog_mask

x_test = x_test_full[test_mask]
y_test = y_test_full[test_mask]

# Convert labels: cat=0, dog=1
y_test = np.where(y_test == dog_class, 1, 0)

print(f"\nFiltered dataset:")
print(f"  Training: {x_train.shape[0]} images")
print(f"  Test: {x_test.shape[0]} images")
print(f"  Image shape: {x_train.shape[1:]}")

# Check class distribution
print(f"\nTraining set - Cats: {(y_train == 0).sum()}, Dogs: {(y_train == 1).sum()}")
print(f"Test set - Cats: {(y_test == 0).sum()}, Dogs: {(y_test == 1).sum()}")

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten images for simple neural network (32x32x3 = 3072)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print(f"\nFlattened shape: {x_train_flat.shape}")

# Visualize some sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    row = i // 5
    col = i % 5
    axes[row, col].imshow(x_train[i])
    label = 'Cat' if y_train[i] == 0 else 'Dog'
    axes[row, col].set_title(f'{label}')
    axes[row, col].axis('off')
#plt.tight_layout()
#plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
print("\nSaved sample images to 'sample_images.png'")
#plt.show()

# Build the neural network
print("\n" + "="*50)
print("BUILDING NEURAL NETWORK")
print("="*50)

# Simple feedforward neural network
model = keras.Sequential([
    # Input layer: 3072 features (32*32*3)
    layers.Dense(128, activation='relu', input_shape=(3072,), name='hidden_layer_1'),
    layers.Dense(64, activation='relu', name='hidden_layer_2'),
    # Output layer: 1 neuron for binary classification (dog or cat)
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

# Compile the model
model.compile(
    optimizer='adam',           # Optimization algorithm
    loss='binary_crossentropy', # Loss function for binary classification
    metrics=['accuracy']        # Metric to track during training
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Count parameters
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")

# Train the model
print("\n" + "="*50)
print("TRAINING THE MODEL")
print("="*50)

# Training parameters
EPOCHS = 20
BATCH_SIZE = 32

print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}...")

# Train the model
history = model.fit(
    x_train_flat, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test_flat, y_test),
    verbose=1  # Show progress bar
)

print("\n✓ Training complete!")

# Evaluate the model
print("\n" + "="*50)
print("EVALUATING THE MODEL")
print("="*50)

test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Make predictions on test set
predictions = model.predict(x_test_flat)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Calculate accuracy manually
correct = (predicted_classes == y_test.flatten()).sum()
total = len(y_test)
print(f"\nManual accuracy check: {correct}/{total} ({correct/total*100:.2f}%)")

# Visualize training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot loss
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

#plt.tight_layout()
#plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("\nSaved training history to 'training_history.png'")
#plt.show()

# Test on some sample images
print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    row = i // 5
    col = i % 5
    
    # Get prediction
    img = x_test[i].reshape(1, -1)
    pred = model.predict(img, verbose=0)[0][0]
    pred_class = 'Dog' if pred > 0.5 else 'Cat'
    confidence = pred if pred > 0.5 else 1 - pred
    
    # Actual label
    actual = 'Dog' if y_test[i] == 1 else 'Cat'
    correct = '✓' if (pred > 0.5) == (y_test[i] == 1) else '✗'
    
    axes[row, col].imshow(x_test[i])
    axes[row, col].set_title(f'{pred_class} ({confidence:.2f})\nActual: {actual} {correct}')
    axes[row, col].axis('off')

#plt.tight_layout()
#plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
print("Saved sample predictions to 'sample_predictions.png'")
#plt.show()

# Export model weights
print("\n" + "="*50)
print("EXPORTING MODEL WEIGHTS")
print("="*50)

# Create directory for saved models
import os
os.makedirs('saved_models', exist_ok=True)

# Method 1: Save entire model (architecture + weights + optimizer state)
model_path = 'saved_models/dog_cat_classifier_full.h5'
model.save(model_path)
print(f"✓ Saved full model to: {model_path}")
print(f"  File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

# Method 2: Save only weights
weights_path = 'saved_models/dog_cat_classifier.weights.h5'
model.save_weights(weights_path)
print(f"✓ Saved weights only to: {weights_path}")
print(f"  File size: {os.path.getsize(weights_path) / 1024 / 1024:.2f} MB")

# Method 3: Save in native Keras format (recommended)
keras_path = 'saved_models/dog_cat_classifier.keras'
model.save(keras_path)
print(f"✓ Saved in native Keras format to: {keras_path}")
print(f"  File size: {os.path.getsize(keras_path) / 1024 / 1024:.2f} MB")

# Verify we can load the model
print("\nVerifying saved model...")
loaded_model = keras.models.load_model(model_path)
test_loss_loaded, test_acc_loaded = loaded_model.evaluate(x_test_flat, y_test, verbose=0)
print(f"Loaded model test accuracy: {test_acc_loaded:.4f}")
print(f"Original model test accuracy: {test_accuracy:.4f}")
print(f"Match: {'✓' if abs(test_acc_loaded - test_accuracy) < 0.001 else '✗'}")

print("\n✓ Model weights exported successfully!")


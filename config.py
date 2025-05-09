"""
Configuration settings for the Facial Recognition System.

This module contains all the configurable parameters used throughout the system.
Modify these settings to customize the behavior of the facial recognition system.
"""

import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "facial_recognition_model.pkl")
LOG_PATH = os.path.join(BASE_DIR, "facial_recognition.log")

# Create directories if they don't exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Face detection parameters
FACE_DETECTION = {
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "min_size": (30, 30)
}

# Face recognition parameters
FACE_RECOGNITION = {
    "distance_threshold": 0.6,  # Threshold for face distance (lower is more strict)
    "confidence_threshold": 0.7  # Confidence threshold for the SVM classifier
}

# Dataset creation parameters
DATASET_CREATION = {
    "default_count": 20,  # Default number of images to capture per person
    "image_quality": 95,  # JPEG quality (0-100)
    "min_face_size": 100  # Minimum face width/height in pixels
}

# Training parameters
TRAINING = {
    "test_size": 0.2,      # Proportion of dataset to use for testing
    "random_state": 42,    # Random seed for reproducibility
    "svm_kernel": "linear", # SVM kernel type
    "cv_folds": 5          # Number of cross-validation folds
}

# Video parameters
VIDEO = {
    "frame_width": 640,    # Width of the video frame
    "frame_height": 480,   # Height of the video frame
    "fps": 30              # Frames per second
}

# GUI parameters (if implemented)
GUI = {
    "window_title": "Facial Recognition System",
    "window_width": 800,
    "window_height": 600,
    "theme": "system"  # "system", "light", or "dark"
}
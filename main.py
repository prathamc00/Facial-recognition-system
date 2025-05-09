#!/usr/bin/env python3
"""
Facial Recognition System - Main Application

This module serves as the entry point for the facial recognition application.
It provides a command-line interface to interact with the system's features.
"""

import os
import sys
import argparse
import time
import cv2
import logging

# Import local modules
from facial_recognition import FacialRecognitionSystem
from utils.dataset_utils import create_dataset
from config import DATASET_PATH, MODEL_PATH, LOG_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('facial_recognition')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Facial Recognition System')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Dataset creation parser
    dataset_parser = subparsers.add_parser('create-dataset', help='Create a dataset of face images')
    dataset_parser.add_argument('--name', type=str, help='Name of the person')
    dataset_parser.add_argument('--count', type=int, default=20, help='Number of images to capture')
    dataset_parser.add_argument('--output', type=str, default=DATASET_PATH, help='Output directory')
    
    # Model training parser
    train_parser = subparsers.add_parser('train', help='Train the facial recognition model')
    train_parser.add_argument('--dataset', type=str, default=DATASET_PATH, help='Dataset directory')
    train_parser.add_argument('--output', type=str, default=MODEL_PATH, help='Output model file')
    
    # Recognition parser
    recognize_parser = subparsers.add_parser('recognize', help='Run facial recognition')
    recognize_parser.add_argument('--model', type=str, default=MODEL_PATH, help='Model file')
    recognize_parser.add_argument('--camera', type=int, default=0, help='Camera index')
    recognize_parser.add_argument('--image', type=str, help='Image file (instead of camera)')
    
    # Info parser
    info_parser = subparsers.add_parser('info', help='Display information about the system')
    
    return parser.parse_args()


def run_dataset_creation(args):
    """Run the dataset creation process."""
    logger.info(f"Creating dataset in {args.output}")
    
    if args.name:
        create_dataset(args.output, args.name, args.count)
    else:
        # Interactive mode
        while True:
            name = input("Enter person's name (or 'q' to quit): ")
            if name.lower() == 'q':
                break
            count = int(input(f"Number of images to capture (default: {args.count}): ") or args.count)
            create_dataset(args.output, name, count)
            
            another = input("Add another person? (y/n): ")
            if another.lower() != 'y':
                break
    
    logger.info("Dataset creation completed")


def run_model_training(args):
    """Run the model training process."""
    logger.info(f"Training model using dataset from {args.dataset}")
    
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset directory {args.dataset} does not exist")
        return
    
    # Create directories for model if they don't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    facial_system = FacialRecognitionSystem()
    
    # Start timing
    start_time = time.time()
    
    # Train the model
    if facial_system.train_model(args.dataset):
        # Save the model
        facial_system.save_model(args.output)
        
        # Report training time
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
    else:
        logger.error("Model training failed")


def run_recognition(args):
    """Run the facial recognition process."""
    facial_system = FacialRecognitionSystem()
    
    # Load the model if specified
    if args.model and os.path.exists(args.model):
        logger.info(f"Loading model from {args.model}")
        facial_system.load_model(args.model)
    else:
        logger.warning("No model specified or model not found. Using simple recognition mode.")
    
    # Run on image if specified
    if args.image:
        if not os.path.exists(args.image):
            logger.error(f"Image file {args.image} not found")
            return
        
        logger.info(f"Processing image {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            logger.error("Failed to load image")
            return
        
        # Process the image
        results = facial_system.process_image(image)
        image = facial_system.draw_results(image, results)
        
        # Display results
        for result in results:
            logger.info(f"Detected: {result['name']} (confidence: {result['confidence']:.2f})")
        
        # Show the image
        cv2.imshow('Recognition Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Run on webcam
        logger.info(f"Starting video recognition using camera {args.camera}")
        facial_system.run_video_recognition(camera_index=args.camera)


def show_system_info():
    """Display information about the system."""
    print("\nFacial Recognition System Information")
    print("-" * 40)
    
    # Check OpenCV version
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Check if models directory exists
    model_dir = os.path.dirname(MODEL_PATH)
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        print(f"Available Models: {len(models)}")
        for model in models:
            model_path = os.path.join(model_dir, model)
            model_size = os.path.getsize(model_path) / 1024  # KB
            print(f"  - {model} ({model_size:.2f} KB)")
    else:
        print("Models Directory: Not found")
    
    # Check if dataset directory exists
    if os.path.exists(DATASET_PATH):
        people = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
        print(f"Dataset Information:")
        print(f"  - People: {len(people)}")
        for person in people:
            person_dir = os.path.join(DATASET_PATH, person)
            images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  - {person}: {len(images)} images")
    else:
        print("Dataset Directory: Not found")
    
    # Check available cameras
    max_cameras = 5  # Check up to 5 cameras
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    print(f"Available Cameras: {available_cameras}")
    print("-" * 40)


def main():
    """Main function to run the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # If no command specified, show help
    if not args.command:
        print("Error: No command specified")
        print("Run 'python main.py -h' for help")
        return
    
    # Execute the appropriate command
    if args.command == 'create-dataset':
        run_dataset_creation(args)
    elif args.command == 'train':
        run_model_training(args)
    elif args.command == 'recognize':
        run_recognition(args)
    elif args.command == 'info':
        show_system_info()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("An unexpected error occurred")
        print(f"Error: {str(e)}")
        sys.exit(1)
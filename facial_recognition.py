"""
Facial Recognition System Core Module

This module contains the FacialRecognitionSystem class which provides
all the core functionality for face detection and recognition.
"""

import cv2
import numpy as np
import os
import pickle
import logging
try:
    import face_recognition
except ImportError:
    raise ImportError(
        "Could not import face_recognition. Please install it using:\n"
        "pip install face-recognition"
    )
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Import configuration
from config import FACE_DETECTION, FACE_RECOGNITION, TRAINING, VIDEO

# Set up logging
logger = logging.getLogger('facial_recognition.core')


class FacialRecognitionSystem:
    """
    Main class for facial recognition operations including face detection,
    feature extraction, model training, and face recognition.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the facial recognition system.
        
        Args:
            model_path (str, optional): Path to a pre-trained model. Defaults to None.
        """
        # Initialize the face detector
        cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        
        # Initialize variables for storing face data
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_recognizer = None
        self.label_encoder = LabelEncoder()
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
        logger.debug("FacialRecognitionSystem initialized")
    
    def detect_faces(self, image):
        """
        Detect faces in an image using OpenCV's Haar Cascade.
        
        Args:
            image (numpy.ndarray): The input image

        Returns:
            list: List of (x, y, w, h) tuples for detected faces
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION["scale_factor"],
            minNeighbors=FACE_DETECTION["min_neighbors"],
            minSize=FACE_DETECTION["min_size"]
        )
        
        logger.debug(f"Detected {len(faces)} faces")
        return faces
    
    def extract_face_encodings(self, image):
        """
        Extract face encodings using face_recognition library.
        
        Args:
            image (numpy.ndarray): The input image

        Returns:
            tuple: (face_encodings, face_locations)
        """
        # Convert to RGB (face_recognition uses RGB, OpenCV uses BGR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        logger.debug(f"Extracted {len(face_encodings)} face encodings")
        return face_encodings, face_locations
    
    def add_face(self, image, name):
        """
        Add a face to the known faces database.
        
        Args:
            image (numpy.ndarray): Image containing a face
            name (str): Name of the person

        Returns:
            bool: True if successful, False otherwise
        """
        face_encodings, _ = self.extract_face_encodings(image)
        
        if face_encodings:
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            logger.info(f"Added face for {name}")
            return True
        
        logger.warning(f"No face found to add for {name}")
        return False
    
    def train_model(self, data_dir):
        """
        Train facial recognition model from a directory of images.
        
        Args:
            data_dir (str): Directory containing subdirectories of face images

        Returns:
            bool: True if successful, False otherwise
        """
        X = []  # Face encodings
        y = []  # Person names
        
        logger.info(f"Starting model training from {data_dir}")
        
        # Process each person's directory
        person_count = 0
        total_images = 0
        
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            logger.info(f"Processing images for {person_name}")
            image_count = 0
            
            # Process each image for this person
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                try:
                    # Read and process image
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.warning(f"Failed to load image: {image_path}")
                        continue
                    
                    face_encodings, _ = self.extract_face_encodings(image)
                    
                    if face_encodings:
                        X.append(face_encodings[0])
                        y.append(person_name)
                        image_count += 1
                    else:
                        logger.warning(f"No face detected in {image_path}")
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
            
            if image_count > 0:
                logger.info(f"Processed {image_count} images for {person_name}")
                person_count += 1
                total_images += image_count
        
        if not X:
            logger.error("No faces found in the training data")
            return False
        
        logger.info(f"Training with {total_images} images from {person_count} people")
            
        # Convert labels to numerical form
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=TRAINING["test_size"], 
            random_state=TRAINING["random_state"]
        )
        
        # Train SVM classifier
        self.face_recognizer = SVC(
            kernel=TRAINING["svm_kernel"], 
            probability=True
        )
        self.face_recognizer.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.face_recognizer.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log detailed classification report
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        logger.info(f"Classification Report:\n{report}")
        
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        # Store the known face encodings and names
        self.known_face_encodings = X
        self.known_face_names = y
        
        return True
    
    def recognize_face(self, face_encoding):
        """
        Recognize a face from its encoding.
        
        Args:
            face_encoding (numpy.ndarray): The encoding of the face to recognize

        Returns:
            tuple: (name, confidence)
        """
        if not self.face_recognizer:
            # Use simple distance comparison if no trained model
            if not self.known_face_encodings:
                return "Unknown", 0.0
                
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if face_distances[best_match_index] < FACE_RECOGNITION["distance_threshold"]:
                confidence = 1 - face_distances[best_match_index]
                return self.known_face_names[best_match_index], confidence
            else:
                return "Unknown", 0.0
        else:
            # Use trained model
            prediction = self.face_recognizer.predict([face_encoding])
            proba = self.face_recognizer.predict_proba([face_encoding])
            max_proba = np.max(proba)

            if max_proba > FACE_RECOGNITION["confidence_threshold"]:
                person_id = prediction[0]
                # Handle case where inverse_transform returns None
                try:
                    transformed = self.label_encoder.inverse_transform([person_id])
                    person_name = transformed[0] if transformed is not None and len(transformed) > 0 else "Unknown"
                except (IndexError, TypeError):
                    person_name = "Unknown"
                return person_name, max_proba
            else:
                return "Unknown", max_proba
    
    def process_image(self, image):
        """
        Process an image to detect and recognize faces.
        
        Args:
            image (numpy.ndarray): The input image

        Returns:
            list: List of dictionaries containing recognition results
        """
        results = []
        face_encodings, face_locations = self.extract_face_encodings(image)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            name, confidence = self.recognize_face(face_encoding)
            results.append({
                'name': name,
                'confidence': confidence,
                'location': face_location
            })
        
        return results
    
    def draw_results(self, image, results):
        """
        Draw bounding boxes and names on the image.
        
        Args:
            image (numpy.ndarray): The input image
            results (list): Recognition results from process_image

        Returns:
            numpy.ndarray: Image with bounding boxes and names drawn
        """
        # Make a copy of the image to avoid modifying the original
        annotated_image = image.copy()
        
        for result in results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']
            
            # Draw a box around the face
            cv2.rectangle(annotated_image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(annotated_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            label = f"{name} ({confidence:.2f})"
            cv2.putText(annotated_image, label, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_image
    
    def save_model(self, model_path="facial_recognition_model.pkl"):
        """
        Save the model to disk.
        
        Args:
            model_path (str, optional): Path to save the model. Defaults to "facial_recognition_model.pkl".

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.face_recognizer:
            logger.error("No model to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            
            model_data = {
                'recognizer': self.face_recognizer,
                'label_encoder': self.label_encoder,
                'known_face_encodings': self.known_face_encodings,
                'known_face_names': self.known_face_names
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path="facial_recognition_model.pkl"):
        """
        Load the model from disk.
        
        Args:
            model_path (str, optional): Path to the model file. Defaults to "facial_recognition_model.pkl".

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return False
            
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.face_recognizer = model_data['recognizer']
            self.label_encoder = model_data['label_encoder']
            self.known_face_encodings = model_data['known_face_encodings']
            self.known_face_names = model_data['known_face_names']
            
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Model contains {len(self.known_face_names)} unique people")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def run_video_recognition(self, camera_index=0):
        """
        Run facial recognition on webcam video.
        
        Args:
            camera_index (int, optional): Index of the camera to use. Defaults to 0.
        """
        video_capture = cv2.VideoCapture(camera_index)
        
        # Set video parameters
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO["frame_width"])
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO["frame_height"])
        video_capture.set(cv2.CAP_PROP_FPS, VIDEO["fps"])
        
        if not video_capture.isOpened():
            logger.error(f"Could not open camera {camera_index}")
            return
            
        logger.info("Starting video recognition. Press 'q' to quit")
        
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()
                
                if not ret:
                    logger.error("Failed to grab frame")
                    break
                    
                # Process the frame
                results = self.process_image(frame)
                frame = self.draw_results(frame, results)
                
                # Display the FPS
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the resulting frame
                cv2.imshow('Facial Recognition', frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Release the webcam and close windows
            video_capture.release()
            cv2.destroyAllWindows()
            logger.info("Video recognition stopped")
#!/usr/bin/env python3

import cv2
import numpy as np
import os
import json
import rospy
from datetime import datetime

class OpenCVFaceRecognizer:
    def __init__(self, db_path="data/faces/"):
        """
        OpenCV-based face recognizer as alternative to face_recognition library.
        """
        self.db_path = db_path
        self.ensure_database_directory()
        
        # OpenCV face detector and recognizer
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Face database
        self.known_faces = []
        self.known_labels = []
        self.known_names = {}  # label -> name mapping
        self.known_metadata = {}
        self.next_label = 0
        self.is_trained = False
        
        # Load existing data
        self.load_face_database()
        
        rospy.loginfo(f"OpenCV Face Recognizer initialized with {len(self.known_names)} known faces")
    
    def ensure_database_directory(self):
        """Ensure face database directories exist."""
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(os.path.join(self.db_path, "photos"), exist_ok=True)
        os.makedirs(os.path.join(self.db_path, "metadata"), exist_ok=True)
    
    def load_face_database(self):
        """Load existing face data from disk."""
        try:
            # Load metadata
            metadata_file = os.path.join(self.db_path, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.known_metadata = data.get('metadata', {})
                    self.known_names = data.get('names', {})
                    self.next_label = data.get('next_label', 0)
                
                rospy.loginfo(f"Loaded metadata for {len(self.known_names)} people")
            
            # Load trained model if it exists
            model_file = os.path.join(self.db_path, "face_model.yml")
            if os.path.exists(model_file):
                self.recognizer.read(model_file)
                self.is_trained = True
                rospy.loginfo("Loaded trained face recognition model")
                
        except Exception as e:
            rospy.logwarn(f"Error loading face database: {e}")
            self.known_metadata = {}
            self.known_names = {}
            self.next_label = 0
    
    def save_face_database(self):
        """Save face data to disk."""
        try:
            # Save metadata
            metadata_file = os.path.join(self.db_path, "metadata.json")
            data = {
                'metadata': self.known_metadata,
                'names': self.known_names,
                'next_label': self.next_label
            }
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Save trained model
            if self.is_trained:
                model_file = os.path.join(self.db_path, "face_model.yml")
                self.recognizer.save(model_file)
            
            rospy.loginfo("Face database saved successfully")
            
        except Exception as e:
            rospy.logerr(f"Error saving face database: {e}")
    
    def detect_faces(self, image):
        """
        Detect faces in image using OpenCV.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def add_face(self, image, person_id, person_name=None, save_photo=True):
        """
        Add a new face to the recognition database.

        """
        try:
            rospy.loginfo(f"Adding face for person: {person_id}")
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                rospy.logwarn("No faces detected in the image")
                return False
            
            if len(faces) > 1:
                rospy.logwarn(f"Multiple faces detected ({len(faces)}), using the largest one")
            
            # Get the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Extract face region
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize face to standard size
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Get or create label for this person
            if person_id in self.known_names.values():
                # Find existing label
                label = [k for k, v in self.known_names.items() if v == person_id][0]
                label = int(label)
            else:
                # Create new label
                label = self.next_label
                self.known_names[str(label)] = person_id
                self.next_label += 1
            
            # Add to training data
            self.known_faces.append(face_roi)
            self.known_labels.append(label)
            
            # Update metadata
            timestamp = datetime.now().isoformat()
            if person_id not in self.known_metadata:
                self.known_metadata[person_id] = {
                    'first_added': timestamp,
                    'recognition_count': 0,
                    'last_seen': timestamp,
                    'label': label
                }
            
            self.known_metadata[person_id]['last_updated'] = timestamp
            if person_name:
                self.known_metadata[person_id]['name'] = person_name
            
            # Train the recognizer
            self.train_recognizer()
            
            # Save photo if requested
            if save_photo:
                self.save_person_photo(image, person_id, largest_face)
            
            # Save database
            self.save_face_database()
            
            rospy.loginfo(f"Successfully added face for {person_id}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error adding face: {e}")
            return False
    
    def train_recognizer(self):
        """Train the face recognizer with current data."""
        try:
            if len(self.known_faces) > 0:
                self.recognizer.train(self.known_faces, np.array(self.known_labels))
                self.is_trained = True
                rospy.loginfo("Face recognizer trained successfully")
            else:
                rospy.logwarn("No faces available for training")
                
        except Exception as e:
            rospy.logerr(f"Error training recognizer: {e}")
    
    def recognize_face(self, image):
        """
        Recognize faces in the given image.
        """
        try:
            if not self.is_trained:
                rospy.loginfo("No trained model available")
                return []
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                rospy.loginfo("No faces detected in image")
                return []
            
            results = []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            for (x, y, w, h) in faces:
                # Extract and resize face
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                # Recognize face
                label, confidence = self.recognizer.predict(face_roi)
                
                # Convert confidence to similarity (lower is better for LBPH)
                similarity = max(0, (100 - confidence) / 100)
                
                person_id = "unknown"
                is_known = False
                
                # Check if confidence is good enough (threshold)
                if confidence < 80:  # Adjust threshold as needed
                    person_id = self.known_names.get(str(label), "unknown")
                    is_known = person_id != "unknown"
                    
                    if is_known:
                        # Update metadata
                        self.known_metadata[person_id]['recognition_count'] += 1
                        self.known_metadata[person_id]['last_seen'] = datetime.now().isoformat()
                
                results.append({
                    'person_id': person_id,
                    'confidence': similarity,
                    'location': (x, y, w, h),
                    'is_known': is_known
                })
                
                rospy.loginfo(f"Face recognized: {person_id} (confidence: {similarity:.2f})")
            
            # Save updated metadata
            if results:
                self.save_face_database()
            
            return results
            
        except Exception as e:
            rospy.logerr(f"Error recognizing faces: {e}")
            return []
    
    def save_person_photo(self, image, person_id, face_location):
        """Save a photo of the person."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save full image
            full_image_path = os.path.join(self.db_path, "photos", f"{person_id}_full_{timestamp}.jpg")
            cv2.imwrite(full_image_path, image)
            
            # Save face crop
            x, y, w, h = face_location
            face_img = image[y:y+h, x:x+w]
            face_image_path = os.path.join(self.db_path, "photos", f"{person_id}_face_{timestamp}.jpg")
            cv2.imwrite(face_image_path, face_img)
            
            rospy.loginfo(f"Saved photos for {person_id}")
            
        except Exception as e:
            rospy.logerr(f"Error saving person photo: {e}")
    
    def get_person_info(self, person_id):
        """Get stored information about a person."""
        return self.known_metadata.get(person_id, None)
    
    def get_all_known_people(self):
        """Get list of all known people."""
        return list(self.known_metadata.keys())
    
    def visualize_recognition_results(self, image, results):
        """Draw bounding boxes and labels on the image."""
        try:
            annotated_image = image.copy()
            
            for result in results:
                person_id = result['person_id']
                confidence = result['confidence']
                x, y, w, h = result['location']
                
                # Choose colors
                if result['is_known']:
                    color = (0, 255, 0)  # Green for known faces
                    label = f"{person_id} ({confidence:.2f})"
                else:
                    color = (0, 0, 255)  # Red for unknown faces
                    label = "Unknown"
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
                
                # Draw label
                cv2.putText(annotated_image, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return annotated_image
            
        except Exception as e:
            rospy.logerr(f"Error visualizing results: {e}")
            return image
    
    def get_database_stats(self):
        """Get database statistics."""
        total_recognitions = sum(
            metadata.get('recognition_count', 0) 
            for metadata in self.known_metadata.values()
        )
        
        return {
            'total_people': len(self.known_metadata),
            'total_recognitions': total_recognitions,
            'is_trained': self.is_trained,
            'training_samples': len(self.known_faces)
        }
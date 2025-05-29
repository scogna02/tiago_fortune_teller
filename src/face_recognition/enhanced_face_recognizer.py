#!/usr/bin/env python3

import face_recognition
import numpy as np
import cv2
import os
import pickle
import rospy
from datetime import datetime
import json

class FaceRecognizer:
    def __init__(self, db_path="data/faces/"):
        """
        Enhanced face recognizer with camera integration and persistent storage.
        
        Args:
            db_path (str): Path to store face encodings and metadata
        """
        self.db_path = db_path
        self.ensure_database_directory()
        
        # Face database
        self.known_encodings = []
        self.known_names = []
        self.known_metadata = {}
        
        # Face recognition settings
        self.face_recognition_tolerance = 0.6
        self.face_detection_model = "hog"  # or "cnn" for better accuracy but slower
        
        # Load existing face database
        self.load_face_database()
        
        rospy.loginfo(f"Face Recognizer initialized with {len(self.known_names)} known faces")
    
    def ensure_database_directory(self):
        """Ensure face database directories exist."""
        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(os.path.join(self.db_path, "encodings"), exist_ok=True)
        os.makedirs(os.path.join(self.db_path, "photos"), exist_ok=True)
        os.makedirs(os.path.join(self.db_path, "metadata"), exist_ok=True)
    
    def load_face_database(self):
        """Load existing face encodings and metadata from disk."""
        try:
            # Load encodings
            encodings_file = os.path.join(self.db_path, "face_database.pkl")
            if os.path.exists(encodings_file):
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get('encodings', [])
                    self.known_names = data.get('names', [])
                
                rospy.loginfo(f"Loaded {len(self.known_names)} face encodings from database")
            
            # Load metadata
            metadata_file = os.path.join(self.db_path, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.known_metadata = json.load(f)
                
                rospy.loginfo(f"Loaded metadata for {len(self.known_metadata)} people")
                
        except Exception as e:
            rospy.logwarn(f"Error loading face database: {e}")
            self.known_encodings = []
            self.known_names = []
            self.known_metadata = {}
    
    def save_face_database(self):
        """Save face encodings and metadata to disk."""
        try:
            # Save encodings
            encodings_file = os.path.join(self.db_path, "face_database.pkl")
            data = {
                'encodings': self.known_encodings,
                'names': self.known_names
            }
            with open(encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            metadata_file = os.path.join(self.db_path, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(self.known_metadata, f, indent=2)
            
            rospy.loginfo("Face database saved successfully")
            
        except Exception as e:
            rospy.logerr(f"Error saving face database: {e}")
    
    def add_face(self, image, person_id, person_name=None, save_photo=True):
        """
        Add a new face to the recognition database.
        
        Args:
            image (numpy.ndarray): Image containing the face (RGB format)
            person_id (str): Unique identifier for the person
            person_name (str, optional): Human-readable name for the person
            save_photo (bool): Whether to save the photo to disk
            
        Returns:
            bool: True if face was successfully added, False otherwise
        """
        try:
            rospy.loginfo(f"Adding face for person: {person_id}")
            
            # Detect faces in the image
            face_locations = face_recognition.face_locations(image, model=self.face_detection_model)
            
            if not face_locations:
                rospy.logwarn("No faces detected in the image")
                return False
            
            if len(face_locations) > 1:
                rospy.logwarn(f"Multiple faces detected ({len(face_locations)}), using the first one")
            
            # Get face encoding for the first detected face
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                rospy.logwarn("Could not generate face encoding")
                return False
            
            face_encoding = face_encodings[0]
            
            # Check if this person already exists
            if person_id in self.known_names:
                # Update existing encoding
                person_index = self.known_names.index(person_id)
                self.known_encodings[person_index] = face_encoding
                rospy.loginfo(f"Updated face encoding for existing person: {person_id}")
            else:
                # Add new person
                self.known_encodings.append(face_encoding)
                self.known_names.append(person_id)
                rospy.loginfo(f"Added new person to database: {person_id}")
            
            # Update metadata
            timestamp = datetime.now().isoformat()
            if person_id not in self.known_metadata:
                self.known_metadata[person_id] = {
                    'first_added': timestamp,
                    'recognition_count': 0,
                    'last_seen': timestamp
                }
            
            self.known_metadata[person_id]['last_updated'] = timestamp
            if person_name:
                self.known_metadata[person_id]['name'] = person_name
            
            # Save photo if requested
            if save_photo:
                self.save_person_photo(image, person_id, face_locations[0])
            
            # Save updated database
            self.save_face_database()
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Error adding face: {e}")
            return False
    
    def save_person_photo(self, image, person_id, face_location):
        """
        Save a photo of the person for reference.
        
        Args:
            image (numpy.ndarray): Original image (RGB)
            person_id (str): Person identifier
            face_location (tuple): Bounding box of the face (top, right, bottom, left)
        """
        try:
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract face region with some padding
            top, right, bottom, left = face_location
            padding = 50
            face_img = bgr_image[max(0, top-padding):bottom+padding, 
                                max(0, left-padding):right+padding]
            
            # Save full image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_image_path = os.path.join(self.db_path, "photos", f"{person_id}_full_{timestamp}.jpg")
            cv2.imwrite(full_image_path, bgr_image)
            
            # Save face crop
            face_image_path = os.path.join(self.db_path, "photos", f"{person_id}_face_{timestamp}.jpg")
            cv2.imwrite(face_image_path, face_img)
            
            rospy.loginfo(f"Saved photos for {person_id}")
            
        except Exception as e:
            rospy.logerr(f"Error saving person photo: {e}")
    
    def recognize_face(self, image):
        """
        Recognize faces in the given image.
        
        Args:
            image (numpy.ndarray): Image to analyze (RGB format)
            
        Returns:
            list: List of dictionaries containing recognition results
                  [{'person_id': str, 'confidence': float, 'location': tuple}, ...]
        """
        try:
            if not self.known_encodings:
                rospy.loginfo("No known faces in database")
                return []
            
            rospy.loginfo("Recognizing faces in image...")
            
            # Detect faces
            face_locations = face_recognition.face_locations(image, model=self.face_detection_model)
            
            if not face_locations:
                rospy.loginfo("No faces detected in image")
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            results = []
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_encodings, face_encoding, tolerance=self.face_recognition_tolerance
                )
                
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                
                person_id = "unknown"
                confidence = 0.0
                
                if True in matches:
                    # Find best match
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        person_id = self.known_names[best_match_index]
                        confidence = 1.0 - face_distances[best_match_index]
                        
                        # Update metadata
                        self.known_metadata[person_id]['recognition_count'] += 1
                        self.known_metadata[person_id]['last_seen'] = datetime.now().isoformat()
                
                results.append({
                    'person_id': person_id,
                    'confidence': confidence,
                    'location': face_location,
                    'is_known': person_id != "unknown"
                })
                
                rospy.loginfo(f"Face recognized: {person_id} (confidence: {confidence:.2f})")
            
            # Save updated metadata
            if results:
                self.save_face_database()
            
            return results
            
        except Exception as e:
            rospy.logerr(f"Error recognizing faces: {e}")
            return []
    
    def get_person_info(self, person_id):
        """
        Get stored information about a person.
        
        Args:
            person_id (str): Person identifier
            
        Returns:
            dict: Person metadata or None if not found
        """
        return self.known_metadata.get(person_id, None)
    
    def get_all_known_people(self):
        """
        Get list of all known people.
        
        Returns:
            list: List of person IDs
        """
        return list(self.known_names)
    
    def remove_person(self, person_id):
        """
        Remove a person from the database.
        
        Args:
            person_id (str): Person identifier
            
        Returns:
            bool: True if person was removed, False if not found
        """
        try:
            if person_id in self.known_names:
                index = self.known_names.index(person_id)
                del self.known_names[index]
                del self.known_encodings[index]
                
                if person_id in self.known_metadata:
                    del self.known_metadata[person_id]
                
                self.save_face_database()
                rospy.loginfo(f"Removed person from database: {person_id}")
                return True
            else:
                rospy.logwarn(f"Person not found in database: {person_id}")
                return False
                
        except Exception as e:
            rospy.logerr(f"Error removing person: {e}")
            return False
    
    def get_database_stats(self):
        """
        Get statistics about the face database.
        
        Returns:
            dict: Database statistics
        """
        total_recognitions = sum(
            metadata.get('recognition_count', 0) 
            for metadata in self.known_metadata.values()
        )
        
        return {
            'total_people': len(self.known_names),
            'total_recognitions': total_recognitions,
            'database_size_mb': len(pickle.dumps(self.known_encodings)) / 1024 / 1024
        }
    
    def visualize_recognition_results(self, image, results):
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image (numpy.ndarray): Original image (RGB)
            results (list): Recognition results from recognize_face()
            
        Returns:
            numpy.ndarray: Image with annotations (BGR for display)
        """
        try:
            # Convert RGB to BGR for OpenCV
            annotated_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
            
            for result in results:
                person_id = result['person_id']
                confidence = result['confidence']
                top, right, bottom, left = result['location']
                
                # Choose colors based on recognition status
                if result['is_known']:
                    color = (0, 255, 0)  # Green for known faces
                    label = f"{person_id} ({confidence:.2f})"
                else:
                    color = (0, 0, 255)  # Red for unknown faces
                    label = "Unknown"
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (left, top), (right, bottom), color, 2)
                
                # Draw label
                cv2.putText(annotated_image, label, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return annotated_image
            
        except Exception as e:
            rospy.logerr(f"Error visualizing results: {e}")
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
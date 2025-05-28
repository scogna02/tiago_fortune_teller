#!/usr/bin/env python3

import os
import json
import pickle
import numpy as np
from datetime import datetime
import rospy

class GraphManager:
    def __init__(self, data_path="data/knowledge_graph/"):
        """
        Initialize the Knowledge Graph Manager for storing user interactions and data.
        
        Args:
            data_path (str): Path to store knowledge graph data
        """
        self.data_path = data_path
        self.ensure_data_directory()
        
        # In-memory storage for current session
        self.user_data = {}
        self.interaction_history = {}
        self.embeddings = {}
        
        # Load existing data
        self.load_data()
        
        rospy.loginfo("Knowledge Graph Manager initialized")
    
    def ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "users"), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "interactions"), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "embeddings"), exist_ok=True)
    
    def load_data(self):
        """Load existing user data from disk."""
        try:
            # Load user data
            user_data_file = os.path.join(self.data_path, "user_data.json")
            if os.path.exists(user_data_file):
                with open(user_data_file, 'r') as f:
                    self.user_data = json.load(f)
                    
            # Load interaction history
            interaction_file = os.path.join(self.data_path, "interactions.json")
            if os.path.exists(interaction_file):
                with open(interaction_file, 'r') as f:
                    self.interaction_history = json.load(f)
                    
            rospy.loginfo(f"Loaded data for {len(self.user_data)} users")
            
        except Exception as e:
            rospy.logwarn(f"Error loading knowledge graph data: {e}")
            self.user_data = {}
            self.interaction_history = {}
    
    def save_data(self):
        """Save current data to disk."""
        try:
            # Save user data
            user_data_file = os.path.join(self.data_path, "user_data.json")
            with open(user_data_file, 'w') as f:
                json.dump(self.user_data, f, indent=2)
                
            # Save interaction history
            interaction_file = os.path.join(self.data_path, "interactions.json")
            with open(interaction_file, 'w') as f:
                json.dump(self.interaction_history, f, indent=2)
                
            rospy.loginfo("Knowledge graph data saved successfully")
            
        except Exception as e:
            rospy.logerr(f"Error saving knowledge graph data: {e}")
    
    def update(self, person_id, interaction_data):
        """
        Update the knowledge graph with new user interaction or data.
        
        Args:
            person_id (str): Unique identifier for the person
            interaction_data (str): Information about the interaction or user attribute
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize user data if not exists
        if person_id not in self.user_data:
            self.user_data[person_id] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'visit_count': 0,
                'attributes': {}
            }
        
        # Update last seen and visit count
        self.user_data[person_id]['last_seen'] = timestamp
        
        # Parse interaction data
        if interaction_data == "visited_fortune_teller":
            self.user_data[person_id]['visit_count'] += 1
        elif ":" in interaction_data:
            # Parse attribute data (e.g., "name:John", "age:25")
            key, value = interaction_data.split(":", 1)
            self.user_data[person_id]['attributes'][key] = value
        
        # Add to interaction history
        if person_id not in self.interaction_history:
            self.interaction_history[person_id] = []
        
        self.interaction_history[person_id].append({
            'timestamp': timestamp,
            'interaction': interaction_data
        })
        
        # Update embeddings
        self.update_embedding(person_id)
        
        # Save data periodically
        self.save_data()
        
        rospy.loginfo(f"Updated knowledge graph for {person_id}: {interaction_data}")
    
    def update_embedding(self, person_id):
        """
        Update the embedding vector for a user based on their data and interactions.
        
        Args:
            person_id (str): Unique identifier for the person
        """
        try:
            user_info = self.user_data.get(person_id, {})
            
            # Create a simple embedding based on user attributes and interactions
            # In a real implementation, you would use PyKEEN or similar
            embedding_features = []
            
            # Visit frequency (normalized)
            visit_count = user_info.get('visit_count', 0)
            embedding_features.append(min(visit_count / 10.0, 1.0))  # Normalize to 0-1
            
            # User attributes encoding
            attributes = user_info.get('attributes', {})
            
            # Age encoding (normalized to 0-1, assuming max age 100)
            try:
                age = int(attributes.get('age', 30))
                embedding_features.append(age / 100.0)
            except ValueError:
                embedding_features.append(0.3)  # Default value
            
            # Profession encoding (simple hash-based)
            profession = attributes.get('profession', 'unknown').lower()
            profession_hash = hash(profession) % 100 / 100.0
            embedding_features.append(profession_hash)
            
            # Interaction history length
            interaction_count = len(self.interaction_history.get(person_id, []))
            embedding_features.append(min(interaction_count / 50.0, 1.0))
            
            # Time since first visit (days, normalized)
            try:
                first_seen = datetime.fromisoformat(user_info.get('first_seen', datetime.now().isoformat()))
                days_since_first = (datetime.now() - first_seen).days
                embedding_features.append(min(days_since_first / 365.0, 1.0))
            except:
                embedding_features.append(0.0)
            
            # Pad to fixed size (16 dimensions)
            while len(embedding_features) < 16:
                embedding_features.append(0.0)
            
            # Create numpy array
            embedding = np.array(embedding_features[:16], dtype=np.float32)
            
            # Store embedding
            self.embeddings[person_id] = embedding
            
            # Save embedding to disk
            embedding_file = os.path.join(self.data_path, "embeddings", f"{person_id}.npy")
            np.save(embedding_file, embedding)
            
        except Exception as e:
            rospy.logwarn(f"Error updating embedding for {person_id}: {e}")
            # Default embedding
            self.embeddings[person_id] = np.random.random(16).astype(np.float32)
    
    def get_embedding(self, person_id):
        """
        Get the latest embedding vector for a person.
        
        Args:
            person_id (str): Unique identifier for the person
            
        Returns:
            numpy.ndarray: Embedding vector for the person
        """
        if person_id in self.embeddings:
            return self.embeddings[person_id]
        
        # Try to load from disk
        embedding_file = os.path.join(self.data_path, "embeddings", f"{person_id}.npy")
        if os.path.exists(embedding_file):
            try:
                embedding = np.load(embedding_file)
                self.embeddings[person_id] = embedding
                return embedding
            except Exception as e:
                rospy.logwarn(f"Error loading embedding for {person_id}: {e}")
        
        # Generate new embedding if not found
        self.update_embedding(person_id)
        return self.embeddings.get(person_id, np.random.random(16).astype(np.float32))
    
    def get_user_info(self, person_id):
        """
        Get stored information about a user.
        
        Args:
            person_id (str): Unique identifier for the person
            
        Returns:
            dict: User information and attributes
        """
        return self.user_data.get(person_id, {})
    
    def get_interaction_history(self, person_id):
        """
        Get interaction history for a user.
        
        Args:
            person_id (str): Unique identifier for the person
            
        Returns:
            list: List of interactions with timestamps
        """
        return self.interaction_history.get(person_id, [])
    
    def get_all_users(self):
        """
        Get list of all known users.
        
        Returns:
            list: List of person IDs
        """
        return list(self.user_data.keys())
    
    def get_stats(self):
        """
        Get statistics about the knowledge graph.
        
        Returns:
            dict: Statistics about users and interactions
        """
        total_users = len(self.user_data)
        total_interactions = sum(len(interactions) for interactions in self.interaction_history.values())
        
        return {
            'total_users': total_users,
            'total_interactions': total_interactions,
            'average_interactions_per_user': total_interactions / max(total_users, 1)
        }
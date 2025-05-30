#!/usr/bin/env python3

"""
PyKEEN Integration for Tiago Fortune Teller Knowledge Graph
This module shows how to integrate PyKEEN for sophisticated knowledge graph embeddings
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import rospy

# PyKEEN imports
from pykeen.datasets import Dataset
from pykeen.models import TransE, ComplEx, RotatE
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch

class PyKEENGraphManager:
    def __init__(self, data_path="data/knowledge_graph/"):
        """
        PyKEEN-based Knowledge Graph Manager for sophisticated embeddings.
        
        Args:
            data_path (str): Path to store knowledge graph data
        """
        self.data_path = data_path
        self.ensure_data_directory()
        
        # Knowledge graph storage
        self.triples = []  # (subject, predicate, object) triples
        self.entities = set()
        self.relations = set()
        
        # PyKEEN components
        self.model = None
        self.triples_factory = None
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.id_to_entity = {}
        self.id_to_relation = {}
        
        # User data for context
        self.user_data = {}
        self.interaction_history = {}
        
        # Model configuration
        self.embedding_dim = 64
        self.model_type = "TransE"  # Can be TransE, ComplEx, RotatE, etc.
        
        # Load existing data
        self.load_data()
        
        rospy.loginfo("PyKEEN Knowledge Graph Manager initialized")
    
    def ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "triples"), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "embeddings"), exist_ok=True)
    
    def load_data(self):
        """Load existing knowledge graph data."""
        try:
            # Load triples
            triples_file = os.path.join(self.data_path, "triples.json")
            if os.path.exists(triples_file):
                with open(triples_file, 'r') as f:
                    data = json.load(f)
                    self.triples = data.get('triples', [])
                    self.entities = set(data.get('entities', []))
                    self.relations = set(data.get('relations', []))
            
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
            
            # Load trained model if available
            self.load_trained_model()
            
            rospy.loginfo(f"Loaded {len(self.triples)} triples, {len(self.entities)} entities, {len(self.relations)} relations")
            
        except Exception as e:
            rospy.logwarn(f"Error loading knowledge graph data: {e}")
    
    def save_data(self):
        """Save knowledge graph data to disk."""
        try:
            # Save triples and entities
            triples_file = os.path.join(self.data_path, "triples.json")
            data = {
                'triples': self.triples,
                'entities': list(self.entities),
                'relations': list(self.relations)
            }
            with open(triples_file, 'w') as f:
                json.dump(data, f, indent=2)
            
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
    
    def add_triple(self, subject, predicate, object_):
        """
        Add a triple to the knowledge graph.
        
        Args:
            subject (str): Subject entity
            predicate (str): Relation/predicate
            object_ (str): Object entity
        """
        triple = (subject, predicate, object_)
        
        if triple not in self.triples:
            self.triples.append(triple)
            self.entities.add(subject)
            self.entities.add(object_)
            self.relations.add(predicate)
            
            rospy.loginfo(f"Added triple: {subject} --{predicate}--> {object_}")
            
            # Retrain model when we have enough data and every 10 new triples
            if len(self.triples) >= 10 and len(self.triples) % 10 == 0:
                rospy.loginfo("Triggering model retraining...")
                self.train_model()
    
    def update_user_data(self, person_id, interaction_data):
        """
        Update user data and create knowledge graph triples.
        
        Args:
            person_id (str): Unique identifier for the person
            interaction_data (str): Information about the interaction
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize user data if not exists
        if person_id not in self.user_data:
            self.user_data[person_id] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'visit_count': 1,
                'attributes': {}
            }
        
        # Update user data
        self.user_data[person_id]['last_seen'] = timestamp
        
        # Create knowledge graph triples based on interaction
        if interaction_data == "visited_fortune_teller":
            self.user_data[person_id]['visit_count'] += 1
            
            # Add triples
            self.add_triple(person_id, "visited", "fortune_teller")
            self.add_triple(person_id, "is_a", "person")
            
            # Add visit count as a property
            visit_count = self.user_data[person_id]['visit_count']
            self.add_triple(person_id, "has_visit_count", f"visit_{visit_count}")
            
        elif ":" in interaction_data:
            # Parse attribute data (e.g., "name:John", "age:25", "profession:engineer")
            key, value = interaction_data.split(":", 1)
            self.user_data[person_id]['attributes'][key] = value
            
            # Add attribute triples
            self.add_triple(person_id, f"has_{key}", value.lower().replace(" ", "_"))
            
            # Add profession-specific triples
            if key == "profession":
                self.add_triple(value.lower().replace(" ", "_"), "is_a", "profession")
                self.add_triple("profession", "category_of", "work")
            
            # Add age-related triples
            if key == "age":
                try:
                    age_num = int(value)
                    if age_num < 25:
                        age_group = "young_adult"
                    elif age_num < 45:
                        age_group = "middle_aged"
                    elif age_num < 65:
                        age_group = "mature_adult"
                    else:
                        age_group = "senior"
                    
                    self.add_triple(person_id, "belongs_to", age_group)
                    self.add_triple(age_group, "is_a", "age_group")
                except ValueError:
                    pass
        
        # Add to interaction history
        if person_id not in self.interaction_history:
            self.interaction_history[person_id] = []
        
        self.interaction_history[person_id].append({
            'timestamp': timestamp,
            'interaction': interaction_data
        })
        
        # Save data
        self.save_data()
    
    def prepare_training_data(self):
        """Prepare triples for PyKEEN training."""
        if len(self.triples) < 5:  # Reduced minimum threshold
            rospy.logwarn(f"Not enough triples for training ({len(self.triples)} < 5)")
            return None
        
        try:
            # Convert triples to DataFrame
            df = pd.DataFrame(self.triples, columns=['subject', 'predicate', 'object'])
            
            # Remove any duplicate triples
            df = df.drop_duplicates()
            
            rospy.loginfo(f"Preparing {len(df)} unique triples for training")
            
            # Create triples factory with proper configuration
            self.triples_factory = TriplesFactory.from_labeled_triples(
                triples=df.values,
                create_inverse_triples=False,  # Start without inverse triples for simplicity
                entity_to_id=None,  # Let PyKEEN create mappings
                relation_to_id=None
            )
            
            # Store mappings
            self.entity_to_id = self.triples_factory.entity_to_id
            self.relation_to_id = self.triples_factory.relation_to_id
            self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
            self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
            
            rospy.loginfo(f"Created triples factory with {self.triples_factory.num_entities} entities and {self.triples_factory.num_relations} relations")
            
            return self.triples_factory
            
        except Exception as e:
            rospy.logerr(f"Error preparing training data: {e}")
            import traceback
            rospy.logerr(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def train_model(self):
        """Train the PyKEEN model on current triples."""
        try:
            triples_factory = self.prepare_training_data()
            if triples_factory is None:
                return
            
            rospy.loginfo(f"Training {self.model_type} model with {len(self.triples)} triples...")
            
            # Configure model based on type
            if self.model_type == "TransE":
                model_class = TransE
            elif self.model_type == "ComplEx":
                model_class = ComplEx
            elif self.model_type == "RotatE":
                model_class = RotatE
            else:
                model_class = TransE  # Default
            
            # Split triples for training/validation (80/20 split)
            training_factory, testing_factory = triples_factory.split([0.8, 0.2])
            
            # Train model using pipeline with proper training/testing split
            result = pipeline(
                training=training_factory,
                testing=testing_factory,
                model=model_class,
                model_kwargs=dict(embedding_dim=self.embedding_dim),
                training_kwargs=dict(
                    num_epochs=100,
                    batch_size=min(32, len(training_factory.mapped_triples)),  # Adjust batch size
                ),
                evaluation_kwargs=dict(
                    batch_size=min(64, len(testing_factory.mapped_triples)),
                ),
                random_seed=42,
                device='cpu',  # Force CPU usage for compatibility
            )
            
            self.model = result.model
            self.triples_factory = training_factory  # Store training factory
            
            # Save trained model
            self.save_trained_model()
            
            rospy.loginfo("Model training completed successfully")
            rospy.loginfo(f"Training results: {result.metric_results}")
            
        except Exception as e:
            # Fallback: try simpler training approach
            self.train_model_simple()
    
    def train_model_simple(self):
        """Simplified training approach when main pipeline fails."""
        try:
            rospy.loginfo("Attempting simplified training approach...")
            
            triples_factory = self.prepare_training_data()
            if triples_factory is None:
                return
            
            # Create model directly
            if self.model_type == "TransE":
                model = TransE(
                    triples_factory=triples_factory,
                    embedding_dim=self.embedding_dim,
                    random_seed=42
                )
            elif self.model_type == "ComplEx":
                model = ComplEx(
                    triples_factory=triples_factory,
                    embedding_dim=self.embedding_dim,
                    random_seed=42
                )
            else:
                model = TransE(  # Default fallback
                    triples_factory=triples_factory,
                    embedding_dim=self.embedding_dim,
                    random_seed=42
                )
            
            # Simple training loop
            training_loop = SLCWATrainingLoop(
                model=model,
                triples_factory=triples_factory,
                optimizer='Adam',
                optimizer_kwargs=dict(lr=0.01),
            )
            
            # Train with fewer epochs for stability
            training_loop.train(
                triples_factory=triples_factory,
                num_epochs=50,
                batch_size=min(16, len(triples_factory.mapped_triples)),
                use_tqdm=False  # Disable progress bar for cleaner logs
            )
            
            self.model = model
            self.save_trained_model()
            
            rospy.loginfo("Simplified model training completed successfully")
            
        except Exception as e:
            rospy.logerr(f"Error in simplified training: {e}")
            rospy.logwarn("Using fallback embeddings without PyKEEN training")
            import traceback
            rospy.logerr(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback: try simpler training approach
            self.train_model_simple()
    
    def save_trained_model(self):
        """Save the trained PyKEEN model."""
        if self.model is not None:
            try:
                model_path = os.path.join(self.data_path, "models", "pykeen_model.pkl")
                torch.save(self.model.state_dict(), model_path)
                
                # Save mappings
                mappings_path = os.path.join(self.data_path, "models", "mappings.json")
                mappings = {
                    'entity_to_id': self.entity_to_id,
                    'relation_to_id': self.relation_to_id,
                    'embedding_dim': self.embedding_dim,
                    'model_type': self.model_type
                }
                with open(mappings_path, 'w') as f:
                    json.dump(mappings, f, indent=2)
                
                rospy.loginfo("Trained model saved successfully")
                
            except Exception as e:
                rospy.logerr(f"Error saving trained model: {e}")
    
    def load_trained_model(self):
        """Load a previously trained PyKEEN model."""
        try:
            model_path = os.path.join(self.data_path, "models", "pykeen_model.pkl")
            mappings_path = os.path.join(self.data_path, "models", "mappings.json")
            
            if os.path.exists(model_path) and os.path.exists(mappings_path):
                # Load mappings
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                    self.entity_to_id = mappings['entity_to_id']
                    self.relation_to_id = mappings['relation_to_id']
                    self.embedding_dim = mappings.get('embedding_dim', 64)
                    self.model_type = mappings.get('model_type', 'TransE')
                
                self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
                self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
                
                # Recreate triples factory
                if self.triples:
                    self.prepare_training_data()
                
                # Load model
                if self.model_type == "TransE":
                    model_class = TransE
                elif self.model_type == "ComplEx":
                    model_class = ComplEx
                elif self.model_type == "RotatE":
                    model_class = RotatE
                else:
                    model_class = TransE
                
                if self.triples_factory:
                    self.model = model_class(
                        triples_factory=self.triples_factory,
                        embedding_dim=self.embedding_dim
                    )
                    self.model.load_state_dict(torch.load(model_path))
                    self.model.eval()
                
                rospy.loginfo("Trained model loaded successfully")
                
        except Exception as e:
            rospy.logwarn(f"Could not load trained model: {e}")
    
    def get_entity_embedding(self, person_id):
        """
        Get PyKEEN embedding for a person/entity.
        
        Args:
            person_id (str): Entity identifier
            
        Returns:
            numpy.ndarray: Entity embedding vector
        """
        if self.model is None:
            rospy.logwarn("No trained model available")
            return self._generate_fallback_embedding(person_id)
        
        try:
            if person_id in self.entity_to_id:
                entity_id = self.entity_to_id[person_id]
                
                # Get embedding from model
                with torch.no_grad():
                    embedding = self.model.entity_representations[0](
                        torch.tensor([entity_id])
                    ).numpy()[0]
                
                return embedding
            else:
                rospy.logwarn(f"Entity {person_id} not found in knowledge graph")
                return self._generate_fallback_embedding(person_id)
                
        except Exception as e:
            rospy.logerr(f"Error getting entity embedding: {e}")
            return self._generate_fallback_embedding(person_id)
    
    def _generate_fallback_embedding(self, person_id):
        """Generate a fallback embedding when PyKEEN is not available."""
        # Use the old method as fallback
        user_info = self.user_data.get(person_id, {})
        
        embedding_features = []
        
        # Visit frequency
        visit_count = user_info.get('visit_count', 0)
        embedding_features.append(min(visit_count / 10.0, 1.0))
        
        # User attributes
        attributes = user_info.get('attributes', {})
        
        # Age encoding
        try:
            age = int(attributes.get('age', 30))
            embedding_features.append(age / 100.0)
        except ValueError:
            embedding_features.append(0.3)
        
        # Profession encoding
        profession = attributes.get('profession', 'unknown').lower()
        profession_hash = hash(profession) % 100 / 100.0
        embedding_features.append(profession_hash)
        
        # Pad to embedding dimension
        while len(embedding_features) < self.embedding_dim:
            embedding_features.append(0.0)
        
        return np.array(embedding_features[:self.embedding_dim], dtype=np.float32)
    
    def predict_relations(self, person_id, top_k=5):
        """
        Predict likely relations for a person using the trained model.
        
        Args:
            person_id (str): Person identifier
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of predicted relations and objects
        """
        if self.model is None or person_id not in self.entity_to_id:
            return []
        
        try:
            entity_id = self.entity_to_id[person_id]
            predictions = []
            
            # For each relation, predict most likely objects
            for relation_name, relation_id in self.relation_to_id.items():
                # Create batch of triples to score
                batch_entities = torch.tensor([entity_id] * len(self.entity_to_id))
                batch_relations = torch.tensor([relation_id] * len(self.entity_to_id))
                batch_objects = torch.tensor(list(range(len(self.entity_to_id))))
                
                # Score triples
                with torch.no_grad():
                    scores = self.model.score_hrt(
                        torch.stack([batch_entities, batch_relations, batch_objects], dim=1)
                    )
                
                # Get top predictions
                top_indices = torch.argsort(scores, descending=True)[:top_k]
                
                for idx in top_indices:
                    object_id = batch_objects[idx].item()
                    object_name = self.id_to_entity[object_id]
                    score = scores[idx].item()
                    
                    predictions.append({
                        'relation': relation_name,
                        'object': object_name,
                        'score': score
                    })
            
            # Sort by score and return top_k
            predictions.sort(key=lambda x: x['score'], reverse=True)
            return predictions[:top_k]
            
        except Exception as e:
            rospy.logerr(f"Error predicting relations: {e}")
            return []
    
    def get_user_context(self, person_id):
        """
        Get rich context about a user using knowledge graph.
        
        Args:
            person_id (str): Person identifier
            
        Returns:
            dict: Rich user context including embeddings and predictions
        """
        context = {
            'user_data': self.user_data.get(person_id, {}),
            'interaction_history': self.interaction_history.get(person_id, []),
            'embedding': self.get_entity_embedding(person_id).tolist(),
            'predicted_relations': self.predict_relations(person_id),
            'knowledge_graph_stats': {
                'total_triples': len(self.triples),
                'total_entities': len(self.entities),
                'total_relations': len(self.relations),
                'model_trained': self.model is not None
            }
        }
        
        return context
    
    # Compatibility methods for existing code
    def update(self, person_id, interaction_data):
        """Compatibility method for existing code."""
        self.update_user_data(person_id, interaction_data)
    
    def get_embedding(self, person_id):
        """Compatibility method for existing code."""
        return self.get_entity_embedding(person_id)
    
    def get_user_info(self, person_id):
        """Compatibility method for existing code."""
        return self.user_data.get(person_id, {})


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the PyKEEN integration
    
    print("🧪 Testing PyKEEN Integration")
    print("=" * 40)
    
    # Initialize the knowledge graph manager
    kg = PyKEENGraphManager()
    
    # Add some example user interactions
    print("📝 Adding sample user interactions...")
    
    # User 1: Alice, software engineer
    kg.update_user_data("alice", "visited_fortune_teller")
    kg.update_user_data("alice", "name:Alice")
    kg.update_user_data("alice", "age:28")
    kg.update_user_data("alice", "profession:software engineer")
    
    # User 2: Bob, teacher
    kg.update_user_data("bob", "visited_fortune_teller")
    kg.update_user_data("bob", "name:Bob")
    kg.update_user_data("bob", "age:35")
    kg.update_user_data("bob", "profession:teacher")
    
    # User 3: Carol, artist
    kg.update_user_data("carol", "visited_fortune_teller")
    kg.update_user_data("carol", "name:Carol")
    kg.update_user_data("carol", "age:42")
    kg.update_user_data("carol", "profession:artist")
    
    print(f"✅ Added {len(kg.triples)} triples to knowledge graph")
    
    # Get user embeddings
    print("\n🔢 Getting user embeddings...")
    for user in ["alice", "bob", "carol"]:
        embedding = kg.get_entity_embedding(user)
        print(f"{user}: {embedding.shape} - {embedding[:5]}...")
    
    # Get user context
    print("\n📊 Getting rich user context...")
    alice_context = kg.get_user_context("alice")
    print(f"Alice context keys: {list(alice_context.keys())}")
    
    # Predict relations
    print("\n🔮 Predicting relations...")
    predictions = kg.predict_relations("alice", top_k=3)
    for pred in predictions:
        print(f"  {pred['relation']} -> {pred['object']} (score: {pred['score']:.3f})")
    
    print("\n✅ PyKEEN integration test completed!")
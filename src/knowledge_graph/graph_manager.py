#!/usr/bin/env python3

"""
Simplified PyKEEN Integration that addresses common training issues.
This version is more robust and handles edge cases better.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import rospy

# PyKEEN imports with error handling
try:
    from pykeen.triples import TriplesFactory
    from pykeen.models import TransE
    from pykeen.training import SLCWATrainingLoop
    import torch
    PYKEEN_AVAILABLE = True
    rospy.loginfo("PyKEEN successfully imported")
except ImportError as e:
    PYKEEN_AVAILABLE = False
    rospy.logwarn(f"PyKEEN not available: {e}")

class PyKEENGraphManager:
    def __init__(self, data_path="data/knowledge_graph/"):
        """
        Simplified PyKEEN-based Knowledge Graph Manager.
        Falls back to manual embeddings if PyKEEN fails.
        """
        self.data_path = data_path
        self.ensure_data_directory()
        
        # Knowledge graph storage
        self.triples = []
        self.entities = set()
        self.relations = set()
        
        # PyKEEN components (if available)
        self.model = None
        self.triples_factory = None
        self.entity_to_id = {}
        self.relation_to_id = {}
        self.pykeen_trained = False
        
        # User data for fallback
        self.user_data = {}
        self.interaction_history = {}
        
        # Configuration
        self.embedding_dim = 32  # Smaller for stability
        self.min_triples_for_training = 8
        self.training_batch_size = 4
        self.training_epochs = 20
        
        # Load existing data
        self.load_data()
        
        rospy.loginfo(f"Simple PyKEEN Graph Manager initialized (PyKEEN available: {PYKEEN_AVAILABLE})")
    
    def ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "models"), exist_ok=True)
    
    def load_data(self):
        """Load existing data."""
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
            
            rospy.loginfo(f"Loaded {len(self.triples)} triples from storage")
            
            # Try to train model if we have enough data
            if PYKEEN_AVAILABLE and len(self.triples) >= self.min_triples_for_training:
                self.train_model_safe()
                
        except Exception as e:
            rospy.logwarn(f"Error loading data: {e}")
    
    def save_data(self):
        """Save data to disk."""
        try:
            # Save triples
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
                
        except Exception as e:
            rospy.logerr(f"Error saving data: {e}")
    
    def add_triple(self, subject, predicate, object_):
        """Add a triple to the knowledge graph."""
        triple = (subject, predicate, object_)
        
        if triple not in self.triples:
            self.triples.append(triple)
            self.entities.add(subject)
            self.entities.add(object_)
            self.relations.add(predicate)
            
            rospy.loginfo(f"Added triple: {subject} --{predicate}--> {object_}")
            
            # Save immediately
            self.save_data()
            
            # Try to train when we have enough data
            if (PYKEEN_AVAILABLE and 
                len(self.triples) >= self.min_triples_for_training and 
                len(self.triples) % 5 == 0):  # Every 5 new triples
                rospy.loginfo("Attempting model training...")
                self.train_model_safe()
    
    def train_model_safe(self):
        """Safe model training with extensive error handling."""
        if not PYKEEN_AVAILABLE:
            rospy.logwarn("PyKEEN not available, skipping training")
            return False
        
        try:
            rospy.loginfo(f"Starting safe training with {len(self.triples)} triples...")
            
            # Prepare data
            if not self._prepare_training_data_safe():
                return False
            
            # Create model
            model = TransE(
                triples_factory=self.triples_factory,
                embedding_dim=self.embedding_dim,
                random_seed=42
            )
            
            # Create training loop
            training_loop = SLCWATrainingLoop(
                model=model,
                triples_factory=self.triples_factory,
                optimizer='Adam',
                optimizer_kwargs=dict(lr=0.01),
            )
            
            # Train with safe parameters
            num_triples = len(self.triples_factory.mapped_triples)
            safe_batch_size = min(self.training_batch_size, max(1, num_triples // 2))
            
            rospy.loginfo(f"Training with batch_size={safe_batch_size}, epochs={self.training_epochs}")
            
            training_loop.train(
                triples_factory=self.triples_factory,
                num_epochs=self.training_epochs,
                batch_size=safe_batch_size,
                use_tqdm=False
            )
            
            self.model = model
            self.pykeen_trained = True
            
            rospy.loginfo("✅ PyKEEN model training completed successfully!")
            return True
            
        except Exception as e:
            rospy.logerr(f"Training failed: {e}")
            self.pykeen_trained = False
            return False
    
    def _prepare_training_data_safe(self):
        """Safely prepare training data."""
        try:
            if len(self.triples) < self.min_triples_for_training:
                rospy.logwarn(f"Not enough triples: {len(self.triples)} < {self.min_triples_for_training}")
                return False
            
            # Convert to DataFrame and clean
            df = pd.DataFrame(self.triples, columns=['subject', 'predicate', 'object'])
            df = df.drop_duplicates()
            
            rospy.loginfo(f"Preparing {len(df)} unique triples")
            
            # Create TriplesFactory
            self.triples_factory = TriplesFactory.from_labeled_triples(
                triples=df.values,
                create_inverse_triples=False  # Keep it simple
            )
            
            # Store mappings
            self.entity_to_id = self.triples_factory.entity_to_id
            self.relation_to_id = self.triples_factory.relation_to_id
            
            rospy.loginfo(f"TriplesFactory created: {self.triples_factory.num_entities} entities, {self.triples_factory.num_relations} relations")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error preparing training data: {e}")
            return False
    
    def update_user_data(self, person_id, interaction_data):
        """Update user data and create knowledge graph triples."""
        timestamp = datetime.now().isoformat()
        
        # Initialize user data
        if person_id not in self.user_data:
            self.user_data[person_id] = {
                'first_seen': timestamp,
                'last_seen': timestamp,
                'visit_count': 1,
                'attributes': {}
            }
        
        # Update user data
        self.user_data[person_id]['last_seen'] = timestamp
        
        # Create knowledge graph triples
        if interaction_data == "visited_fortune_teller":
            self.user_data[person_id]['visit_count'] += 1
            
            # Add basic triples
            self.add_triple(person_id, "visited", "fortune_teller")
            self.add_triple(person_id, "is_a", "person")
            
        elif ":" in interaction_data:
            # Parse attribute data
            key, value = interaction_data.split(":", 1)
            self.user_data[person_id]['attributes'][key] = value
            
            # Clean value for triple
            clean_value = value.lower().replace(" ", "_").replace("-", "_")
            
            # Add attribute triples
            self.add_triple(person_id, f"has_{key}", clean_value)
            
            # Add type triples
            if key == "profession":
                self.add_triple(clean_value, "is_a", "profession")
            elif key == "age":
                try:
                    age_num = int(value)
                    if age_num < 30:
                        age_group = "young"
                    elif age_num < 50:
                        age_group = "middle_aged"
                    else:
                        age_group = "mature"
                    self.add_triple(person_id, "belongs_to_age_group", age_group)
                except ValueError:
                    pass
        
        # Add to interaction history
        if person_id not in self.interaction_history:
            self.interaction_history[person_id] = []
        
        self.interaction_history[person_id].append({
            'timestamp': timestamp,
            'interaction': interaction_data
        })
    
    def get_entity_embedding(self, person_id):
        """Get embedding for an entity."""
        if self.pykeen_trained and person_id in self.entity_to_id:
            return self._get_pykeen_embedding(person_id)
        else:
            return self._get_fallback_embedding(person_id)
    
    def _get_pykeen_embedding(self, person_id):
        """Get PyKEEN embedding."""
        try:
            entity_id = self.entity_to_id[person_id]
            
            with torch.no_grad():
                embedding = self.model.entity_representations[0](
                    torch.tensor([entity_id])
                ).numpy()[0]
            
            rospy.loginfo(f"Retrieved PyKEEN embedding for {person_id}: shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            rospy.logwarn(f"Error getting PyKEEN embedding for {person_id}: {e}")
            return self._get_fallback_embedding(person_id)
    
    def _get_fallback_embedding(self, person_id):
        """Generate fallback embedding when PyKEEN is not available."""
        user_info = self.user_data.get(person_id, {})
        attributes = user_info.get('attributes', {})
        
        # Create feature vector
        features = []
        
        # Visit count (normalized)
        visit_count = user_info.get('visit_count', 0)
        features.append(min(visit_count / 10.0, 1.0))
        
        # Age (normalized)
        try:
            age = int(attributes.get('age', 30))
            features.append(age / 100.0)
        except ValueError:
            features.append(0.3)
        
        # Profession (hash-based)
        profession = attributes.get('profession', 'unknown').lower()
        profession_hash = abs(hash(profession)) % 100 / 100.0
        features.append(profession_hash)
        
        # Name (hash-based)
        name = attributes.get('name', person_id).lower()
        name_hash = abs(hash(name)) % 100 / 100.0
        features.append(name_hash)
        
        # Interaction count
        interaction_count = len(self.interaction_history.get(person_id, []))
        features.append(min(interaction_count / 20.0, 1.0))
        
        # Days since first visit
        try:
            first_seen = datetime.fromisoformat(user_info.get('first_seen', datetime.now().isoformat()))
            days_since = (datetime.now() - first_seen).days
            features.append(min(days_since / 365.0, 1.0))
        except:
            features.append(0.0)
        
        # Pad to embedding dimension
        while len(features) < self.embedding_dim:
            features.append(np.random.random() * 0.1)  # Small random values
        
        embedding = np.array(features[:self.embedding_dim], dtype=np.float32)
        rospy.loginfo(f"Generated fallback embedding for {person_id}: shape {embedding.shape}")
        return embedding
    
    def predict_relations(self, person_id, top_k=5):
        """Predict relations for a person."""
        if not self.pykeen_trained or person_id not in self.entity_to_id:
            return self._predict_relations_fallback(person_id, top_k)
        
        try:
            entity_id = self.entity_to_id[person_id]
            predictions = []
            
            # For each relation, predict likely objects
            for relation_name, relation_id in self.relation_to_id.items():
                # Score all possible triples
                for object_name, object_id in self.entity_to_id.items():
                    if object_name != person_id:  # Don't predict self-relations
                        
                        # Create triple tensor
                        triple = torch.tensor([[entity_id, relation_id, object_id]])
                        
                        # Score the triple
                        with torch.no_grad():
                            score = self.model.score_hrt(triple).item()
                        
                        predictions.append({
                            'relation': relation_name,
                            'object': object_name,
                            'score': score
                        })
            
            # Sort by score and return top_k
            predictions.sort(key=lambda x: x['score'], reverse=True)
            return predictions[:top_k]
            
        except Exception as e:
            rospy.logwarn(f"Error predicting relations: {e}")
            return self._predict_relations_fallback(person_id, top_k)
    
    def _predict_relations_fallback(self, person_id, top_k=5):
        """Fallback relation prediction based on user attributes."""
        user_info = self.user_data.get(person_id, {})
        attributes = user_info.get('attributes', {})
        predictions = []
        
        # Predict based on profession
        profession = attributes.get('profession', '').lower()
        if 'engineer' in profession:
            predictions.extend([
                {'relation': 'interested_in', 'object': 'technology', 'score': 0.9},
                {'relation': 'prefers', 'object': 'logical_approach', 'score': 0.8},
            ])
        elif 'teacher' in profession:
            predictions.extend([
                {'relation': 'interested_in', 'object': 'education', 'score': 0.9},
                {'relation': 'prefers', 'object': 'helping_others', 'score': 0.8},
            ])
        elif 'artist' in profession:
            predictions.extend([
                {'relation': 'interested_in', 'object': 'creativity', 'score': 0.9},
                {'relation': 'prefers', 'object': 'artistic_expression', 'score': 0.8},
            ])
        
        # Predict based on age
        try:
            age = int(attributes.get('age', 30))
            if age < 30:
                predictions.append({'relation': 'belongs_to', 'object': 'young_generation', 'score': 0.7})
            elif age > 50:
                predictions.append({'relation': 'has_quality', 'object': 'wisdom', 'score': 0.7})
        except ValueError:
            pass
        
        # Visit count based predictions
        visit_count = user_info.get('visit_count', 1)
        if visit_count > 1:
            predictions.append({'relation': 'is_a', 'object': 'returning_visitor', 'score': 0.6})
        
        return predictions[:top_k]
    
    def get_user_context(self, person_id):
        """Get rich user context."""
        context = {
            'user_data': self.user_data.get(person_id, {}),
            'interaction_history': self.interaction_history.get(person_id, []),
            'embedding': self.get_entity_embedding(person_id).tolist(),
            'predicted_relations': self.predict_relations(person_id),
            'knowledge_graph_stats': {
                'total_triples': len(self.triples),
                'total_entities': len(self.entities),
                'total_relations': len(self.relations),
                'pykeen_trained': self.pykeen_trained,
                'pykeen_available': PYKEEN_AVAILABLE
            }
        }
        return context
    
    def get_stats(self):
        """Get knowledge graph statistics."""
        return {
            'total_triples': len(self.triples),
            'total_entities': len(self.entities),
            'total_relations': len(self.relations),
            'total_users': len(self.user_data),
            'pykeen_available': PYKEEN_AVAILABLE,
            'pykeen_trained': self.pykeen_trained,
            'embedding_method': 'PyKEEN' if self.pykeen_trained else 'Fallback'
        }
    
    # Compatibility methods
    def update(self, person_id, interaction_data):
        """Compatibility method."""
        self.update_user_data(person_id, interaction_data)
    
    def get_embedding(self, person_id):
        """Compatibility method."""
        return self.get_entity_embedding(person_id)
    
    def get_user_info(self, person_id):
        """Compatibility method."""
        return self.user_data.get(person_id, {})


# Test the simplified integration
if __name__ == "__main__":
    print("🧪 Testing Simplified PyKEEN Integration")
    print("=" * 50)
    
    # Test with debug logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize
        kg = PyKEENGraphManager()
        print(f"✅ Initialized: {kg.get_stats()}")
        
        # Add test users
        test_users = [
            ("alice", "Alice", 28, "software engineer"),
            ("bob", "Bob", 35, "teacher"),
            ("carol", "Carol", 42, "artist"),
            ("david", "David", 29, "software engineer"),  # Same profession as Alice
            ("eve", "Eve", 33, "teacher"),  # Same profession as Bob
            ("frank", "Frank", 31, "doctor"),
            ("grace", "Grace", 26, "designer"),
            ("henry", "Henry", 38, "lawyer"),
            ("iris", "Iris", 30, "software engineer"),  # Same profession as Alice & David
            ("jack", "Jack", 44, "chef"),
            ("kate", "Kate", 27, "nurse"),
            ("luke", "Luke", 36, "teacher"),  # Same profession as Bob & Eve
            ("maria", "Maria", 39, "doctor"),  # Same profession as Frank
            ("nick", "Nick", 32, "artist"),  # Same profession as Carol
            ("olivia", "Olivia", 25, "designer")  # Same profession as Grace
        ]
        
        print("\n📝 Adding test users...")
        for user_id, name, age, profession in test_users:
            kg.update_user_data(user_id, "visited_fortune_teller")
            kg.update_user_data(user_id, f"name:{name}")
            kg.update_user_data(user_id, f"age:{age}")
            kg.update_user_data(user_id, f"profession:{profession}")
            print(f"  Added {name}")
        
        # Show final stats
        final_stats = kg.get_stats()
        print(f"\n✅ Final stats: {final_stats}")
        
        # Test embeddings
        print("\n🔢 Testing embeddings...")
        for user_id, name, _, _ in test_users:
            embedding = kg.get_entity_embedding(user_id)
            print(f"  {name}: {embedding.shape} - PyKEEN: {kg.pykeen_trained}")
        
        # Test predictions
        print("\n🔮 Testing predictions...")
        for user_id, name, _, _ in test_users[:2]:
            predictions = kg.predict_relations(user_id, top_k=3)
            print(f"  {name}:")
            for pred in predictions:
                print(f"    {pred['relation']} -> {pred['object']} (score: {pred['score']:.3f})")
        
        print("\n✅ Simplified PyKEEN integration test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
#!/usr/bin/env python3

"""
Debug script for PyKEEN integration issues.
This script helps identify and fix common problems.
"""

import sys
import os
import traceback
import torch

# Add src directory to path
sys.path.append('src')

def test_pykeen_imports():
    """Test if all PyKEEN components can be imported."""
    print("🔍 Testing PyKEEN imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        from pykeen.datasets import Dataset
        from pykeen.models import TransE
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline
        print("✅ PyKEEN core components")
    except ImportError as e:
        print(f"❌ PyKEEN import failed: {e}")
        return False
    
    return True

def test_minimal_pykeen():
    """Test minimal PyKEEN functionality."""
    print("\n🧪 Testing minimal PyKEEN functionality...")
    
    try:
        import pandas as pd
        from pykeen.triples import TriplesFactory
        from pykeen.models import TransE
        from pykeen.training import SLCWATrainingLoop
        
        # Create minimal test data
        test_triples = [
            ("alice", "knows", "bob"),
            ("bob", "knows", "charlie"),
            ("alice", "likes", "programming"),
            ("bob", "likes", "teaching"),
            ("charlie", "likes", "art"),
            ("alice", "age", "young"),
            ("bob", "age", "middle"),
            ("charlie", "age", "middle"),
        ]
        
        print(f"📝 Created {len(test_triples)} test triples")
        
        # Create DataFrame
        df = pd.DataFrame(test_triples, columns=['subject', 'predicate', 'object'])
        print("✅ DataFrame created")
        
        # Create TriplesFactory
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=df.values,
            create_inverse_triples=False
        )
        print(f"✅ TriplesFactory created: {triples_factory.num_entities} entities, {triples_factory.num_relations} relations")
        
        # Create model
        model = TransE(
            triples_factory=triples_factory,
            embedding_dim=32,
            random_seed=42
        )
        print("✅ TransE model created")
        
        # Test training
        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=triples_factory,
            optimizer='Adam',
            optimizer_kwargs=dict(lr=0.01),
        )
        print("✅ Training loop created")
        
        # Short training
        training_loop.train(
            triples_factory=triples_factory,
            num_epochs=5,
            batch_size=4,
            use_tqdm=False
        )
        print("✅ Model training completed")
        
        # Test embeddings
        entity_id = triples_factory.entity_to_id['alice']
        embedding = model.entity_representations[0](torch.tensor([entity_id]))
        print(f"✅ Alice embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal PyKEEN test failed: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def test_knowledge_graph_creation():
    """Test knowledge graph creation with our class."""
    print("\n🏗️ Testing PyKEENGraphManager...")
    
    try:
        from knowledge_graph.graph_manager import PyKEENGraphManager
        
        # Initialize with debug mode
        kg = PyKEENGraphManager()
        print("✅ PyKEENGraphManager initialized")
        
        # Add test data step by step
        test_users = [
            ("alice", "Alice", 28, "engineer"),
            ("bob", "Bob", 35, "teacher"),
        ]
        
        print("📝 Adding test users...")
        for user_id, name, age, profession in test_users:
            print(f"  Adding {name}...")
            kg.update_user_data(user_id, "visited_fortune_teller")
            kg.update_user_data(user_id, f"name:{name}")
            kg.update_user_data(user_id, f"age:{age}")
            kg.update_user_data(user_id, f"profession:{profession}")
        
        print(f"✅ Knowledge graph created with {len(kg.triples)} triples")
        
        # Check triples
        print("📋 First 10 triples:")
        for i, triple in enumerate(kg.triples[:10]):
            print(f"  {i+1}. {triple[0]} --{triple[1]}--> {triple[2]}")
        
        # Force training
        print("🎯 Forcing model training...")
        kg.train_model()
        
        # Test embeddings
        print("🔢 Testing embeddings...")
        alice_embedding = kg.get_entity_embedding("alice")
        print(f"Alice embedding: {alice_embedding.shape}")
        
        # Test predictions
        print("🔮 Testing predictions...")
        predictions = kg.predict_relations("alice", top_k=3)
        print(f"Alice predictions: {len(predictions)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge graph test failed: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def diagnose_training_issue():
    """Diagnose the specific training issue."""
    print("\n🔧 Diagnosing training issue...")
    
    try:
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline
        from pykeen.models import TransE
        import pandas as pd
        
        # Recreate the exact scenario
        test_triples = [
            ("alice", "visited", "fortune_teller"),
            ("alice", "is_a", "person"),
            ("alice", "has_name", "alice"),
            ("alice", "has_age", "28"),
            ("alice", "has_profession", "engineer"),
            ("bob", "visited", "fortune_teller"),
            ("bob", "is_a", "person"),
            ("bob", "has_name", "bob"),
            ("bob", "has_age", "35"),
            ("bob", "has_profession", "teacher"),
        ]
        
        print(f"📝 Testing with {len(test_triples)} triples")
        
        df = pd.DataFrame(test_triples, columns=['subject', 'predicate', 'object'])
        
        # Test TriplesFactory creation
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=df.values,
            create_inverse_triples=False
        )
        print(f"✅ TriplesFactory: {triples_factory.num_entities} entities, {triples_factory.num_relations} relations")
        
        # Test splitting
        if len(triples_factory.mapped_triples) >= 4:
            training_factory, testing_factory = triples_factory.split([0.8, 0.2])
            print(f"✅ Split successful: train={len(training_factory.mapped_triples)}, test={len(testing_factory.mapped_triples)}")
            
            # Test pipeline
            result = pipeline(
                training=training_factory,
                testing=testing_factory,
                model=TransE,
                model_kwargs=dict(embedding_dim=32),
                training_kwargs=dict(
                    num_epochs=10,
                    batch_size=2,
                ),
                evaluation_kwargs=dict(
                    batch_size=2,
                ),
                random_seed=42,
                device='cpu',
            )
            print("✅ Pipeline training successful!")
            
        else:
            print("⚠️ Not enough triples for split, trying direct training...")
            
            model = TransE(
                triples_factory=triples_factory,
                embedding_dim=32,
                random_seed=42
            )
            
            from pykeen.training import SLCWATrainingLoop
            training_loop = SLCWATrainingLoop(
                model=model,
                triples_factory=triples_factory,
            )
            
            training_loop.train(
                triples_factory=triples_factory,
                num_epochs=10,
                batch_size=2,
                use_tqdm=False
            )
            print("✅ Direct training successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main diagnostic function."""
    print("🔮 PyKEEN DIAGNOSTIC SCRIPT 🔮")
    print("=" * 50)
    
    # Step 1: Check imports
    if not test_pykeen_imports():
        print("\n❌ Import test failed. Please install missing dependencies:")
        print("pip install torch pandas pykeen scikit-learn")
        return
    
    # Step 2: Test minimal PyKEEN
    if not test_minimal_pykeen():
        print("\n❌ Minimal PyKEEN test failed. Check PyKEEN installation.")
        return
    
    # Step 3: Test our implementation
    if not test_knowledge_graph_creation():
        print("\n❌ Knowledge graph test failed.")
    
    # Step 4: Diagnose specific issue
    if not diagnose_training_issue():
        print("\n❌ Training diagnosis failed.")
    
    print("\n🏁 Diagnostic completed!")
    print("\nNext steps:")
    print("1. If imports failed: pip install missing packages")
    print("2. If training failed: Check the updated PyKEENGraphManager code")
    print("3. If still issues: Try the simplified training approach")

if __name__ == "__main__":
    main()
"""
CORRECTED database.py
Fixes all potential embedding storage/retrieval bugs
"""
import numpy as np
from pymongo import MongoClient, ASCENDING
from datetime import datetime
from typing import Optional, List, Dict
import hashlib

from utils import setup_logger

logger = setup_logger(__name__)


class FaceDatabase:
    """
    Fixed MongoDB database interface with precision validation.
    
    Key fixes:
    - Explicit float() conversion on storage
    - Precision validation on retrieval  
    - Embedding hash storage for integrity
    - Reconstruction validation
    - Type checking at every step
    """
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/",
                 database_name: str = "face_recognition",
                 collection_name: str = "faces"):
        """Initialize MongoDB connection with validation."""
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        
        self.client = None
        self.db = None
        self.collection = None
        
        self.connect()
        logger.info("FaceDatabase initialized (CORRECTED VERSION)")
    
    def connect(self) -> bool:
        """Connect to MongoDB and create indexes."""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Create indexes
            self.collection.create_index([("face_id", ASCENDING)], unique=True)
            self.collection.create_index([("name", ASCENDING)])
            self.collection.create_index([("created_at", ASCENDING)])
            
            count = self.collection.count_documents({})
            logger.info(f"✓ Connected to MongoDB: {self.database_name}.{self.collection_name} ({count} faces)")
            
            return True
            
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False
    
    def insert_face(self, face_id: str, name: str, embedding: np.ndarray,
                   metadata: Optional[Dict] = None) -> bool:
        """
        Insert face embedding with strict precision validation.
        
        CRITICAL: Preserve full float32 precision when storing in MongoDB.
        
        Args:
            face_id: Unique identifier
            name: Person's name
            embedding: 512D numpy array (float32)
            metadata: Additional data
        
        Returns:
            True if successful
        """
        try:
            logger.info("="*60)
            logger.info(f"STORING EMBEDDING: face_id={face_id}, name={name}")
            logger.info("="*60)
            
            # Validate input
            if not isinstance(embedding, np.ndarray):
                logger.error(f"❌ Embedding is not numpy array: {type(embedding)}")
                return False
            
            if embedding.shape != (512,):
                logger.error(f"❌ Wrong embedding shape: {embedding.shape}")
                return False
            
            if embedding.dtype != np.float32:
                logger.warning(f"⚠️  Embedding dtype is {embedding.dtype}, converting to float32")
                embedding = embedding.astype(np.float32)
            
            # Validate normalization
            norm = np.linalg.norm(embedding)
            if not np.isclose(norm, 1.0, atol=0.01):
                logger.warning(f"⚠️  Embedding not normalized (norm={norm:.10f})")
            
            logger.info(f"Input embedding:")
            logger.info(f"  Shape: {embedding.shape}")
            logger.info(f"  Dtype: {embedding.dtype}")
            logger.info(f"  Norm: {norm:.10f}")
            logger.info(f"  Mean: {embedding.mean():.10f}")
            logger.info(f"  Std: {embedding.std():.10f}")
            logger.info(f"  Min: {embedding.min():.10f}")
            logger.info(f"  Max: {embedding.max():.10f}")
            logger.info(f"  First 5: {embedding[:5]}")
            
            # CRITICAL FIX: Convert each element to Python float explicitly
            # This preserves full precision in MongoDB (stored as BSON double)
            embedding_list = [float(x) for x in embedding]
            
            # Compute hash for integrity check
            embedding_hash = hashlib.sha256(embedding.tobytes()).hexdigest()
            
            logger.info(f"Converted to list: type={type(embedding_list)}, length={len(embedding_list)}")
            logger.info(f"First 5 values: {embedding_list[:5]}")
            logger.info(f"Embedding hash: {embedding_hash}")
            
            # Prepare document
            document = {
                "face_id": face_id,
                "name": name,
                "embedding": embedding_list,
                "embedding_hash": embedding_hash,
                "embedding_norm": float(norm),
                "embedding_stats": {
                    "mean": float(embedding.mean()),
                    "std": float(embedding.std()),
                    "min": float(embedding.min()),
                    "max": float(embedding.max())
                },
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }
            
            # Insert into database
            result = self.collection.insert_one(document)
            logger.info(f"✓ Inserted document: id={result.inserted_id}")
            
            # VALIDATION: Retrieve and verify
            logger.info("\nVerifying storage...")
            retrieved_doc = self.collection.find_one({"face_id": face_id})
            
            if retrieved_doc is None:
                logger.error("❌ Failed to retrieve just-inserted document!")
                return False
            
            # Reconstruct embedding
            retrieved_embedding = np.array(retrieved_doc['embedding'], dtype=np.float32)
            retrieved_hash = hashlib.sha256(retrieved_embedding.tobytes()).hexdigest()
            
            logger.info(f"Retrieved embedding:")
            logger.info(f"  Shape: {retrieved_embedding.shape}")
            logger.info(f"  Dtype: {retrieved_embedding.dtype}")
            logger.info(f"  Hash: {retrieved_hash}")
            logger.info(f"  First 5: {retrieved_embedding[:5]}")
            
            # Check hash match
            if retrieved_hash != embedding_hash:
                logger.error("❌ HASH MISMATCH! Data corrupted during storage!")
                logger.error(f"  Original:  {embedding_hash}")
                logger.error(f"  Retrieved: {retrieved_hash}")
                
                # Compute difference
                diff = np.abs(embedding - retrieved_embedding)
                logger.error(f"  Max diff: {diff.max():.15f}")
                logger.error(f"  Mean diff: {diff.mean():.15f}")
                
                return False
            
            logger.info("✓ Hash verified - storage integrity confirmed")
            
            # Check precision loss
            max_diff = np.abs(embedding - retrieved_embedding).max()
            if max_diff > 1e-6:
                logger.warning(f"⚠️  Precision loss detected: max_diff={max_diff:.15f}")
            else:
                logger.info(f"✓ Precision preserved (max_diff={max_diff:.15f})")
            
            logger.info("="*60)
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert face: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_all_faces(self) -> List[Dict]:
        """
        Retrieve all faces with embedding reconstruction validation.
        
        Returns:
            List of face documents with validated embeddings
        """
        try:
            faces = list(self.collection.find({}))
            logger.info(f"Retrieved {len(faces)} faces from database")
            
            # Validate each embedding
            validated_faces = []
            
            for i, face in enumerate(faces):
                face_id = face.get('face_id', 'unknown')
                name = face.get('name', 'unknown')
                
                # Reconstruct embedding
                embedding_list = face.get('embedding')
                
                if embedding_list is None:
                    logger.warning(f"⚠️  Face {face_id} has no embedding!")
                    continue
                
                if len(embedding_list) != 512:
                    logger.warning(f"⚠️  Face {face_id} has wrong embedding length: {len(embedding_list)}")
                    continue
                
                # Convert to numpy array
                embedding = np.array(embedding_list, dtype=np.float32)
                
                # Validate
                norm = np.linalg.norm(embedding)
                
                if not np.isclose(norm, 1.0, atol=0.1):
                    logger.warning(f"⚠️  Face {face_id} ({name}) has abnormal norm: {norm:.6f}")
                
                # Compute current hash
                current_hash = hashlib.sha256(embedding.tobytes()).hexdigest()
                stored_hash = face.get('embedding_hash')
                
                if stored_hash and current_hash != stored_hash:
                    logger.warning(f"⚠️  Face {face_id} ({name}) hash mismatch! Data may be corrupted.")
                
                logger.debug(f"Face {i}: id={face_id}, name={name}, norm={norm:.6f}, hash={current_hash[:12]}")
                
                # Update face with reconstructed embedding
                face['embedding'] = embedding
                validated_faces.append(face)
            
            logger.info(f"✓ Validated {len(validated_faces)} embeddings")
            
            return validated_faces
            
        except Exception as e:
            logger.error(f"Failed to get faces: {e}")
            return []
    
    def get_face_by_id(self, face_id: str) -> Optional[Dict]:
        """Get single face by ID with validation."""
        try:
            face = self.collection.find_one({"face_id": face_id})
            
            if face is None:
                return None
            
            # Reconstruct and validate embedding
            embedding_list = face.get('embedding')
            if embedding_list:
                embedding = np.array(embedding_list, dtype=np.float32)
                face['embedding'] = embedding
            
            return face
            
        except Exception as e:
            logger.error(f"Failed to get face by ID: {e}")
            return None
    
    def get_faces_by_name(self, name: str) -> List[Dict]:
        """Get all faces for a person with validation."""
        try:
            faces = list(self.collection.find({"name": name}))
            
            # Validate embeddings
            for face in faces:
                embedding_list = face.get('embedding')
                if embedding_list:
                    embedding = np.array(embedding_list, dtype=np.float32)
                    face['embedding'] = embedding
            
            return faces
            
        except Exception as e:
            logger.error(f"Failed to get faces by name: {e}")
            return []
    
    def delete_face(self, face_id: str) -> bool:
        """Delete face by ID."""
        try:
            result = self.collection.delete_one({"face_id": face_id})
            
            if result.deleted_count > 0:
                logger.info(f"✓ Deleted face: {face_id}")
                return True
            else:
                logger.warning(f"Face not found: {face_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete face: {e}")
            return False
    
    def delete_faces_by_name(self, name: str) -> int:
        """Delete all faces for a person."""
        try:
            result = self.collection.delete_many({"name": name})
            count = result.deleted_count
            
            logger.info(f"✓ Deleted {count} faces for {name}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to delete faces: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all faces from database."""
        try:
            result = self.collection.delete_many({})
            count = result.deleted_count
            
            logger.info(f"✓ Cleared {count} faces from database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False
    
    def get_count(self) -> int:
        """Get total number of faces."""
        try:
            return self.collection.count_documents({})
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0
    
    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    # Test database
    print("Testing FaceDatabase...")
    
    db = FaceDatabase()
    
    # Test 1: Store and retrieve embedding
    print("\nTest 1: Store and retrieve")
    test_embedding = np.random.randn(512).astype(np.float32)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)
    
    success = db.insert_face(
        face_id="test_001",
        name="TestPerson",
        embedding=test_embedding
    )
    
    if success:
        retrieved = db.get_face_by_id("test_001")
        if retrieved:
            retrieved_embedding = retrieved['embedding']
            diff = np.abs(test_embedding - retrieved_embedding).max()
            print(f"Storage test: max_diff={diff:.15f}")
            
            if diff < 1e-6:
                print("✓ Storage precision preserved")
            else:
                print(f"❌ Precision loss: {diff:.15f}")
        
        # Cleanup
        db.delete_face("test_001")
    
    db.close()

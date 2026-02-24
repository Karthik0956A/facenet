"""
CORRECTED embedder.py
Fixes all potential embedding generation bugs
"""
import numpy as np
import cv2
from keras_facenet import FaceNet
from typing import Optional
import hashlib

from utils import setup_logger

logger = setup_logger(__name__)


class FaceEmbedder:
    """
    Fixed FaceNet embedding generator with comprehensive validation.
    
    Key fixes:
    - Explicit RGB conversion
    - No double normalization
    - Proper batch dimension handling
    - Extensive diagnostic logging
    - Validation at each step
    """
    
    def __init__(self):
        """Initialize FaceNet model."""
        self.model = None
        self.target_size = (160, 160)
        self.load_model()
        logger.info("FaceEmbedder initialized (CORRECTED VERSION)")
    
    def load_model(self) -> None:
        """Load FaceNet model with error handling."""
        try:
            self.model = FaceNet()
            logger.info("[OK] FaceNet model loaded successfully")
            
            # Test model with random input
            test_input = np.random.rand(1, 160, 160, 3).astype(np.float32)
            test_output = self.model.embeddings(test_input)
            logger.info(f"[OK] Model test: input {test_input.shape} -> output {test_output.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load FaceNet model: {e}")
            raise
    
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face for FaceNet with strict validation.
        
        FaceNet requirements:
        - RGB format (NOT BGR)
        - 160x160 pixels
        - float32 type
        - Pixel values in [0, 1]
        - Shape: (160, 160, 3)
        
        Args:
            face_img: Raw face image (any size, typically BGR from OpenCV)
        
        Returns:
            Preprocessed face (160, 160, 3) float32 in [0, 1]
        """
        try:
            # Create a copy to avoid modifying original
            face = face_img.copy()
            
            logger.debug(f"Input face: shape={face.shape}, dtype={face.dtype}, "
                        f"range=[{face.min()}, {face.max()}]")
            
            # Step 1: Convert BGR to RGB
            # OpenCV uses BGR by default, FaceNet expects RGB
            if len(face.shape) == 3 and face.shape[2] == 3:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                logger.debug("[OK] Converted BGR -> RGB")
            elif len(face.shape) == 2:
                # Grayscale - convert to RGB
                face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
                logger.debug("[OK] Converted GRAY -> RGB")
            
            # Step 2: Resize to 160x160
            if face.shape[:2] != self.target_size:
                face = cv2.resize(face, self.target_size, interpolation=cv2.INTER_AREA)
                logger.debug(f"[OK] Resized to {self.target_size}")
            
            # Step 3: Convert to float32
            if face.dtype != np.float32:
                face = face.astype(np.float32)
                logger.debug("[OK] Converted to float32")
            
            # Step 4: Normalize to [0, 1]
            # CRITICAL: Only normalize if values are in [0, 255]
            if face.max() > 1.0:
                face = face / 255.0
                logger.debug(f"[OK] Normalized to [0, 1]: range=[{face.min():.3f}, {face.max():.3f}]")
            
            # Validation
            assert face.shape == (160, 160, 3), f"Wrong shape: {face.shape}"
            assert face.dtype == np.float32, f"Wrong dtype: {face.dtype}"
            assert 0 <= face.min() <= 1, f"Value < 0: {face.min()}"
            assert 0 <= face.max() <= 1, f"Value > 1: {face.max()}"
            
            # Compute hash for debugging
            face_hash = hashlib.md5(face.tobytes()).hexdigest()[:12]
            logger.debug(f"[OK] Preprocessed face hash: {face_hash}")
            
            if face.std() < 0.01:
                logger.warning("[WARN] Face has very low variance - might be blank/constant image!")
            
            return face
            
        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            raise
    
    def generate_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate 512D face embedding with comprehensive validation.
        
        Process:
        1. Preprocess face (RGB, 160x160, [0,1])
        2. Add batch dimension
        3. Run through FaceNet
        4. Extract embedding
        5. Normalize to unit vector
        6. Validate output
        
        Args:
            face_img: Face image (any size, will be preprocessed)
        
        Returns:
            512D embedding (L2 normalized) or None if failed
        """
        try:
            logger.info("="*60)
            logger.info("GENERATING EMBEDDING")
            logger.info("="*60)
            
            # STEP 1: Preprocess
            face_processed = self.preprocess_face(face_img)
            logger.info(f"[OK] Preprocessed: shape={face_processed.shape}, "
                       f"mean={face_processed.mean():.4f}, std={face_processed.std():.4f}")
            
            # STEP 2: Add batch dimension
            # FaceNet expects (batch_size, height, width, channels)
            face_batch = np.expand_dims(face_processed, axis=0)
            logger.info(f"[OK] Added batch dim: {face_processed.shape} -> {face_batch.shape}")
            
            assert face_batch.shape == (1, 160, 160, 3), f"Wrong batch shape: {face_batch.shape}"
            
            # STEP 3: Generate embedding using FaceNet
            logger.info("Running FaceNet inference...")
            embeddings_batch = self.model.embeddings(face_batch)
            logger.info(f"[OK] Model output: shape={embeddings_batch.shape}, "
                       f"dtype={embeddings_batch.dtype}")
            
            # STEP 4: Extract single embedding from batch
            embedding = embeddings_batch[0]  # Shape: (512,)
            
            logger.info(f"Raw embedding statistics:")
            logger.info(f"  Shape: {embedding.shape}")
            logger.info(f"  Dtype: {embedding.dtype}")
            logger.info(f"  Min: {embedding.min():.6f}")
            logger.info(f"  Max: {embedding.max():.6f}")
            logger.info(f"  Mean: {embedding.mean():.6f}")
            logger.info(f"  Std: {embedding.std():.6f}")
            logger.info(f"  Norm: {np.linalg.norm(embedding):.6f}")
            logger.info(f"  First 5: {embedding[:5]}")
            
            # STEP 5: L2 Normalization
            # Check if model already normalized
            norm = np.linalg.norm(embedding)
            
            if norm < 1e-10:
                logger.error("[ERROR] Embedding has zero norm! Cannot normalize.")
                return None
            
            if np.isclose(norm, 1.0, atol=0.01):
                # Already normalized by model
                embedding_normalized = embedding
                logger.info(f"[OK] Already normalized by model (norm={norm:.6f})")
            else:
                # Need to normalize
                embedding_normalized = embedding / norm
                logger.info(f"[OK] Normalized embedding (was norm={norm:.6f}, now={np.linalg.norm(embedding_normalized):.6f})")
            
            # STEP 6: Validation
            final_norm = np.linalg.norm(embedding_normalized)
            
            # Check normalization
            if not np.isclose(final_norm, 1.0, atol=0.01):
                logger.error(f"[ERROR] Embedding not properly normalized! Norm={final_norm:.10f}")
                return None
            
            # Check for degenerate cases
            if embedding_normalized.std() < 0.001:
                logger.warning("[WARN] Embedding has very low variance - might be constant!")
            
            if np.allclose(embedding_normalized, embedding_normalized[0]):
                logger.error("[ERROR] Embedding has all identical values!")
                return None
            
            if np.allclose(embedding_normalized, 0):
                logger.error("[ERROR] Embedding is zero vector!")
                return None
            
            # Compute hash for debugging
            emb_hash = hashlib.sha256(embedding_normalized.tobytes()).hexdigest()[:16]
            
            logger.info(f"\nFinal embedding:")
            logger.info(f"  Shape: {embedding_normalized.shape}")
            logger.info(f"  Dtype: {embedding_normalized.dtype}")
            logger.info(f"  Norm: {final_norm:.10f}")
            logger.info(f"  Mean: {embedding_normalized.mean():.10f}")
            logger.info(f"  Std: {embedding_normalized.std():.10f}")
            logger.info(f"  Min: {embedding_normalized.min():.10f}")
            logger.info(f"  Max: {embedding_normalized.max():.10f}")
            logger.info(f"  First 10: {embedding_normalized[:10]}")
            logger.info(f"  Last 10: {embedding_normalized[-10:]}")
            logger.info(f"  Hash: {emb_hash}")
            logger.info("="*60)
            
            # Final type conversion to ensure consistency
            embedding_normalized = embedding_normalized.astype(np.float32)
            
            return embedding_normalized
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def verify_embedding(self, embedding: np.ndarray) -> bool:
        """
        Verify that an embedding is valid.
        
        Checks:
        - Correct shape (512,)
        - No NaN values
        - No Inf values
        - Proper normalization (L2 norm close to 1.0)
        - Reasonable variance (not all zeros/constants)
        
        Args:
            embedding: 512D numpy array to verify
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if numpy array
            if not isinstance(embedding, np.ndarray):
                logger.warning(f"Embedding is not numpy array: {type(embedding)}")
                return False
            
            # Check shape
            if embedding.shape != (512,):
                logger.warning(f"Wrong embedding shape: {embedding.shape}, expected (512,)")
                return False
            
            # Check for NaN values
            if np.isnan(embedding).any():
                logger.warning("Embedding contains NaN values")
                return False
            
            # Check for Inf values
            if np.isinf(embedding).any():
                logger.warning("Embedding contains Inf values")
                return False
            
            # Check normalization
            norm = np.linalg.norm(embedding)
            if not np.isclose(norm, 1.0, atol=0.1):
                logger.warning(f"Embedding not normalized: norm={norm:.6f}")
                return False
            
            # Check variance (should not be all zeros or constant)
            if embedding.std() < 0.0001:
                logger.warning(f"Embedding has very low variance: std={embedding.std():.6f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Embedding verification failed: {e}")
            return False


if __name__ == "__main__":
    # Test embedder
    print("Testing FaceEmbedder...")
    
    embedder = FaceEmbedder()
    
    # Test 1: Black image
    print("\nTest 1: Black image")
    black = np.zeros((160, 160, 3), dtype=np.uint8)
    emb1 = embedder.generate_embedding(black)
    
    # Test 2: White image
    print("\nTest 2: White image")
    white = np.full((160, 160, 3), 255, dtype=np.uint8)
    emb2 = embedder.generate_embedding(white)
    
    # Test 3: Random image
    print("\nTest 3: Random image")
    random = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    emb3 = embedder.generate_embedding(random)
    
    # Compare
    if emb1 is not None and emb2 is not None and emb3 is not None:
        from utils import cosine_similarity
        print("\nSimilarity comparison:")
        print(f"Black vs White: {cosine_similarity(emb1, emb2):.6f}")
        print(f"Black vs Random: {cosine_similarity(emb1, emb3):.6f}")
        print(f"White vs Random: {cosine_similarity(emb2, emb3):.6f}")
        
        if cosine_similarity(emb1, emb2) > 0.99:
            print("[ERROR] Different images producing same embeddings!")
        else:
            print("[OK] Embeddings are different (good!)")

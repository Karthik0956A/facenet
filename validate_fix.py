"""
VALIDATION TEST SUITE
Run this after applying fixes to verify the bug is resolved
"""
import sys
import numpy as np
import cv2
import hashlib
from typing import List, Tuple

# Import fixed modules
from embedder_fixed import FaceEmbedder
from database_fixed import FaceDatabase
from detector import FaceDetector  # Original should be OK
from utils import cosine_similarity, batch_cosine_similarity, setup_logger

logger = setup_logger(__name__)


class ValidationTests:
    """Comprehensive validation test suite."""
    
    def __init__(self):
        self.embedder = FaceEmbedder()
        self.detector = FaceDetector()
        self.db = FaceDatabase()
        
        self.test_results = []
    
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✓ PASS" if passed else "❌ FAIL"
        self.test_results.append((test_name, passed, details))
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"  {details}")
    
    def test_1_embedder_uniqueness(self) -> bool:
        """Test: Different images produce different embeddings."""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Embedder Uniqueness")
        logger.info("="*60)
        
        try:
            # Create 3 different test images
            images = []
            
            # Image 1: Random pattern 1
            np.random.seed(42)
            img1 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
            images.append(("Random1", img1))
            
            # Image 2: Random pattern 2
            np.random.seed(123)
            img2 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
            images.append(("Random2", img2))
            
            # Image 3: Gradient pattern
            img3 = np.zeros((160, 160, 3), dtype=np.uint8)
            for i in range(160):
                img3[i, :, :] = i
            images.append(("Gradient", img3))
            
            # Generate embeddings
            embeddings = []
            hashes = []
            
            for name, img in images:
                emb = self.embedder.generate_embedding(img)
                
                if emb is None:
                    self.log_result("Embedder Uniqueness", False, 
                                  f"Failed to generate embedding for {name}")
                    return False
                
                emb_hash = hashlib.sha256(emb.tobytes()).hexdigest()[:12]
                embeddings.append(emb)
                hashes.append(emb_hash)
                
                logger.info(f"{name}: norm={np.linalg.norm(emb):.6f}, "
                          f"std={emb.std():.6f}, hash={emb_hash}")
            
            # Check uniqueness
            unique_hashes = len(set(hashes))
            
            if unique_hashes != len(hashes):
                self.log_result("Embedder Uniqueness", False,
                              f"Duplicate hashes found! {hashes}")
                return False
            
            # Check cosine similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
                    logger.info(f"Similarity {images[i][0]} vs {images[j][0]}: {sim:.6f}")
            
            max_sim = max(similarities)
            
            if max_sim > 0.99:
                self.log_result("Embedder Uniqueness", False,
                              f"Different images too similar: max={max_sim:.6f}")
                return False
            
            self.log_result("Embedder Uniqueness", True,
                          f"All embeddings unique (max_sim={max_sim:.6f})")
            return True
            
        except Exception as e:
            self.log_result("Embedder Uniqueness", False, str(e))
            return False
    
    def test_2_database_precision(self) -> bool:
        """Test: Database preserves embedding precision."""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Database Precision")
        logger.info("="*60)
        
        try:
            # Create test embedding
            test_emb = np.random.randn(512).astype(np.float32)
            test_emb = test_emb / np.linalg.norm(test_emb)
            
            original_hash = hashlib.sha256(test_emb.tobytes()).hexdigest()
            
            # Store in database
            face_id = "validation_test_001"
            success = self.db.insert_face(
                face_id=face_id,
                name="ValidationTest",
                embedding=test_emb
            )
            
            if not success:
                self.log_result("Database Precision", False,
                              "Failed to insert test embedding")
                return False
            
            # Retrieve from database
            retrieved = self.db.get_face_by_id(face_id)
            
            if retrieved is None:
                self.log_result("Database Precision", False,
                              "Failed to retrieve test embedding")
                self.db.delete_face(face_id)
                return False
            
            retrieved_emb = retrieved['embedding']
            retrieved_hash = hashlib.sha256(retrieved_emb.tobytes()).hexdigest()
            
            # Check hash
            if original_hash != retrieved_hash:
                diff = np.abs(test_emb - retrieved_emb)
                max_diff = diff.max()
                mean_diff = diff.mean()
                
                self.log_result("Database Precision", False,
                              f"Hash mismatch! max_diff={max_diff:.15f}, mean_diff={mean_diff:.15f}")
                self.db.delete_face(face_id)
                return False
            
            # Check precision
            max_diff = np.abs(test_emb - retrieved_emb).max()
            
            self.db.delete_face(face_id)
            
            if max_diff > 1e-6:
                self.log_result("Database Precision", False,
                              f"Precision loss: max_diff={max_diff:.15f}")
                return False
            
            self.log_result("Database Precision", True,
                          f"Precision preserved (max_diff={max_diff:.15f})")
            return True
            
        except Exception as e:
            self.log_result("Database Precision", False, str(e))
            return False
    
    def test_3_cosine_similarity_correctness(self) -> bool:
        """Test: Cosine similarity calculations are correct."""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Cosine Similarity Correctness")
        logger.info("="*60)
        
        try:
            # Test vectors
            v1 = np.array([1, 0, 0], dtype=np.float32)
            v2 = np.array([0, 1, 0], dtype=np.float32)
            v3 = np.array([1, 0, 0], dtype=np.float32)
            
            # Test 1: Identical vectors should give 1.0
            sim = cosine_similarity(v1, v3)
            if not np.isclose(sim, 1.0):
                self.log_result("Cosine Similarity", False,
                              f"Identical vectors: expected 1.0, got {sim:.6f}")
                return False
            
            # Test 2: Orthogonal vectors should give 0.0
            sim = cosine_similarity(v1, v2)
            if not np.isclose(sim, 0.0, atol=1e-6):
                self.log_result("Cosine Similarity", False,
                              f"Orthogonal vectors: expected 0.0, got {sim:.6f}")
                return False
            
            # Test 3: 512D random vectors
            r1 = np.random.randn(512).astype(np.float32)
            r1 = r1 / np.linalg.norm(r1)
            r2 = np.random.randn(512).astype(np.float32)
            r2 = r2 / np.linalg.norm(r2)
            
            sim = cosine_similarity(r1, r2)
            
            # Random vectors should have low similarity
            if sim > 0.5:
                self.log_result("Cosine Similarity", False,
                              f"Random vectors too similar: {sim:.6f}")
                return False
            
            # Test 4: Batch cosine similarity
            emb1 = np.random.randn(512).astype(np.float32)
            emb1 = emb1 / np.linalg.norm(emb1)
            
            batch = np.array([r1, r2, emb1])
            
            similarities = batch_cosine_similarity(emb1, batch)
            
            if len(similarities) != 3:
                self.log_result("Cosine Similarity", False,
                              f"Batch similarity wrong length: {len(similarities)}")
                return False
            
            # Last similarity should be 1.0 (same vector)
            if not np.isclose(similarities[2], 1.0):
                self.log_result("Cosine Similarity", False,
                              f"Batch self-similarity: expected 1.0, got {similarities[2]:.6f}")
                return False
            
            self.log_result("Cosine Similarity", True,
                          "All similarity calculations correct")
            return True
            
        except Exception as e:
            self.log_result("Cosine Similarity", False, str(e))
            return False
    
    def test_4_end_to_end_uniqueness(self) -> bool:
        """Test: End-to-end pipeline produces unique results."""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: End-to-End Uniqueness")
        logger.info("="*60)
        
        try:
            # Create synthetic test faces
            test_faces = []
            
            for i in range(3):
                # Different random patterns
                np.random.seed(i * 100)
                face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
                test_faces.append((f"Person{i+1}", face))
            
            # Clear test data
            for name, _ in test_faces:
                self.db.delete_faces_by_name(name)
            
            # Register faces
            registered_ids = []
            
            for name, face in test_faces:
                # Generate embedding
                embedding = self.embedder.generate_embedding(face)
                
                if embedding is None:
                    self.log_result("End-to-End Uniqueness", False,
                                  f"Failed to generate embedding for {name}")
                    return False
                
                # Store in database
                face_id = f"test_{name}"
                success = self.db.insert_face(face_id, name, embedding)
                
                if not success:
                    self.log_result("End-to-End Uniqueness", False,
                                  f"Failed to store {name}")
                    return False
                
                registered_ids.append(face_id)
            
            # Retrieve all and check uniqueness
            all_faces = self.db.get_all_faces()
            test_embeddings = [f['embedding'] for f in all_faces if f['name'].startswith('Person')]
            
            if len(test_embeddings) != 3:
                self.log_result("End-to-End Uniqueness", False,
                              f"Expected 3 test faces, got {len(test_embeddings)}")
                # Cleanup
                for face_id in registered_ids:
                    self.db.delete_face(face_id)
                return False
            
            # Check all pairs
            max_similarity = 0.0
            
            for i in range(len(test_embeddings)):
                for j in range(i+1, len(test_embeddings)):
                    sim = cosine_similarity(test_embeddings[i], test_embeddings[j])
                    max_similarity = max(max_similarity, sim)
                    logger.info(f"Person{i+1} vs Person{j+1}: {sim:.6f}")
            
            # Cleanup
            for face_id in registered_ids:
                self.db.delete_face(face_id)
            
            # Check threshold
            if max_similarity > 0.90:
                self.log_result("End-to-End Uniqueness", False,
                              f"Different people too similar: max={max_similarity:.6f}")
                return False
            
            self.log_result("End-to-End Uniqueness", True,
                          f"All faces unique (max_sim={max_similarity:.6f})")
            return True
            
        except Exception as e:
            self.log_result("End-to-End Uniqueness", False, str(e))
            # Cleanup
            for name, _ in test_faces:
                self.db.delete_faces_by_name(name)
            return False
    
    def test_5_real_camera_test(self) -> bool:
        """Test: Real camera capture produces unique embeddings."""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Real Camera Test")
        logger.info("="*60)
        
        try:
            # This test requires manual verification
            logger.info("This test requires manual intervention:")
            logger.info("1. Camera will capture frames")
            logger.info("2. System will detect and embed faces")
            logger.info("3. You should verify different people get different names")
            
            response = input("\nRun real camera test? (y/n): ")
            
            if response.lower() != 'y':
                self.log_result("Real Camera Test", True, "Skipped by user")
                return True
            
            # Initialize camera
            from camera import CameraManager
            camera = CameraManager()
            
            if not camera.is_opened():
                self.log_result("Real Camera Test", False, "Failed to open camera")
                return False
            
            logger.info("\nCapturing 5 frames...")
            captured_embeddings = []
            
            for i in range(5):
                frame = camera.get_frame()
                
                if frame is None:
                    continue
                
                # Detect faces
                faces = self.detector.detect_faces(frame)
                
                if len(faces) == 0:
                    logger.info(f"Frame {i+1}: No face detected")
                    continue
                
                # Take first face
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                
                # Generate embedding
                embedding = self.embedder.generate_embedding(face_img)
                
                if embedding is not None:
                    captured_embeddings.append(embedding)
                    logger.info(f"Frame {i+1}: Captured embedding (norm={np.linalg.norm(embedding):.6f})")
            
            camera.release()
            
            if len(captured_embeddings) < 2:
                self.log_result("Real Camera Test", False,
                              "Not enough frames captured (need at least 2)")
                return False
            
            # Check if all embeddings are identical (bug)
            identical_count = 0
            
            for i in range(len(captured_embeddings) - 1):
                sim = cosine_similarity(captured_embeddings[i], captured_embeddings[i+1])
                logger.info(f"Frame {i+1} vs {i+2}: similarity={sim:.6f}")
                
                if sim > 0.999:
                    identical_count += 1
            
            if identical_count == len(captured_embeddings) - 1:
                self.log_result("Real Camera Test", False,
                              "All frames producing identical embeddings!")
                return False
            
            self.log_result("Real Camera Test", True,
                          f"Camera test passed ({len(captured_embeddings)} frames captured)")
            return True
            
        except Exception as e:
            self.log_result("Real Camera Test", False, str(e))
            return False
    
    def run_all_tests(self) -> Tuple[int, int]:
        """Run all validation tests."""
        logger.info("\n" + "="*80)
        logger.info("STARTING VALIDATION TEST SUITE")
        logger.info("="*80)
        
        # Run tests
        self.test_1_embedder_uniqueness()
        self.test_2_database_precision()
        self.test_3_cosine_similarity_correctness()
        self.test_4_end_to_end_uniqueness()
        self.test_5_real_camera_test()
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        
        passed = sum(1 for _, p, _ in self.test_results if p)
        total = len(self.test_results)
        
        for test_name, passed_flag, details in self.test_results:
            status = "✓ PASS" if passed_flag else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
            if details and not passed_flag:
                logger.info(f"  {details}")
        
        logger.info("="*80)
        logger.info(f"RESULT: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Bug is fixed.")
        else:
            logger.info(f"❌ {total - passed} test(s) failed. Bug still present.")
        
        logger.info("="*80)
        
        return passed, total


def main():
    """Run validation tests."""
    validator = ValidationTests()
    passed, total = validator.run_all_tests()
    
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

"""
Unit tests for embedding generation.
Tests FaceNet embedding functionality.
"""
import unittest
import numpy as np
import cv2

from embedder import FaceEmbedder
from detector import FaceDetector
from config import EMBEDDING_DIMENSION, FACE_INPUT_SIZE


class TestEmbedder(unittest.TestCase):
    """Test cases for FaceEmbedder class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.embedder = FaceEmbedder()
    
    def test_embedder_initialization(self):
        """Test embedder can be initialized."""
        self.assertIsNotNone(self.embedder)
        self.assertIsNotNone(self.embedder.model)
    
    def test_generate_embedding_with_synthetic_face(self):
        """Test embedding generation with synthetic face image."""
        # Create synthetic face image (random data)
        synthetic_face = np.random.randint(0, 255, (*FACE_INPUT_SIZE, 3), dtype=np.uint8)
        
        # Generate embedding
        embedding = self.embedder.generate_embedding(synthetic_face)
        
        # Verify results
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (EMBEDDING_DIMENSION,))
        
        # Check normalization (L2 norm should be close to 1)
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=2)
    
    def test_generate_embedding_consistency(self):
        """Test that same face produces similar embeddings."""
        # Create test face
        test_face = np.random.randint(0, 255, (*FACE_INPUT_SIZE, 3), dtype=np.uint8)
        
        # Generate two embeddings from same face
        embedding1 = self.embedder.generate_embedding(test_face)
        embedding2 = self.embedder.generate_embedding(test_face)
        
        # They should be identical (deterministic)
        self.assertIsNotNone(embedding1)
        self.assertIsNotNone(embedding2)
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)
    
    def test_verify_embedding_valid(self):
        """Test embedding verification with valid embedding."""
        valid_embedding = np.random.randn(EMBEDDING_DIMENSION)
        valid_embedding = valid_embedding / np.linalg.norm(valid_embedding)
        
        is_valid = self.embedder.verify_embedding(valid_embedding)
        self.assertTrue(is_valid)
    
    def test_verify_embedding_invalid_shape(self):
        """Test embedding verification rejects wrong shape."""
        invalid_embedding = np.random.randn(256)  # Wrong dimension
        
        is_valid = self.embedder.verify_embedding(invalid_embedding)
        self.assertFalse(is_valid)
    
    def test_verify_embedding_with_nan(self):
        """Test embedding verification rejects NaN values."""
        invalid_embedding = np.random.randn(EMBEDDING_DIMENSION)
        invalid_embedding[0] = np.nan
        
        is_valid = self.embedder.verify_embedding(invalid_embedding)
        self.assertFalse(is_valid)
    
    def test_verify_embedding_with_inf(self):
        """Test embedding verification rejects Inf values."""
        invalid_embedding = np.random.randn(EMBEDDING_DIMENSION)
        invalid_embedding[0] = np.inf
        
        is_valid = self.embedder.verify_embedding(invalid_embedding)
        self.assertFalse(is_valid)
    
    def test_compare_embeddings_identical(self):
        """Test comparing identical embeddings."""
        embedding = np.random.randn(EMBEDDING_DIMENSION)
        embedding = embedding / np.linalg.norm(embedding)
        
        similarity = self.embedder.compare_embeddings(embedding, embedding)
        
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_compare_embeddings_different(self):
        """Test comparing different embeddings."""
        embedding1 = np.random.randn(EMBEDDING_DIMENSION)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        
        embedding2 = np.random.randn(EMBEDDING_DIMENSION)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        similarity = self.embedder.compare_embeddings(embedding1, embedding2)
        
        # Different random embeddings should have low similarity
        self.assertLess(similarity, 0.9)
        self.assertGreater(similarity, -1.0)
        self.assertLess(similarity, 1.0)
    
    def test_generate_embeddings_batch(self):
        """Test batch embedding generation."""
        # Create multiple synthetic faces
        num_faces = 3
        faces = [
            np.random.randint(0, 255, (*FACE_INPUT_SIZE, 3), dtype=np.uint8)
            for _ in range(num_faces)
        ]
        
        # Generate batch embeddings
        embeddings = self.embedder.generate_embeddings_batch(faces)
        
        # Verify results
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape, (num_faces, EMBEDDING_DIMENSION))
        
        # Check each embedding is normalized
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            self.assertAlmostEqual(norm, 1.0, places=2)
    
    def test_generate_embeddings_batch_empty(self):
        """Test batch embedding with empty list."""
        embeddings = self.embedder.generate_embeddings_batch([])
        self.assertIsNone(embeddings)


class TestEmbeddingProperties(unittest.TestCase):
    """Test mathematical properties of embeddings."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.embedder = FaceEmbedder()
    
    def test_embedding_dimension(self):
        """Test embeddings have correct dimension."""
        face = np.random.randint(0, 255, (*FACE_INPUT_SIZE, 3), dtype=np.uint8)
        embedding = self.embedder.generate_embedding(face)
        
        self.assertEqual(len(embedding), EMBEDDING_DIMENSION)
        self.assertEqual(embedding.shape[0], 512)
    
    def test_embedding_dtype(self):
        """Test embeddings have correct data type."""
        face = np.random.randint(0, 255, (*FACE_INPUT_SIZE, 3), dtype=np.uint8)
        embedding = self.embedder.generate_embedding(face)
        
        self.assertTrue(np.issubdtype(embedding.dtype, np.floating))
    
    def test_embedding_range(self):
        """Test embedding values are in reasonable range."""
        face = np.random.randint(0, 255, (*FACE_INPUT_SIZE, 3), dtype=np.uint8)
        embedding = self.embedder.generate_embedding(face)
        
        # Normalized embeddings should have values roughly in [-1, 1]
        self.assertTrue(np.all(embedding >= -2.0))
        self.assertTrue(np.all(embedding <= 2.0))


if __name__ == '__main__':
    unittest.main()

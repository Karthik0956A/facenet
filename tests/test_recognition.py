"""
Unit tests for face recognition.
Tests recognition accuracy and similarity matching.
"""
import unittest
import numpy as np
from unittest.mock import Mock, patch

from recognizer import FaceRecognizer
from database import FaceDatabase
from config import EMBEDDING_DIMENSION, RECOGNITION_THRESHOLD
from utils import generate_face_id


class TestRecognizer(unittest.TestCase):
    """Test cases for FaceRecognizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock database to avoid actual MongoDB connection during tests
        self.mock_db = Mock(spec=FaceDatabase)
        self.recognizer = FaceRecognizer(db=self.mock_db)
    
    def test_recognizer_initialization(self):
        """Test recognizer can be initialized."""
        self.assertIsNotNone(self.recognizer)
        self.assertEqual(self.recognizer.recognition_threshold, RECOGNITION_THRESHOLD)
    
    def test_recognize_with_empty_database(self):
        """Test recognition with no faces in database."""
        # Mock empty database
        self.mock_db.get_all_faces.return_value = []
        
        # Create query embedding
        query_embedding = np.random.randn(EMBEDDING_DIMENSION)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Attempt recognition
        is_recognized, face_data, confidence = self.recognizer.recognize(query_embedding)
        
        self.assertFalse(is_recognized)
        self.assertIsNone(face_data)
        self.assertEqual(confidence, 0.0)
    
    def test_recognize_with_exact_match(self):
        """Test recognition with exact matching embedding."""
        # Create test embedding
        test_embedding = np.random.randn(EMBEDDING_DIMENSION)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        # Mock database with one face
        mock_face = {
            'face_id': generate_face_id(),
            'name': 'Test Person',
            'embedding': test_embedding.copy(),
            'summary': 'Test',
            'created_at': '2024-01-01T00:00:00'
        }
        
        self.mock_db.get_all_faces.return_value = [mock_face]
        
        # Recognize with same embedding
        is_recognized, face_data, confidence = self.recognizer.recognize(test_embedding)
        
        self.assertTrue(is_recognized)
        self.assertIsNotNone(face_data)
        self.assertEqual(face_data['name'], 'Test Person')
        self.assertAlmostEqual(confidence, 1.0, places=4)
    
    def test_recognize_below_threshold(self):
        """Test recognition fails when below threshold."""
        # Create two different embeddings
        stored_embedding = np.random.randn(EMBEDDING_DIMENSION)
        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
        
        query_embedding = np.random.randn(EMBEDDING_DIMENSION)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Mock database
        mock_face = {
            'face_id': generate_face_id(),
            'name': 'Test Person',
            'embedding': stored_embedding,
            'summary': '',
            'created_at': '2024-01-01T00:00:00'
        }
        
        self.mock_db.get_all_faces.return_value = [mock_face]
        
        # Recognize
        is_recognized, face_data, confidence = self.recognizer.recognize(query_embedding)
        
        # Should not be recognized (random embeddings unlikely to match)
        # unless by chance they're similar
        if confidence >= RECOGNITION_THRESHOLD:
            self.assertTrue(is_recognized)
        else:
            self.assertFalse(is_recognized)
    
    def test_recognize_top_k(self):
        """Test top-K recognition."""
        # Create multiple test embeddings
        query_embedding = np.random.randn(EMBEDDING_DIMENSION)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Create 5 stored faces with varying similarity
        stored_faces = []
        for i in range(5):
            # Create embedding similar to query with varying noise
            embedding = query_embedding + np.random.randn(EMBEDDING_DIMENSION) * (0.1 * (i + 1))
            embedding = embedding / np.linalg.norm(embedding)
            
            face = {
                'face_id': generate_face_id(),
                'name': f'Person {i+1}',
                'embedding': embedding,
                'summary': '',
                'created_at': '2024-01-01T00:00:00'
            }
            stored_faces.append(face)
        
        self.mock_db.get_all_faces.return_value = stored_faces
        
        # Get top 3 matches
        matches = self.recognizer.recognize_top_k(query_embedding, k=3, threshold=0.0)
        
        # Should return at most 3 matches
        self.assertLessEqual(len(matches), 3)
        
        # Matches should be sorted by similarity (descending)
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                self.assertGreaterEqual(matches[i][1], matches[i+1][1])
    
    def test_get_confidence_level(self):
        """Test confidence level categorization."""
        # High confidence
        level = self.recognizer.get_confidence_level(0.9)
        self.assertEqual(level, "HIGH")
        
        # Medium confidence
        level = self.recognizer.get_confidence_level(0.65)
        self.assertEqual(level, "MEDIUM")
        
        # Low confidence
        level = self.recognizer.get_confidence_level(0.5)
        self.assertEqual(level, "LOW")
        
        # Very low confidence
        level = self.recognizer.get_confidence_level(0.3)
        self.assertEqual(level, "VERY LOW")
    
    def test_verify_identity_success(self):
        """Test identity verification with matching embedding."""
        # Create test embedding
        test_embedding = np.random.randn(EMBEDDING_DIMENSION)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        # Mock database
        mock_face = {
            'face_id': generate_face_id(),
            'name': 'John Doe',
            'embedding': test_embedding.copy(),
            'summary': '',
            'created_at': '2024-01-01T00:00:00'
        }
        
        self.mock_db.get_face_by_name.return_value = mock_face
        
        # Verify identity
        is_verified, similarity = self.recognizer.verify_identity(test_embedding, 'John Doe')
        
        self.assertTrue(is_verified)
        self.assertAlmostEqual(similarity, 1.0, places=4)
    
    def test_verify_identity_not_found(self):
        """Test identity verification with non-existent name."""
        # Mock database returns None
        self.mock_db.get_face_by_name.return_value = None
        
        # Create query embedding
        query_embedding = np.random.randn(EMBEDDING_DIMENSION)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Verify identity
        is_verified, similarity = self.recognizer.verify_identity(query_embedding, 'Unknown Person')
        
        self.assertFalse(is_verified)
        self.assertEqual(similarity, 0.0)
    
    def test_set_threshold(self):
        """Test updating recognition threshold."""
        new_threshold = 0.75
        self.recognizer.set_threshold(new_threshold)
        
        self.assertEqual(self.recognizer.recognition_threshold, new_threshold)
    
    def test_set_invalid_threshold(self):
        """Test setting invalid threshold is rejected."""
        original_threshold = self.recognizer.recognition_threshold
        
        # Try to set invalid threshold
        self.recognizer.set_threshold(-0.5)
        
        # Threshold should remain unchanged
        self.assertEqual(self.recognizer.recognition_threshold, original_threshold)
    
    def test_get_statistics(self):
        """Test getting system statistics."""
        self.mock_db.count_faces.return_value = 10
        self.mock_db.client = Mock()  # Mock client to indicate connection
        
        stats = self.recognizer.get_statistics()
        
        self.assertIn('total_faces', stats)
        self.assertEqual(stats['total_faces'], 10)
        self.assertIn('recognition_threshold', stats)
        self.assertIn('database_connected', stats)


class TestRecognitionAccuracy(unittest.TestCase):
    """Test recognition accuracy with synthetic data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_db = Mock(spec=FaceDatabase)
        self.recognizer = FaceRecognizer(db=self.mock_db)
    
    def test_similar_faces_recognition(self):
        """Test that similar embeddings are recognized as same person."""
        # Create base embedding
        base_embedding = np.random.randn(EMBEDDING_DIMENSION)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        # Create slightly perturbed version (simulating same person, different photo)
        similar_embedding = base_embedding + np.random.randn(EMBEDDING_DIMENSION) * 0.05
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
        
        # Mock database
        mock_face = {
            'face_id': generate_face_id(),
            'name': 'Test Person',
            'embedding': base_embedding,
            'summary': '',
            'created_at': '2024-01-01T00:00:00'
        }
        
        self.mock_db.get_all_faces.return_value = [mock_face]
        
        # Recognize with similar embedding
        is_recognized, face_data, confidence = self.recognizer.recognize(similar_embedding)
        
        # Should be recognized with high confidence
        self.assertTrue(is_recognized)
        self.assertGreater(confidence, 0.8)


if __name__ == '__main__':
    unittest.main()

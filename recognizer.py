"""
Face recognition module.
Matches face embeddings against database using cosine similarity.
"""
from typing import Optional, Tuple, List, Dict
import numpy as np

from database import FaceDatabase
from config import RECOGNITION_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD
from utils import setup_logger, batch_cosine_similarity

logger = setup_logger(__name__)


class FaceRecognizer:
    """
    Face recognition engine using embedding similarity matching.
    Compares query embeddings against stored embeddings in database.
    """
    
    def __init__(self, db: Optional[FaceDatabase] = None):
        """
        Initialize recognizer.
        
        Args:
            db: Database instance (creates new if not provided)
        """
        self.db = db if db is not None else FaceDatabase()
        self.recognition_threshold = RECOGNITION_THRESHOLD
        self.high_confidence_threshold = HIGH_CONFIDENCE_THRESHOLD
        logger.info(
            f"Recognizer initialized with threshold: {self.recognition_threshold}"
        )
    
    def recognize(
        self,
        query_embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[bool, Optional[Dict], float]:
        """
        Recognize face from embedding.
        
        Args:
            query_embedding: 512-dimensional query embedding
            threshold: Custom threshold (uses default if None)
        
        Returns:
            Tuple of (is_recognized, face_data, confidence_score)
            - is_recognized: True if match found above threshold
            - face_data: Database record of matched face or None
            - confidence_score: Similarity score (0-1)
        """
        if threshold is None:
            threshold = self.recognition_threshold
        
        try:
            # Get all stored faces
            stored_faces = self.db.get_all_faces()
            
            if not stored_faces:
                logger.warning("No faces in database")
                return False, None, 0.0
            
            # Extract embeddings
            stored_embeddings = [face['embedding'] for face in stored_faces]
            
            # Calculate similarities using batch operation for efficiency
            similarities = batch_cosine_similarity(query_embedding, stored_embeddings)
            
            # Log ALL similarity scores for debugging
            logger.info("=== Similarity Scores ===")
            for i, (face, sim) in enumerate(zip(stored_faces, similarities)):
                logger.info(f"  {face['name']}: {sim:.4f}")
            logger.info(f"  Threshold: {threshold:.4f}")
            logger.info("========================")
            
            # Find best match
            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])
            best_match_name = stored_faces[best_idx]['name']
            
            # Log detailed comparison
            logger.info(
                f"Best match: {best_match_name} "
                f"(similarity: {best_similarity:.4f}, threshold: {threshold:.4f})"
            )
            
            # Check if above threshold
            if best_similarity >= threshold:
                best_match = stored_faces[best_idx]
                
                # Additional validation: check if this is significantly better than others
                sorted_similarities = sorted(similarities, reverse=True)
                if len(sorted_similarities) > 1:
                    second_best = sorted_similarities[1]
                    margin = best_similarity - second_best
                    logger.info(f"Margin over second best: {margin:.4f}")
                    
                    # If margin is too small, might be uncertain
                    if margin < 0.05:
                        logger.warning(f"Low confidence margin: {margin:.4f}")
                
                logger.info(
                    f"✓ Face recognized: {best_match['name']} "
                    f"(confidence: {best_similarity:.4f})"
                )
                return True, best_match, best_similarity
            else:
                logger.info(
                    f"✗ No match above threshold "
                    f"(best: {best_similarity:.4f}, threshold: {threshold:.4f})"
                )
                return False, None, best_similarity
            
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return False, None, 0.0
    
    def recognize_top_k(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Get top K matches for query embedding.
        
        Args:
            query_embedding: 512-dimensional query embedding
            k: Number of top matches to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (face_data, similarity) tuples, sorted by similarity
        """
        if threshold is None:
            threshold = self.recognition_threshold
        
        try:
            # Get all stored faces
            stored_faces = self.db.get_all_faces()
            
            if not stored_faces:
                logger.warning("No faces in database")
                return []
            
            # Extract embeddings
            stored_embeddings = [face['embedding'] for face in stored_faces]
            
            # Calculate similarities
            similarities = batch_cosine_similarity(query_embedding, stored_embeddings)
            
            # Create list of (face, similarity) pairs
            matches = list(zip(stored_faces, similarities))
            
            # Filter by threshold
            matches = [
                (face, float(sim)) for face, sim in matches
                if sim >= threshold
            ]
            
            # Sort by similarity (descending)
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            top_k = matches[:k]
            
            logger.debug(f"Found {len(top_k)} matches above threshold")
            
            return top_k
            
        except Exception as e:
            logger.error(f"Top-K recognition failed: {e}")
            return []
    
    def get_confidence_level(self, similarity: float) -> str:
        """
        Get confidence level description from similarity score.
        
        Args:
            similarity: Similarity score (0-1)
        
        Returns:
            Confidence level string
        """
        if similarity >= self.high_confidence_threshold:
            return "HIGH"
        elif similarity >= self.recognition_threshold:
            return "MEDIUM"
        elif similarity >= self.recognition_threshold * 0.7:
            return "LOW"
        else:
            return "VERY LOW"
    
    def verify_identity(
        self,
        query_embedding: np.ndarray,
        claimed_name: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Verify if query embedding matches claimed identity.
        
        Args:
            query_embedding: 512-dimensional query embedding
            claimed_name: Name being claimed
            threshold: Verification threshold
        
        Returns:
            Tuple of (is_verified, similarity_score)
        """
        if threshold is None:
            threshold = self.recognition_threshold
        
        try:
            # Get claimed identity from database
            claimed_face = self.db.get_face_by_name(claimed_name)
            
            if claimed_face is None:
                logger.warning(f"Identity not found: {claimed_name}")
                return False, 0.0
            
            # Calculate similarity
            stored_embedding = claimed_face['embedding']
            from utils import cosine_similarity
            similarity = cosine_similarity(query_embedding, stored_embedding)
            
            # Verify
            is_verified = similarity >= threshold
            
            logger.info(
                f"Identity verification for '{claimed_name}': "
                f"{'PASSED' if is_verified else 'FAILED'} "
                f"(similarity: {similarity:.4f})"
            )
            
            return is_verified, similarity
            
        except Exception as e:
            logger.error(f"Identity verification failed: {e}")
            return False, 0.0
    
    def set_threshold(self, threshold: float) -> None:
        """
        Update recognition threshold.
        
        Args:
            threshold: New threshold value (0-1)
        """
        if not 0.0 <= threshold <= 1.0:
            logger.error(f"Invalid threshold: {threshold}")
            return
        
        self.recognition_threshold = threshold
        logger.info(f"Recognition threshold updated to {threshold}")
    
    def get_statistics(self) -> Dict:
        """
        Get recognition system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            total_faces = self.db.count_faces()
            
            stats = {
                'total_faces': total_faces,
                'recognition_threshold': self.recognition_threshold,
                'high_confidence_threshold': self.high_confidence_threshold,
                'database_connected': self.db.client is not None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


if __name__ == "__main__":
    # Test recognizer
    print("Testing face recognizer...")
    
    try:
        recognizer = FaceRecognizer()
        print("✓ Recognizer initialized")
        
        stats = recognizer.get_statistics()
        print(f"✓ Database statistics: {stats}")
        
        # Test with camera
        from camera import Camera
        from detector import FaceDetector
        from embedder import FaceEmbedder
        
        detector = FaceDetector()
        embedder = FaceEmbedder()
        
        with Camera() as cam:
            print("\nPress 'c' to capture and recognize, 'q' to quit")
            frame = cam.show_stream()
            
            if frame is not None:
                # Detect face
                face, detection = detector.detect_and_extract_largest(frame)
                
                if face is not None:
                    # Generate embedding
                    embedding = embedder.generate_embedding(face)
                    
                    if embedding is not None:
                        # Recognize
                        is_recognized, face_data, confidence = recognizer.recognize(embedding)
                        
                        if is_recognized:
                            print(f"✓ Face recognized: {face_data['name']}")
                            print(f"  Confidence: {confidence:.4f}")
                            print(f"  Level: {recognizer.get_confidence_level(confidence)}")
                        else:
                            print("✗ Face not recognized")
                            print(f"  Best similarity: {confidence:.4f}")
                    else:
                        print("✗ Failed to generate embedding")
                else:
                    print("✗ No face detected")
            else:
                print("✗ No frame captured")
                
    except Exception as e:
        print(f"✗ Test failed: {e}")

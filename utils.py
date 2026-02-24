"""
Utility functions for facial recognition system.
Includes logging setup, image preprocessing, and similarity calculations.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import cv2
from datetime import datetime

from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger(name: str, log_file: Optional[Path] = None, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Setup structured logger with file and console handlers.
    
    Args:
        name: Logger name (typically module name)
        log_file: Optional file path for logging
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = LOG_FILE
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Cosine similarity = (A · B) / (||A|| * ||B||)
    Range: [-1, 1], where 1 means identical, 0 means orthogonal, -1 means opposite
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score
    """
    # Normalize vectors
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Calculate dot product
    similarity = np.dot(embedding1_norm, embedding2_norm)
    
    return float(similarity)


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate Euclidean (L2) distance between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(embedding1 - embedding2))


def preprocess_face(face_img: np.ndarray, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """
    Preprocess face image for FaceNet model.
    
    Args:
        face_img: Face image (BGR format)
        target_size: Target size for resizing
    
    Returns:
        Preprocessed face image normalized to [-1, 1]
    """
    # Resize to target size
    face_resized = cv2.resize(face_img, target_size)
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1] range (standard for FaceNet)
    face_normalized = (face_rgb.astype('float32') - 127.5) / 127.5
    
    return face_normalized


def draw_face_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: Optional[str] = None,
    confidence: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw bounding box and label on image.
    
    Args:
        image: Input image
        box: Bounding box coordinates (x, y, width, height)
        label: Optional label text
        confidence: Optional confidence score
        color: Box color in BGR format
    
    Returns:
        Image with drawn bounding box
    """
    x, y, width, height = box
    
    # Draw rectangle
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
    
    # Draw label if provided
    if label:
        text = f"{label}"
        if confidence is not None:
            text += f" ({confidence:.2f})"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (x, y - text_height - baseline - 5),
            (x + text_width, y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return image


def validate_name(name: str) -> bool:
    """
    Validate name for registration.
    
    Args:
        name: Name to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not name or not name.strip():
        return False
    
    # Check length
    if len(name) < 2 or len(name) > 50:
        return False
    
    # Check for invalid characters (allow alphanumeric, spaces, hyphens, underscores)
    if not all(c.isalnum() or c in ' -_' for c in name):
        return False
    
    return True


def generate_face_id() -> str:
    """
    Generate unique face ID.
    
    Returns:
        Unique identifier string
    """
    from uuid import uuid4
    return str(uuid4())


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO formatted timestamp string
    """
    return datetime.utcnow().isoformat()


def cleanup_temp_file(file_path: Path) -> bool:
    """
    Safely delete temporary file.
    
    Args:
        file_path: Path to file to delete
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception as e:
        logging.error(f"Failed to delete temp file {file_path}: {e}")
        return False


def batch_cosine_similarity(embedding: np.ndarray, embeddings_list: list) -> np.ndarray:
    """
    Calculate cosine similarity between one embedding and a list of embeddings.
    Optimized for batch processing.
    
    Args:
        embedding: Query embedding
        embeddings_list: List of embeddings to compare against
    
    Returns:
        Array of similarity scores
    """
    if not embeddings_list:
        return np.array([])
    
    # Convert list to numpy array for vectorized operations
    embeddings_matrix = np.array(embeddings_list)
    
    # Normalize query embedding
    embedding_norm = embedding / np.linalg.norm(embedding)
    
    # Normalize all embeddings in matrix
    embeddings_matrix_norm = embeddings_matrix / np.linalg.norm(
        embeddings_matrix, axis=1, keepdims=True
    )
    
    # Calculate all similarities at once using matrix multiplication
    similarities = np.dot(embeddings_matrix_norm, embedding_norm)
    
    return similarities


if __name__ == "__main__":
    # Test utilities
    logger = setup_logger("test")
    logger.info("Logger initialized successfully")
    
    # Test cosine similarity
    emb1 = np.random.randn(512)
    emb2 = emb1 + np.random.randn(512) * 0.1
    similarity = cosine_similarity(emb1, emb2)
    logger.info(f"Cosine similarity: {similarity:.4f}")
    
    # Test batch similarity
    embeddings_list = [np.random.randn(512) for _ in range(5)]
    similarities = batch_cosine_similarity(emb1, embeddings_list)
    logger.info(f"Batch similarities: {similarities}")

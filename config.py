"""
Configuration module for FaceNet facial recognition system.
Manages environment variables and system-wide settings.
"""
import os
from pathlib import Path
from typing import Optional

# MongoDB Configuration
MONGODB_URI: str = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME: str = os.getenv('MONGODB_DB_NAME', 'facenet_db')
MONGODB_COLLECTION: str = os.getenv('MONGODB_COLLECTION', 'faces')

# MongoDB Connection Settings
MONGODB_MAX_POOL_SIZE: int = int(os.getenv('MONGODB_MAX_POOL_SIZE', '10'))
MONGODB_TIMEOUT_MS: int = int(os.getenv('MONGODB_TIMEOUT_MS', '5000'))

# Recognition thresholds  
# Increased to 0.90 to prevent different people from being recognized as the same
RECOGNITION_THRESHOLD: float = float(os.getenv('RECOGNITION_THRESHOLD', '0.90'))
HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.95'))

# Camera Settings
CAMERA_INDEX: int = int(os.getenv('CAMERA_INDEX', '0'))
CAMERA_WIDTH: int = int(os.getenv('CAMERA_WIDTH', '640'))
CAMERA_HEIGHT: int = int(os.getenv('CAMERA_HEIGHT', '480'))
CAPTURE_KEY: str = os.getenv('CAPTURE_KEY', 'c')
EXIT_KEY: str = os.getenv('EXIT_KEY', 'q')

# Face Detection Settings (MTCNN)
MTCNN_MIN_FACE_SIZE: int = int(os.getenv('MTCNN_MIN_FACE_SIZE', '40'))
MTCNN_THRESHOLDS: tuple = (0.6, 0.7, 0.7)  # P-Net, R-Net, O-Net
MTCNN_SCALE_FACTOR: float = 0.709

# Face Processing
FACE_INPUT_SIZE: tuple = (160, 160)  # FaceNet input size
EMBEDDING_DIMENSION: int = 512

# File Paths
PROJECT_ROOT: Path = Path(__file__).parent
MODELS_DIR: Path = PROJECT_ROOT / 'models'
TEMP_DIR: Path = PROJECT_ROOT / 'temp'
LOGS_DIR: Path = PROJECT_ROOT / 'logs'

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Logging Configuration
LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE: Path = LOGS_DIR / 'facenet.log'

# Database Retry Settings
DB_MAX_RETRIES: int = int(os.getenv('DB_MAX_RETRIES', '3'))
DB_RETRY_DELAY: float = float(os.getenv('DB_RETRY_DELAY', '1.0'))

# Performance Settings
BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '1'))
USE_GPU: bool = os.getenv('USE_GPU', 'false').lower() == 'true'

def validate_config() -> bool:
    """
    Validate critical configuration settings.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        assert 0.0 <= RECOGNITION_THRESHOLD <= 1.0, "Recognition threshold must be between 0 and 1"
        assert EMBEDDING_DIMENSION == 512, "FaceNet uses 512-dimensional embeddings"
        assert CAMERA_INDEX >= 0, "Camera index must be non-negative"
        assert MTCNN_MIN_FACE_SIZE > 0, "Minimum face size must be positive"
        return True
    except AssertionError as e:
        print(f"Configuration error: {e}")
        return False

if __name__ == "__main__":
    # Test configuration
    if validate_config():
        print("✓ Configuration is valid")
        print(f"MongoDB URI: {MONGODB_URI}")
        print(f"Recognition Threshold: {RECOGNITION_THRESHOLD}")
        print(f"Camera Index: {CAMERA_INDEX}")
    else:
        print("✗ Configuration validation failed")

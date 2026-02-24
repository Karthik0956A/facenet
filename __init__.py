"""
FaceNet Facial Recognition System
A production-quality facial recognition module using FaceNet embeddings.
"""

__version__ = "1.0.0"
__author__ = "FaceNet Team"
__license__ = "MIT"

from .camera import Camera
from .detector import FaceDetector
from .embedder import FaceEmbedder
from .database import FaceDatabase
from .recognizer import FaceRecognizer
from .register import FaceRegistrar
from .delete import FaceDeleter
from .smart_recognition import SmartRecognition

__all__ = [
    "Camera",
    "FaceDetector",
    "FaceEmbedder",
    "FaceDatabase",
    "FaceRecognizer",
    "FaceRegistrar",
    "FaceDeleter",
    "SmartRecognition",
]

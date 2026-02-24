"""
Face detector module using MTCNN.
Handles face detection from images and video frames.
"""
from typing import List, Tuple, Optional
import numpy as np
import cv2
from mtcnn import MTCNN

from config import (
    MTCNN_MIN_FACE_SIZE,
    MTCNN_THRESHOLDS,
    MTCNN_SCALE_FACTOR
)
from utils import setup_logger

logger = setup_logger(__name__)


class FaceDetector:
    """
    MTCNN-based face detector.
    Detects faces and provides bounding boxes with confidence scores.
    """
    
    def __init__(self):
        """Initialize MTCNN detector."""
        try:
            # Initialize MTCNN with default parameters
            # The mtcnn library has a different API than expected
            self.detector = MTCNN()
            logger.info("MTCNN detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect all faces in an image.
        
        Args:
            image: Input image in BGR format (OpenCV default)
        
        Returns:
            List of detection dictionaries containing:
                - 'box': [x, y, width, height]
                - 'confidence': detection confidence
                - 'keypoints': facial landmarks
        """
        try:
            # Convert BGR to RGB (MTCNN expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detector.detect_faces(image_rgb)
            
            logger.debug(f"Detected {len(detections)} face(s)")
            return detections
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def detect_largest_face(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect the largest face in an image.
        Useful for single-person scenarios.
        
        Args:
            image: Input image in BGR format
        
        Returns:
            Detection dictionary or None if no face found
        """
        detections = self.detect_faces(image)
        
        if not detections:
            logger.warning("No faces detected")
            return None
        
        # Find largest face by area
        largest_face = max(
            detections,
            key=lambda d: d['box'][2] * d['box'][3]  # width * height
        )
        
        logger.debug(f"Largest face confidence: {largest_face['confidence']:.3f}")
        return largest_face
    
    def extract_face(
        self,
        image: np.ndarray,
        detection: dict,
        margin: int = 0
    ) -> Optional[np.ndarray]:
        """
        Extract face region from image based on detection.
        
        Args:
            image: Input image in BGR format
            detection: Detection dictionary from detect_faces
            margin: Pixels to add around face box (default: 0)
        
        Returns:
            Cropped face image or None if extraction fails
        """
        try:
            box = detection['box']
            x, y, width, height = box
            
            # Add margin
            x = max(0, x - margin)
            y = max(0, y - margin)
            width = width + 2 * margin
            height = height + 2 * margin
            
            # Ensure we don't go out of bounds
            x2 = min(image.shape[1], x + width)
            y2 = min(image.shape[0], y + height)
            
            # Extract face region
            face = image[y:y2, x:x2]
            
            if face.size == 0:
                logger.warning("Extracted face region is empty")
                return None
            
            return face
            
        except Exception as e:
            logger.error(f"Face extraction failed: {e}")
            return None
    
    def detect_and_extract_largest(
        self,
        image: np.ndarray,
        margin: int = 20
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Detect and extract the largest face in one operation.
        
        Args:
            image: Input image in BGR format
            margin: Pixels to add around face box
        
        Returns:
            Tuple of (face_image, detection) or (None, None) if no face
        """
        detection = self.detect_largest_face(image)
        
        if detection is None:
            return None, None
        
        face = self.extract_face(image, detection, margin)
        
        if face is None:
            return None, None
        
        return face, detection
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[dict],
        draw_keypoints: bool = False
    ) -> np.ndarray:
        """
        Draw bounding boxes and keypoints on image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            draw_keypoints: Whether to draw facial landmarks
        
        Returns:
            Image with drawn detections
        """
        output = image.copy()
        
        for detection in detections:
            # Draw bounding box
            box = detection['box']
            x, y, width, height = box
            confidence = detection['confidence']
            
            # Box with confidence
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Confidence text
            text = f"{confidence:.2f}"
            cv2.putText(
                output,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            # Draw keypoints if requested
            if draw_keypoints and 'keypoints' in detection:
                keypoints = detection['keypoints']
                for name, point in keypoints.items():
                    cv2.circle(output, point, 2, (0, 0, 255), -1)
        
        return output
    
    def is_valid_detection(
        self,
        detection: dict,
        min_confidence: float = 0.9,
        min_size: int = 40
    ) -> bool:
        """
        Validate detection quality.
        
        Args:
            detection: Detection dictionary
            min_confidence: Minimum confidence threshold
            min_size: Minimum face size (width or height)
        
        Returns:
            True if detection meets quality criteria
        """
        if detection is None:
            return False
        
        # Check confidence
        if detection['confidence'] < min_confidence:
            logger.debug(f"Low confidence: {detection['confidence']:.3f}")
            return False
        
        # Check size
        box = detection['box']
        width, height = box[2], box[3]
        
        if width < min_size or height < min_size:
            logger.debug(f"Face too small: {width}x{height}")
            return False
        
        return True


if __name__ == "__main__":
    # Test face detector
    print("Testing face detector...")
    
    detector = FaceDetector()
    print("✓ Detector initialized")
    
    # Test with webcam
    from camera import Camera
    
    with Camera() as cam:
        print("Press 'c' to capture and detect, 'q' to quit")
        frame = cam.show_stream()
        
        if frame is not None:
            detections = detector.detect_faces(frame)
            print(f"✓ Detected {len(detections)} face(s)")
            
            if detections:
                annotated = detector.draw_detections(frame, detections, draw_keypoints=True)
                cv2.imshow("Detections", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

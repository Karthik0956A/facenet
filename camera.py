"""
Camera module for webcam capture using OpenCV.
Handles real-time video streaming and image capture.
"""
from typing import Optional, Tuple
import cv2
import numpy as np
from pathlib import Path

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAPTURE_KEY,
    EXIT_KEY,
    TEMP_DIR
)
from utils import setup_logger, get_timestamp, cleanup_temp_file

logger = setup_logger(__name__)


class Camera:
    """
    Webcam capture handler with OpenCV.
    Provides real-time video streaming and image capture functionality.
    """
    
    def __init__(self, camera_index: int = CAMERA_INDEX):
        """
        Initialize camera.
        
        Args:
            camera_index: Camera device index (0 for default webcam)
        """
        self.camera_index = camera_index
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        
    def open(self) -> bool:
        """
        Open camera connection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            
            if not self.capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            
            self.is_opened = True
            logger.info(f"Camera {self.camera_index} opened successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read single frame from camera.
        
        Returns:
            Tuple of (success, frame) where frame is BGR image
        """
        if not self.is_opened or self.capture is None:
            logger.error("Camera not opened")
            return False, None
        
        try:
            ret, frame = self.capture.read()
            
            if not ret or frame is None:
                logger.warning("Failed to read frame")
                return False, None
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None
    
    def capture_image(self, save_path: Optional[Path] = None) -> Tuple[bool, Optional[np.ndarray], Optional[Path]]:
        """
        Capture single image from camera.
        
        Args:
            save_path: Optional path to save image
        
        Returns:
            Tuple of (success, image, saved_path)
        """
        ret, frame = self.read_frame()
        
        if not ret or frame is None:
            return False, None, None
        
        # Generate save path if not provided
        if save_path is None:
            timestamp = get_timestamp().replace(':', '-').replace('.', '-')
            save_path = TEMP_DIR / f"capture_{timestamp}.jpg"
        
        # Save image
        try:
            cv2.imwrite(str(save_path), frame)
            logger.info(f"Image captured and saved to {save_path}")
            return True, frame, save_path
            
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False, frame, None
    
    def show_stream(
        self,
        window_name: str = "Camera Feed",
        draw_instructions: bool = True
    ) -> Optional[np.ndarray]:
        """
        Show live camera stream until capture or exit.
        
        Args:
            window_name: Window title
            draw_instructions: Whether to draw instruction text
        
        Returns:
            Captured frame or None if exited without capture
        """
        if not self.is_opened:
            if not self.open():
                return None
        
        logger.info(f"Camera stream started. Press '{CAPTURE_KEY}' to capture, '{EXIT_KEY}' to exit")
        
        while True:
            ret, frame = self.read_frame()
            
            if not ret or frame is None:
                logger.error("Failed to read frame from camera")
                break
            
            # Draw instructions on frame
            if draw_instructions:
                display_frame = frame.copy()
                self._draw_instructions(display_frame)
            else:
                display_frame = frame
            
            cv2.imshow(window_name, display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Capture image
            if key == ord(CAPTURE_KEY):
                logger.info("Capture key pressed")
                cv2.destroyAllWindows()
                return frame
            
            # Exit
            elif key == ord(EXIT_KEY):
                logger.info("Exit key pressed")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        return None
    
    def show_stream_with_callback(
        self,
        callback_func,
        window_name: str = "Camera Feed"
    ) -> None:
        """
        Show live camera stream with frame processing callback.
        
        Args:
            callback_func: Function to process each frame (takes frame, returns processed frame)
            window_name: Window title
        """
        if not self.is_opened:
            if not self.open():
                return
        
        logger.info(f"Camera stream with callback started. Press '{EXIT_KEY}' to exit")
        
        while True:
            ret, frame = self.read_frame()
            
            if not ret or frame is None:
                break
            
            # Apply callback function
            try:
                processed_frame = callback_func(frame)
                cv2.imshow(window_name, processed_frame)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                cv2.imshow(window_name, frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
                break
        
        cv2.destroyAllWindows()
    
    def _draw_instructions(self, frame: np.ndarray) -> None:
        """
        Draw instruction text on frame.
        
        Args:
            frame: Input frame (modified in-place)
        """
        instructions = [
            f"Press '{CAPTURE_KEY}' to capture",
            f"Press '{EXIT_KEY}' to exit"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_offset += 30
    
    def release(self) -> None:
        """Release camera resources."""
        if self.capture is not None:
            self.capture.release()
            self.is_opened = False
            logger.info("Camera released")
        
        cv2.destroyAllWindows()
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.release()


if __name__ == "__main__":
    # Test camera
    print("Testing camera...")
    print(f"Press '{CAPTURE_KEY}' to capture, '{EXIT_KEY}' to exit")
    
    with Camera() as cam:
        frame = cam.show_stream()
        
        if frame is not None:
            print("✓ Frame captured successfully")
            print(f"Frame shape: {frame.shape}")
        else:
            print("✗ No frame captured")

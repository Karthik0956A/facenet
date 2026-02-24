"""
Unit tests for camera module.
Tests webcam capture functionality.
"""
import unittest
import numpy as np
import cv2
from pathlib import Path

from camera import Camera
from config import TEMP_DIR


class TestCamera(unittest.TestCase):
    """Test cases for Camera class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera = Camera()
    
    def tearDown(self):
        """Clean up after tests."""
        if self.camera:
            self.camera.release()
    
    def test_camera_initialization(self):
        """Test camera can be initialized."""
        self.assertIsNotNone(self.camera)
        self.assertEqual(self.camera.camera_index, 0)
        self.assertFalse(self.camera.is_opened)
    
    def test_camera_open(self):
        """Test camera can be opened."""
        success = self.camera.open()
        
        # Camera may not be available in CI/CD
        if success:
            self.assertTrue(self.camera.is_opened)
            self.assertIsNotNone(self.camera.capture)
        else:
            print("Warning: Camera not available for testing")
    
    def test_read_frame_without_opening(self):
        """Test reading frame without opening camera fails gracefully."""
        ret, frame = self.camera.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)
    
    def test_read_frame(self):
        """Test reading frame from camera."""
        if self.camera.open():
            ret, frame = self.camera.read_frame()
            
            if ret:
                self.assertIsNotNone(frame)
                self.assertIsInstance(frame, np.ndarray)
                self.assertEqual(len(frame.shape), 3)  # Height, Width, Channels
                self.assertEqual(frame.shape[2], 3)  # BGR format
        else:
            print("Warning: Camera not available for testing")
    
    def test_capture_image(self):
        """Test capturing and saving image."""
        if self.camera.open():
            test_path = TEMP_DIR / "test_capture.jpg"
            
            ret, image, saved_path = self.camera.capture_image(test_path)
            
            if ret:
                self.assertIsNotNone(image)
                self.assertIsNotNone(saved_path)
                self.assertTrue(saved_path.exists())
                
                # Clean up
                if saved_path.exists():
                    saved_path.unlink()
        else:
            print("Warning: Camera not available for testing")
    
    def test_context_manager(self):
        """Test camera can be used as context manager."""
        with Camera() as cam:
            if cam.is_opened:
                self.assertTrue(cam.is_opened)
        
        # Camera should be released after context
        # This is tested by not throwing errors


class TestCameraIntegration(unittest.TestCase):
    """Integration tests for camera with OpenCV."""
    
    def test_opencv_available(self):
        """Test OpenCV is properly installed."""
        self.assertTrue(hasattr(cv2, 'VideoCapture'))
        self.assertTrue(hasattr(cv2, 'imread'))
        self.assertTrue(hasattr(cv2, 'imwrite'))
    
    def test_temp_directory_exists(self):
        """Test temporary directory is created."""
        self.assertTrue(TEMP_DIR.exists())
        self.assertTrue(TEMP_DIR.is_dir())


if __name__ == '__main__':
    unittest.main()

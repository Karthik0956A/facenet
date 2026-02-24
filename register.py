"""
Face registration module.
Handles new face enrollment into the system.
"""
from typing import Optional, Tuple
import numpy as np

from camera import Camera
from detector import FaceDetector
from embedder import FaceEmbedder
from database import FaceDatabase
from config import TEMP_DIR
from utils import (
    setup_logger,
    validate_name,
    generate_face_id,
    cleanup_temp_file,
    draw_face_box
)

logger = setup_logger(__name__)


class FaceRegistrar:
    """
    Face registration handler.
    Captures face, generates embedding, and stores in database.
    """
    
    def __init__(
        self,
        camera: Optional[Camera] = None,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        db: Optional[FaceDatabase] = None
    ):
        """
        Initialize registrar with optional component injection.
        
        Args:
            camera: Camera instance
            detector: Detector instance
            embedder: Embedder instance
            db: Database instance
        """
        self.camera = camera if camera is not None else Camera()
        self.detector = detector if detector is not None else FaceDetector()
        self.embedder = embedder if embedder is not None else FaceEmbedder()
        self.db = db if db is not None else FaceDatabase()
        
        logger.info("Face registrar initialized")
    
    def register_interactive(self) -> bool:
        """
        Interactive face registration via webcam.
        
        Returns:
            True if registration successful, False otherwise
        """
        print("\n" + "="*60)
        print("FACE REGISTRATION")
        print("="*60)
        
        # Get name
        name = self._get_name_input()
        if not name:
            return False
        
        # Check if name already exists
        existing = self.db.get_face_by_name(name)
        if existing:
            print(f"\n⚠ Warning: Name '{name}' already exists in database.")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Registration cancelled.")
                return False
        
        # Get optional summary
        summary = input("Enter summary/notes (optional): ").strip()
        
        # Capture face
        print("\nStarting camera...")
        print("Position your face in the frame and press 'c' to capture.")
        
        if not self.camera.is_opened:
            self.camera.open()
        
        frame = self.camera.show_stream(window_name="Registration - Press 'c' to capture")
        
        if frame is None:
            print("\n[X] Registration cancelled or camera error.")
            return False
        
        # Process face
        return self._process_and_register(frame, name, summary)
    
    def register_from_image(
        self,
        image: np.ndarray,
        name: str,
        summary: str = ""
    ) -> bool:
        """
        Register face from provided image.
        
        Args:
            image: Input image containing face
            name: Person's name
            summary: Optional summary/notes
        
        Returns:
            True if registration successful, False otherwise
        """
        # Validate name
        if not validate_name(name):
            logger.error(f"Invalid name: {name}")
            return False
        
        return self._process_and_register(image, name, summary)
    
    def _process_and_register(
        self,
        image: np.ndarray,
        name: str,
        summary: str
    ) -> bool:
        """
        Process image and register face.
        
        Args:
            image: Input image
            name: Person's name
            summary: Summary/notes
        
        Returns:
            True if successful, False otherwise
        """
        print("\nProcessing face...")
        
        # Detect face
        face, detection = self.detector.detect_and_extract_largest(image)
        
        if face is None or detection is None:
            print("[X] No face detected in image.")
            logger.error("Face detection failed during registration")
            return False
        
        # Validate detection quality
        if not self.detector.is_valid_detection(detection):
            print("[X] Face detection quality too low.")
            print(f"  Confidence: {detection['confidence']:.3f}")
            return False
        
        print(f"[OK] Face detected (confidence: {detection['confidence']:.3f})")
        
        # Generate embedding
        embedding = self.embedder.generate_embedding(face)
        
        if embedding is None:
            print("[X] Failed to generate face embedding.")
            logger.error("Embedding generation failed during registration")
            return False
        
        # Verify embedding
        if not self.embedder.verify_embedding(embedding):
            print("[X] Invalid face embedding generated.")
            return False
        
        print("[OK] Face embedding generated")
        
        # Generate unique ID
        face_id = generate_face_id()
        
        # Store in database
        success = self.db.insert_face(
            face_id=face_id,
            name=name,
            embedding=embedding,
            summary=summary
        )
        
        if success:
            print(f"\n[OK] Successfully registered: {name}")
            print(f"  Face ID: {face_id}")
            print(f"  Embedding dimension: {len(embedding)}")
            logger.info(f"Face registered successfully: {name} (ID: {face_id})")
            return True
        else:
            print("\n[X] Failed to save to database.")
            logger.error("Database insertion failed during registration")
            return False
    
    def _get_name_input(self) -> Optional[str]:
        """
        Get and validate name input from user.
        
        Returns:
            Validated name or None if invalid/cancelled
        """
        while True:
            name = input("\nEnter name (or 'q' to quit): ").strip()
            
            if name.lower() == 'q':
                print("Registration cancelled.")
                return None
            
            if not name:
                print("⚠ Name cannot be empty.")
                continue
            
            if validate_name(name):
                return name
            else:
                print("⚠ Invalid name. Use 2-50 alphanumeric characters, spaces, hyphens, or underscores.")
    
    def batch_register(self, face_data_list: list) -> Tuple[int, int]:
        """
        Register multiple faces at once.
        
        Args:
            face_data_list: List of dicts with keys: 'image', 'name', 'summary'
        
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        logger.info(f"Starting batch registration of {len(face_data_list)} faces")
        
        for i, face_data in enumerate(face_data_list, 1):
            print(f"\nProcessing {i}/{len(face_data_list)}: {face_data['name']}")
            
            try:
                success = self.register_from_image(
                    image=face_data['image'],
                    name=face_data['name'],
                    summary=face_data.get('summary', '')
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Batch registration error for {face_data['name']}: {e}")
                failed += 1
        
        logger.info(f"Batch registration complete: {successful} successful, {failed} failed")
        
        return successful, failed
    
    def close(self) -> None:
        """Clean up resources."""
        self.camera.release()
        self.db.close()
        logger.info("Registrar resources released")


if __name__ == "__main__":
    # Test registration
    print("Testing face registration...")
    
    try:
        registrar = FaceRegistrar()
        print("[OK] Registrar initialized")
        
        # Run interactive registration
        registrar.register_interactive()
        
        registrar.close()
        
    except Exception as e:
        print(f"[X] Test failed: {e}")
        import traceback
        traceback.print_exc()

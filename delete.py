"""
Face deletion module.
Handles removal of faces from the database.
"""
from typing import List, Optional
import cv2

from database import FaceDatabase
from camera import Camera
from detector import FaceDetector
from embedder import FaceEmbedder
from recognizer import FaceRecognizer
from utils import setup_logger

logger = setup_logger(__name__)


class FaceDeleter:
    """
    Face deletion handler.
    Removes face records from database.
    """
    
    def __init__(self, db: Optional[FaceDatabase] = None):
        """
        Initialize deleter.
        
        Args:
            db: Database instance
        """
        self.db = db if db is not None else FaceDatabase()
        logger.info("Face deleter initialized")
    
    def delete_by_name(self, name: str) -> bool:
        """
        Delete all faces with given name.
        
        Args:
            name: Person's name
        
        Returns:
            True if at least one face deleted, False otherwise
        """
        print(f"\nSearching for faces with name: {name}")
        
        # Check if exists
        face = self.db.get_face_by_name(name)
        
        if face is None:
            print(f"✗ No face found with name: {name}")
            logger.warning(f"Delete failed: name not found - {name}")
            return False
        
        # Confirm deletion
        print(f"\nFound face:")
        print(f"  Name: {face['name']}")
        print(f"  Face ID: {face['face_id']}")
        print(f"  Created: {face['created_at']}")
        if face['summary']:
            print(f"  Summary: {face['summary']}")
        
        response = input("\nAre you sure you want to delete? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("Deletion cancelled.")
            return False
        
        # Delete
        count = self.db.delete_face_by_name(name)
        
        if count > 0:
            print(f"\n✓ Successfully deleted {count} face(s) with name: {name}")
            logger.info(f"Deleted {count} face(s) with name: {name}")
            return True
        else:
            print(f"\n✗ Failed to delete face: {name}")
            return False
    
    def delete_by_id(self, face_id: str) -> bool:
        """
        Delete face by ID.
        
        Args:
            face_id: Unique face identifier
        
        Returns:
            True if deleted, False otherwise
        """
        print(f"\nSearching for face with ID: {face_id}")
        
        # Check if exists
        face = self.db.get_face_by_id(face_id)
        
        if face is None:
            print(f"✗ No face found with ID: {face_id}")
            logger.warning(f"Delete failed: ID not found - {face_id}")
            return False
        
        # Show details
        print(f"\nFound face:")
        print(f"  Name: {face['name']}")
        print(f"  Face ID: {face['face_id']}")
        print(f"  Created: {face['created_at']}")
        
        # Confirm
        response = input("\nAre you sure you want to delete? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("Deletion cancelled.")
            return False
        
        # Delete
        success = self.db.delete_face(face_id)
        
        if success:
            print(f"\n✓ Successfully deleted face: {face['name']}")
            logger.info(f"Deleted face ID: {face_id}")
            return True
        else:
            print(f"\n✗ Failed to delete face")
            return False
    
    def delete_interactive(self) -> bool:
        """
        Interactive deletion - recognize face from camera then delete.
        
        Returns:
            True if deleted, False otherwise
        """
        print("\n" + "="*60)
        print("FACE DELETION")
        print("="*60)
        
        # Get deletion mode
        print("\nDeletion options:")
        print("  1. Delete by name")
        print("  2. Delete by face recognition (camera)")
        print("  3. Cancel")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            name = input("\nEnter name to delete: ").strip()
            if name:
                return self.delete_by_name(name)
            else:
                print("Invalid name.")
                return False
        
        elif choice == '2':
            return self._delete_by_recognition()
        
        else:
            print("Deletion cancelled.")
            return False
    
    def _delete_by_recognition(self) -> bool:
        """
        Delete face by recognizing it from camera.
        
        Returns:
            True if deleted, False otherwise
        """
        print("\nStarting camera for face recognition...")
        print("Position your face in the frame and press 'c' to capture.")
        
        try:
            # Initialize components
            camera = Camera()
            detector = FaceDetector()
            embedder = FaceEmbedder()
            recognizer = FaceRecognizer(self.db)
            
            camera.open()
            
            # Capture frame
            frame = camera.show_stream(window_name="Delete - Press 'c' to capture")
            camera.release()
            
            if frame is None:
                print("\n✗ Capture cancelled.")
                return False
            
            # Detect face
            print("\nDetecting face...")
            face, detection = detector.detect_and_extract_largest(frame)
            
            if face is None:
                print("✗ No face detected.")
                return False
            
            print(f"✓ Face detected (confidence: {detection['confidence']:.3f})")
            
            # Generate embedding
            print("Generating embedding...")
            embedding = embedder.generate_embedding(face)
            
            if embedding is None:
                print("✗ Failed to generate embedding.")
                return False
            
            print("✓ Embedding generated")
            
            # Recognize
            print("Searching database...")
            is_recognized, face_data, confidence = recognizer.recognize(embedding)
            
            if not is_recognized or face_data is None:
                print(f"\n✗ Face not recognized (best match: {confidence:.3f})")
                return False
            
            print(f"\n✓ Face recognized:")
            print(f"  Name: {face_data['name']}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Face ID: {face_data['face_id']}")
            
            # Confirm deletion
            response = input("\nDelete this face? (yes/no): ").strip().lower()
            
            if response != 'yes':
                print("Deletion cancelled.")
                return False
            
            # Delete
            success = self.db.delete_face(face_data['face_id'])
            
            if success:
                print(f"\n✓ Successfully deleted: {face_data['name']}")
                logger.info(f"Deleted face via recognition: {face_data['name']}")
                return True
            else:
                print("\n✗ Failed to delete face.")
                return False
            
        except Exception as e:
            logger.error(f"Delete by recognition failed: {e}")
            print(f"\n✗ Error during deletion: {e}")
            return False
    
    def list_all_faces(self) -> List[dict]:
        """
        List all faces in database.
        
        Returns:
            List of face records
        """
        faces = self.db.get_all_faces()
        
        if not faces:
            print("\nNo faces in database.")
            return []
        
        print(f"\n{'='*60}")
        print(f"{'REGISTERED FACES':<60}")
        print(f"{'='*60}")
        print(f"{'Name':<25} {'Face ID':<36} {'Created':<20}")
        print(f"{'-'*60}")
        
        for face in faces:
            name = face['name'][:24]  # Truncate if too long
            face_id = face['face_id']
            created = face['created_at'][:19]  # Remove milliseconds
            
            print(f"{name:<25} {face_id:<36} {created:<20}")
        
        print(f"{'='*60}")
        print(f"Total: {len(faces)} face(s)")
        
        return faces
    
    def close(self) -> None:
        """Clean up resources."""
        self.db.close()
        logger.info("Deleter resources released")


if __name__ == "__main__":
    # Test deleter
    print("Testing face deleter...")
    
    try:
        deleter = FaceDeleter()
        print("✓ Deleter initialized")
        
        # List all faces
        deleter.list_all_faces()
        
        # Run interactive deletion
        # deleter.delete_interactive()
        
        deleter.close()
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

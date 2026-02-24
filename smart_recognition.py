"""
Smart recognition module with auto-registration.
Recognizes known faces and registers new ones automatically.
"""
from typing import Optional, Tuple
import cv2

from camera import Camera
from detector import FaceDetector
from embedder import FaceEmbedder
from database import FaceDatabase
from recognizer import FaceRecognizer
from utils import (
    setup_logger,
    validate_name,
    generate_face_id,
    draw_face_box
)

logger = setup_logger(__name__)


class SmartRecognition:
    """
    Smart recognition with auto-registration.
    When a face is not recognized, prompts user to register it.
    """
    
    def __init__(self):
        """Initialize all components."""
        self.camera = Camera()
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.db = FaceDatabase()
        self.recognizer = FaceRecognizer(self.db)
        
        logger.info("Smart recognition system initialized")
    
    def run(self) -> bool:
        """
        Run smart recognition workflow.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("SMART RECOGNITION")
        print("="*60)
        print("\nThis mode will:")
        print("  - Recognize your face if already registered")
        print("  - Ask for your name if you're new")
        print("\nStarting camera...")
        
        # Open camera
        if not self.camera.is_opened:
            self.camera.open()
        
        # Capture frame
        print("Position your face in the frame and press 'c' to capture, 'q' to quit")
        frame = self.camera.show_stream(window_name="Smart Recognition - Press 'c'")
        
        if frame is None:
            print("\n✗ Capture cancelled.")
            self.camera.release()
            return False
        
        # Detect face
        print("\n" + "-"*60)
        print("Processing face...")
        face, detection = self.detector.detect_and_extract_largest(frame)
        
        if face is None or detection is None:
            print("✗ No face detected in image.")
            print("  Please try again with better lighting and positioning.")
            self.camera.release()
            return False
        
        print(f"✓ Face detected (confidence: {detection['confidence']:.3f})")
        
        # Validate detection quality
        if not self.detector.is_valid_detection(detection, min_confidence=0.9):
            print("⚠ Face detection quality is low.")
            print(f"  Confidence: {detection['confidence']:.3f} (recommended: >0.9)")
            response = input("  Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                self.camera.release()
                return False
        
        # Generate embedding
        print("Generating face embedding...")
        embedding = self.embedder.generate_embedding(face)
        
        if embedding is None:
            print("✗ Failed to generate face embedding.")
            self.camera.release()
            return False
        
        print("✓ Face embedding generated")
        
        # Try to recognize
        print("Searching database...")
        is_recognized, face_data, confidence = self.recognizer.recognize(embedding)
        
        # Show all similarities for transparency
        print("\n" + "-"*60)
        print("Similarity Analysis:")
        stored_faces = self.db.get_all_faces()
        if stored_faces:
            from utils import batch_cosine_similarity
            stored_embeddings = [f['embedding'] for f in stored_faces]
            similarities = batch_cosine_similarity(embedding, stored_embeddings)
            
            # Sort by similarity (highest first)
            for face, sim in sorted(zip(stored_faces, similarities), key=lambda x: x[1], reverse=True):
                status = "✓ MATCH" if sim >= self.recognizer.recognition_threshold else "✗ Below threshold"
                
                # Add warning for suspicious perfect matches
                if sim >= 0.999:
                    status += " ⚠️ PERFECT MATCH (might be duplicate!)"
                
                print(f"  {face['name']:<20} Similarity: {sim:.6f}  {status}")
            
            print(f"\n  Recognition Threshold: {self.recognizer.recognition_threshold:.4f}")
            print(f"  Total faces in DB: {len(stored_faces)}")
        else:
            print("  Database is empty - this will be the first face!")
        print("-"*60)
        
        print("\n" + "="*60)
        
        if is_recognized and face_data is not None:
            # Face recognized!
            print("✓ FACE RECOGNIZED!")
            print("="*60)
            print(f"\n  Welcome back, {face_data['name']}! 👋")
            print(f"\n  Name:       {face_data['name']}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Level:      {self.recognizer.get_confidence_level(confidence)}")
            print(f"  Face ID:    {face_data['face_id']}")
            
            if face_data.get('summary'):
                print(f"  Summary:    {face_data['summary']}")
            
            print(f"\n  Registered: {face_data['created_at']}")
            
            # Show annotated image
            box = detection['box']
            annotated = draw_face_box(
                frame.copy(),
                box,
                label=face_data['name'],
                confidence=confidence,
                color=(0, 255, 0)
            )
            
            cv2.imshow("Recognized - Press any key to close", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            logger.info(f"Face recognized: {face_data['name']} ({confidence:.3f})")
            
        else:
            # Face not recognized - register it!
            print("✗ FACE NOT RECOGNIZED")
            print("="*60)
            print(f"\n  This is a new face!")
            print(f"  Best match similarity: {confidence:.4f}")
            print(f"  Recognition threshold: {self.recognizer.recognition_threshold:.4f}")
            
            print("\n" + "-"*60)
            print("Let's register this new face...")
            print("-"*60)
            
            # Get name
            name = self._get_name_input()
            
            if name is None:
                print("\nRegistration cancelled.")
                self.camera.release()
                return False
            
            # Get optional summary
            summary = input("\nEnter summary/notes (optional, press Enter to skip): ").strip()
            
            # CRITICAL: Check if this embedding is too similar to existing ones
            if stored_faces:
                from utils import batch_cosine_similarity
                import numpy as np
                
                stored_embeddings = [f['embedding'] for f in stored_faces]
                similarities = batch_cosine_similarity(embedding, stored_embeddings)
                max_similarity = float(np.max(similarities))
                
                if max_similarity >= 0.98:
                    # Embedding is VERY similar to existing one - likely a duplicate!
                    max_idx = int(np.argmax(similarities))
                    similar_face = stored_faces[max_idx]
                    
                    print("\n" + "!"*60)
                    print("⚠️  WARNING: DUPLICATE DETECTION!")
                    print("!"*60)
                    print(f"\nThis face is {max_similarity:.6f} similar to '{similar_face['name']}'")
                    print("This is TOO SIMILAR - likely the SAME PERSON!")
                    print("\nPossible reasons:")
                    print("  1. You're registering the same person twice")
                    print("  2. Very similar photos of different people")
                    print("  3. Bug in embedding generation")
                    print(f"\nExisting face: {similar_face['name']}")
                    print(f"New name: {name}")
                    
                    if similar_face['name'].lower() == name.lower():
                        print("\n✗ ERROR: Same name already exists with identical embedding!")
                        print("  This is definitely a duplicate. Registration cancelled.")
                        self.camera.release()
                        return False
                    
                    print("\n" + "!"*60)
                    response = input("Continue registration anyway? (type 'yes' to continue): ").strip().lower()
                    if response != 'yes':
                        print("\nRegistration cancelled.")
                        print("\nSUGGESTION:")
                        print("  - Make sure you're capturing a DIFFERENT person")
                        print("  - Use better lighting and clear frontal face")
                        print("  - Delete duplicates with: python clear_database.py")
                        self.camera.release()
                        return False
                    else:
                        print("\n⚠️  Proceeding with registration despite high similarity...")
            
            # Register
            success = self._register_face(
                frame=frame,
                face=face,
                detection=detection,
                embedding=embedding,
                name=name,
                summary=summary
            )
            
            if success:
                print(f"\n✓ Successfully registered: {name}")
                print("  Next time you'll be automatically recognized!")
                logger.info(f"New face registered: {name}")
            else:
                print("\n✗ Registration failed.")
                logger.error(f"Failed to register new face: {name}")
        
        print("="*60)
        
        self.camera.release()
        return True
    
    def _get_name_input(self) -> Optional[str]:
        """
        Get and validate name input from user.
        
        Returns:
            Validated name or None if invalid/cancelled
        """
        print("\n📝 Enter your name:")
        
        while True:
            name = input("Name (or 'q' to cancel): ").strip()
            
            if name.lower() == 'q':
                return None
            
            if not name:
                print("⚠ Name cannot be empty. Please try again.")
                continue
            
            if validate_name(name):
                # Check if name already exists
                existing = self.db.get_face_by_name(name)
                if existing:
                    print(f"\n⚠ Warning: Name '{name}' already exists in database.")
                    print("  This might be a different photo of the same person.")
                    response = input("  Use this name anyway? (y/n): ").strip().lower()
                    if response == 'y':
                        return name
                    else:
                        print("  Please enter a different name.")
                        continue
                
                return name
            else:
                print("⚠ Invalid name. Use 2-50 alphanumeric characters, spaces, hyphens, or underscores.")
    
    def _register_face(
        self,
        frame,
        face,
        detection,
        embedding,
        name: str,
        summary: str
    ) -> bool:
        """
        Register face in database.
        
        Args:
            frame: Original captured frame
            face: Extracted face region
            detection: Face detection data
            embedding: Face embedding
            name: Person's name
            summary: Optional summary
        
        Returns:
            True if successful, False otherwise
        """
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
            # Show success visualization
            box = detection['box']
            annotated = draw_face_box(
                frame.copy(),
                box,
                label=f"{name} (NEW)",
                confidence=detection['confidence'],
                color=(0, 255, 0)
            )
            
            cv2.imshow("Registered - Press any key to close", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return success
    
    def close(self) -> None:
        """Clean up resources."""
        self.camera.release()
        self.db.close()
        logger.info("Smart recognition system closed")


if __name__ == "__main__":
    # Test smart recognition
    try:
        smart = SmartRecognition()
        smart.run()
        smart.close()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

"""
Main CLI interface for FaceNet facial recognition system.
Provides command-line interface for registration, recognition, and deletion.
"""
import sys
import argparse
import cv2

from camera import Camera
from detector import FaceDetector
from embedder import FaceEmbedder
from database import FaceDatabase
from recognizer import FaceRecognizer
from register import FaceRegistrar
from delete import FaceDeleter
from smart_recognition import SmartRecognition
from config import validate_config
from utils import setup_logger, draw_face_box

logger = setup_logger(__name__)


def register_face():
    """Register new face via interactive capture."""
    print("\n" + "="*60)
    print("FACE REGISTRATION MODE")
    print("="*60)
    
    try:
        registrar = FaceRegistrar()
        success = registrar.register_interactive()
        registrar.close()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        print(f"\n✗ Registration failed: {e}")
        return 1


def recognize_face():
    """Recognize face via interactive capture."""
    print("\n" + "="*60)
    print("FACE RECOGNITION MODE")
    print("="*60)
    
    try:
        # Initialize components
        camera = Camera()
        detector = FaceDetector()
        embedder = FaceEmbedder()
        recognizer = FaceRecognizer()
        
        print("\nStarting camera...")
        print("Position your face in the frame and press 'c' to capture, 'q' to quit.")
        
        camera.open()
        
        # Capture frame
        frame = camera.show_stream(window_name="Recognition - Press 'c' to capture")
        
        if frame is None:
            print("\n✗ Capture cancelled.")
            camera.release()
            return 1
        
        # Detect face
        print("\nDetecting face...")
        face, detection = detector.detect_and_extract_largest(frame)
        
        if face is None or detection is None:
            print("✗ No face detected in image.")
            camera.release()
            return 1
        
        print(f"✓ Face detected (confidence: {detection['confidence']:.3f})")
        
        # Generate embedding
        print("Generating embedding...")
        embedding = embedder.generate_embedding(face)
        
        if embedding is None:
            print("✗ Failed to generate face embedding.")
            camera.release()
            return 1
        
        print("✓ Face embedding generated")
        
        # Recognize
        print("Searching database...")
        is_recognized, face_data, confidence = recognizer.recognize(embedding)
        
        # Display result
        print("\n" + "="*60)
        print("RECOGNITION RESULT")
        print("="*60)
        
        if is_recognized and face_data is not None:
            print(f"\n✓ FACE RECOGNIZED")
            print(f"\n  Name:       {face_data['name']}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Level:      {recognizer.get_confidence_level(confidence)}")
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
            
            cv2.imshow("Recognition Result - Press any key to close", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:
            print(f"\n✗ FACE NOT RECOGNIZED")
            print(f"\n  Best match similarity: {confidence:.4f}")
            print(f"  Recognition threshold: {recognizer.recognition_threshold:.4f}")
            print(f"\n  This is a new face. Consider registering it.")
        
        print("="*60)
        
        camera.release()
        return 0
        
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        print(f"\n✗ Recognition failed: {e}")
        return 1


def delete_face():
    """Delete face via interactive selection."""
    print("\n" + "="*60)
    print("FACE DELETION MODE")
    print("="*60)
    
    try:
        deleter = FaceDeleter()
        success = deleter.delete_interactive()
        deleter.close()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Deletion error: {e}")
        print(f"\n✗ Deletion failed: {e}")
        return 1


def list_faces():
    """List all registered faces."""
    print("\n" + "="*60)
    print("LIST REGISTERED FACES")
    print("="*60)
    
    try:
        deleter = FaceDeleter()
        deleter.list_all_faces()
        deleter.close()
        return 0
        
    except Exception as e:
        logger.error(f"List faces error: {e}")
        print(f"\n✗ Failed to list faces: {e}")
        return 1


def show_statistics():
    """Show system statistics."""
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)
    
    try:
        recognizer = FaceRecognizer()
        stats = recognizer.get_statistics()
        
        print(f"\n  Total registered faces: {stats['total_faces']}")
        print(f"  Recognition threshold:  {stats['recognition_threshold']:.3f}")
        print(f"  High confidence threshold: {stats['high_confidence_threshold']:.3f}")
        print(f"  Database connected:     {'Yes' if stats['database_connected'] else 'No'}")
        
        print("\n" + "="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        print(f"\n✗ Failed to get statistics: {e}")
        return 1


def smart_recognition():
    """Smart recognition with auto-registration for new faces."""
    try:
        smart = SmartRecognition()
        success = smart.run()
        smart.close()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Smart recognition error: {e}")
        print(f"\n✗ Smart recognition failed: {e}")
        return 1


def continuous_recognition():
    """Continuous face recognition from webcam stream."""
    print("\n" + "="*60)
    print("CONTINUOUS RECOGNITION MODE")
    print("="*60)
    print("\nPress 'q' to quit")
    
    try:
        # Initialize components
        camera = Camera()
        detector = FaceDetector()
        embedder = FaceEmbedder()
        recognizer = FaceRecognizer()
        
        camera.open()
        
        logger.info("Starting continuous recognition")
        
        while True:
            ret, frame = camera.read_frame()
            
            if not ret or frame is None:
                break
            
            # Detect faces
            detections = detector.detect_faces(frame)
            
            # Process each detected face
            for detection in detections:
                # Extract face
                face = detector.extract_face(frame, detection, margin=20)
                
                if face is not None:
                    # Generate embedding
                    embedding = embedder.generate_embedding(face)
                    
                    if embedding is not None:
                        # Recognize
                        is_recognized, face_data, confidence = recognizer.recognize(embedding)
                        
                        # Annotate frame
                        box = detection['box']
                        
                        if is_recognized and face_data is not None:
                            label = face_data['name']
                            color = (0, 255, 0)  # Green for recognized
                        else:
                            label = "Unknown"
                            color = (0, 0, 255)  # Red for unknown
                            confidence = detection['confidence']
                        
                        frame = draw_face_box(
                            frame,
                            box,
                            label=label,
                            confidence=confidence,
                            color=color
                        )
            
            # Display frame
            cv2.imshow("Continuous Recognition - Press 'q' to quit", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        camera.release()
        cv2.destroyAllWindows()
        
        logger.info("Continuous recognition stopped")
        return 0
        
    except Exception as e:
        logger.error(f"Continuous recognition error: {e}")
        print(f"\n✗ Continuous recognition failed: {e}")
        return 1


def main():
    """Main entry point."""
    # Validate configuration
    if not validate_config():
        print("✗ Configuration validation failed")
        return 1
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="FaceNet Facial Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s smart                 Smart mode (auto-register new faces)
  %(prog)s register              Register a new face
  %(prog)s recognize             Recognize a face
  %(prog)s delete                Delete a face
  %(prog)s list                  List all registered faces
  %(prog)s stats                 Show system statistics
  %(prog)s continuous            Continuous recognition mode
        """
    )
    
    parser.add_argument(
        'command',
        choices=['smart', 'register', 'recognize', 'delete', 'list', 'stats', 'continuous'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    # Execute command
    logger.info(f"Executing command: {args.command}")
    
    try:
        if args.command == 'smart':
            return smart_recognition()
        
        elif args.command == 'register':
            return register_face()
        
        elif args.command == 'recognize':
            return recognize_face()
        
        elif args.command == 'delete':
            return delete_face()
        
        elif args.command == 'list':
            return list_faces()
        
        elif args.command == 'stats':
            return show_statistics()
        
        elif args.command == 'continuous':
            return continuous_recognition()
        
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        logger.info("Operation cancelled by user (Ctrl+C)")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

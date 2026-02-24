"""
Quick Registration and Test Script
Tests the fixed system with real camera faces
"""
import sys
import numpy as np
from database import FaceDatabase
from embedder import FaceEmbedder
from detector import FaceDetector
from camera import Camera
from utils import cosine_similarity, setup_logger

logger = setup_logger(__name__)


def main():
    """Register and test with real camera faces."""
    print("="*70)
    print("FACIAL RECOGNITION QUICK TEST")
    print("="*70)
    print()
    print("This script will:")
    print("1. Register Person 1")
    print("2. Register Person 2")
    print("3. Show similarity between them")
    print("4. Test recognition")
    print()
    
    # Initialize components
    print("Initializing components...")
    camera = Camera()
    detector = FaceDetector()
    embedder = FaceEmbedder()
    db = FaceDatabase()
    
    if not camera.is_opened:
        print("[ERROR] Could not open camera")
        return 1
    
    print("[OK] All components ready")
    print()
    
    # Register Person 1
    print("="*70)
    print("STEP 1: Register Person 1")
    print("="*70)
    name1 = input("Enter name for Person 1: ").strip()
    
    if not name1:
        print("[ERROR] Name cannot be empty")
        camera.release()
        return 1
    
    print(f"\nCapturing {name1}...")
    print("Press 'c' to capture, 'q' to quit")
    
    frame1, _ = camera.show_stream("Register Person 1 - Press 'c'")
    
    if frame1 is None:
        print("[ERROR] Failed to capture frame")
        camera.release()
        return 1
    
    # Detect face
    faces1 = detector.detect(frame1)
    if len(faces1) == 0:
        print("[ERROR] No face detected")
        camera.release()
        return 1
    
    # Get largest face
    face1_box = max(faces1, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = face1_box['box']
    face1_img = frame1[y:y+h, x:x+w]
    
    # Generate embedding
    print(f"Generating embedding for {name1}...")
    emb1 = embedder.generate_embedding(face1_img)
    
    if emb1 is None:
        print("[ERROR] Failed to generate embedding")
        camera.release()
        return 1
    
    # Store in database
    from utils import generate_face_id
    face_id1 = generate_face_id()
    success = db.insert_face(face_id1, name1, emb1)
    
    if not success:
        print("[ERROR] Failed to store in database")
        camera.release()
        return 1
    
    print(f"[OK] {name1} registered successfully")
    print(f"    Embedding norm: {np.linalg.norm(emb1):.6f}")
    print(f"    Embedding std: {emb1.std():.6f}")
    print()
    
    # Register Person 2
    print("="*70)
    print("STEP 2: Register Person 2")
    print("="*70)
    name2 = input("Enter name for Person 2: ").strip()
    
    if not name2:
        print("[ERROR] Name cannot be empty")
        camera.release()
        return 1
    
    print(f"\nCapturing {name2}...")
    print("Press 'c' to capture, 'q' to quit")
    
    frame2, _ = camera.show_stream("Register Person 2 - Press 'c'")
    
    if frame2 is None:
        print("[ERROR] Failed to capture frame")
        camera.release()
        return 1
    
    # Detect face
    faces2 = detector.detect(frame2)
    if len(faces2) == 0:
        print("[ERROR] No face detected")
        camera.release()
        return 1
    
    # Get largest face
    face2_box = max(faces2, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = face2_box['box']
    face2_img = frame2[y:y+h, x:x+w]
    
    # Generate embedding
    print(f"Generating embedding for {name2}...")
    emb2 = embedder.generate_embedding(face2_img)
    
    if emb2 is None:
        print("[ERROR] Failed to generate embedding")
        camera.release()
        return 1
    
    # Store in database
    face_id2 = generate_face_id()
    success = db.insert_face(face_id2, name2, emb2)
    
    if not success:
        print("[ERROR] Failed to store in database")
        camera.release()
        return 1
    
    print(f"[OK] {name2} registered successfully")
    print(f"    Embedding norm: {np.linalg.norm(emb2):.6f}")
    print(f"    Embedding std: {emb2.std():.6f}")
    print()
    
    # Compare embeddings
    print("="*70)
    print("STEP 3: Compare Embeddings")
    print("="*70)
    
    similarity = cosine_similarity(emb1, emb2)
    
    print(f"{name1} vs {name2}:")
    print(f"  Similarity: {similarity:.6f}")
    print()
    
    if similarity > 0.95:
        print("[ERROR] VERY HIGH similarity! Different people should be <0.90")
        print("This indicates the bug is still present.")
        status = "FAILED"
    elif similarity > 0.90:
        print("[WARN] HIGH similarity. Check if they are actually different people.")
        print("Current threshold is 0.90, so they might be matched incorrectly.")
        status = "WARNING"
    else:
        print("[OK] Good! Different people have low similarity (<0.90)")
        status = "PASSED"
    
    print()
    
    # Test recognition
    print("="*70)
    print("STEP 4: Test Recognition")
    print("="*70)
    print(f"\nLet's test if we can recognize {name1}...")
    print("Press 'c' to capture, 'q' to quit")
    
    frame_test, _ = camera.show_stream(f"Test Recognition - Show {name1}'s face")
    
    if frame_test is None:
        print("[WARN] Skipping recognition test")
        camera.release()
        db.close()
        return 0 if status == "PASSED" else 1
    
    # Detect face
    faces_test = detector.detect(frame_test)
    if len(faces_test) == 0:
        print("[WARN] No face detected in test")
        camera.release()
        db.close()
        return 0 if status == "PASSED" else 1
    
    # Get largest face
    face_test_box = max(faces_test, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = face_test_box['box']
    face_test_img = frame_test[y:y+h, x:x+w]
    
    # Generate embedding
    print("Generating test embedding...")
    emb_test = embedder.generate_embedding(face_test_img)
    
    if emb_test is None:
        print("[ERROR] Failed to generate test embedding")
        camera.release()
        db.close()
        return 1
    
    # Compare with all stored faces
    print("\nComparing with database:")
    all_faces = db.get_all_faces()
    
    best_match = None
    best_similarity = -1
    
    for face in all_faces:
        face_emb = face['embedding']
        sim = cosine_similarity(emb_test, face_emb)
        print(f"  vs {face['name']}: {sim:.6f}")
        
        if sim > best_similarity:
            best_similarity = sim
            best_match = face
    
    print()
    
    if best_similarity >= 0.90:
        print(f"[OK] Recognized as: {best_match['name']} (confidence: {best_similarity:.6f})")
    else:
        print(f"Not recognized (best match: {best_match['name']} with {best_similarity:.6f})")
    
    print()
    
    # Cleanup
    camera.release()
    db.close()
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Person 1: {name1}")
    print(f"Person 2: {name2}")
    print(f"Similarity: {similarity:.6f}")
    print(f"Status: {status}")
    print()
    
    if status == "PASSED":
        print("[OK] System working correctly!")
        print("Different people have similarity < 0.90")
        return 0
    elif status == "WARNING":
        print("[WARN] Check if people are actually different")
        return 0
    else:
        print("[ERROR] Bug still present!")
        print("Different people should NOT have similarity > 0.95")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

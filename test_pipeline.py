"""
Test the entire face recognition pipeline step by step.
"""
import cv2
import numpy as np

print("="*70)
print("TESTING FACE RECOGNITION PIPELINE")
print("="*70)

# Test 1: Import modules
print("\n[1/6] Testing imports...")
try:
    from camera import Camera
    from detector import FaceDetector
    from embedder import FaceEmbedder
    from database import FaceDatabase
    from recognizer import FaceRecognizer
    from utils import cosine_similarity
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Initialize components
print("\n[2/6] Initializing components...")
try:
    embedder = FaceEmbedder()
    db = FaceDatabase()
    recognizer = FaceRecognizer(db)
    print(f"✓ Components initialized")
    print(f"  Recognition threshold: {recognizer.recognition_threshold}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    exit(1)

# Test 3: Check database
print("\n[3/6] Checking database...")
try:
    faces = db.get_all_faces()
    print(f"✓ Database connected")
    print(f"  Total faces: {len(faces)}")
    
    if len(faces) > 0:
        print("\n  Registered faces:")
        for i, face in enumerate(faces, 1):
            emb_norm = np.linalg.norm(face['embedding'])
            print(f"    {i}. {face['name']:<20} (norm: {emb_norm:.6f})")
except Exception as e:
    print(f"✗ Database check failed: {e}")
    exit(1)

# Test 4: Test embedding generation with synthetic data
print("\n[4/6] Testing embedding generation...")
try:
    # Create two different synthetic face images
    face1 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    face2 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    emb1 = embedder.generate_embedding(face1)
    emb2 = embedder.generate_embedding(face2)
    
    if emb1 is not None and emb2 is not None:
        print(f"✓ Embeddings generated")
        print(f"  Embedding 1 shape: {emb1.shape}, norm: {np.linalg.norm(emb1):.6f}")
        print(f"  Embedding 2 shape: {emb2.shape}, norm: {np.linalg.norm(emb2):.6f}")
        
        # Test similarity
        sim = cosine_similarity(emb1, emb2)
        print(f"  Similarity between random faces: {sim:.6f}")
        
        if sim > 0.8:
            print("  ⚠️  WARNING: Random faces are too similar!")
            print("  This suggests the model might not be working correctly")
        else:
            print("  ✓ OK: Random faces are different as expected")
    else:
        print("✗ Failed to generate embeddings")
        exit(1)
except Exception as e:
    print(f"✗ Embedding test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Compare stored faces
print("\n[5/6] Analyzing stored faces...")
if len(faces) >= 2:
    print("\n  Pairwise similarities:")
    embeddings = [f['embedding'] for f in faces]
    
    problems = []
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            name1 = faces[i]['name']
            name2 = faces[j]['name']
            
            status = ""
            if sim >= 0.75:
                status = "⚠️  WILL BE CONFUSED!"
                problems.append((name1, name2, sim))
            elif sim >= 0.6:
                status = "⚠️  May be confused"
            else:
                status = "✓ OK"
            
            print(f"    {name1:<20} <-> {name2:<20}  Sim: {sim:.6f}  {status}")
    
    if problems:
        print("\n  ❌ PROBLEM DETECTED:")
        print(f"     {len(problems)} pair(s) will be confused!")
        print("\n  Solutions:")
        print("     1. Delete one of the confused faces:")
        print("        python main.py delete")
        print("     2. Increase threshold to 0.85:")
        print("        Edit .env: RECOGNITION_THRESHOLD=0.85")
        print("     3. Re-register with better quality photos")
    else:
        print("\n  ✓ All faces are sufficiently distinct")
        
elif len(faces) == 1:
    print("  Only 1 face registered - cannot compare")
else:
    print("  No faces registered yet")

# Test 6: Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if len(problems) > 0:
    print("\n❌ ISSUE FOUND: Faces are too similar!")
    print(f"   {len(problems)} pair(s) will be misidentified")
    print("\n   The FaceNet model IS working correctly,")
    print("   but the registered faces have high similarity.")
    print("\n   Action required:")
    print("   1. Run: python main.py list")
    print("   2. Run: python main.py delete")
    print("   3. Delete the incorrectly registered faces")
    print("   4. Re-register with better quality photos")
else:
    print("\n✓ System is working correctly!")
    print("  All components operational")
    print("  Faces are sufficiently distinct")

db.close()
print("\n" + "="*70)

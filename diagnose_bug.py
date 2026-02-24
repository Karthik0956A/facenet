"""
Emergency diagnostic - Check if embeddings are actually different.
"""
import numpy as np
from embedder import FaceEmbedder
from database import FaceDatabase
from utils import cosine_similarity
import cv2

print("="*70)
print("EMERGENCY DIAGNOSTIC - CHECKING EMBEDDING GENERATION")
print("="*70)

# Test 1: Check if model generates different embeddings for different images
print("\n[TEST 1] Checking if FaceNet model generates DIFFERENT embeddings...")

embedder = FaceEmbedder()

# Create 3 completely different random "faces"
face1 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
face2 = np.random.randint(100, 200, (160, 160, 3), dtype=np.uint8)
face3 = np.full((160, 160, 3), 128, dtype=np.uint8)

emb1 = embedder.generate_embedding(face1)
emb2 = embedder.generate_embedding(face2)
emb3 = embedder.generate_embedding(face3)

print(f"\nEmbedding 1 sample: {emb1[:5]}")
print(f"Embedding 2 sample: {emb2[:5]}")
print(f"Embedding 3 sample: {emb3[:5]}")

# Check if they're all the same (BUG!)
if np.allclose(emb1, emb2, atol=0.001) and np.allclose(emb2, emb3, atol=0.001):
    print("\n❌ CRITICAL BUG FOUND!")
    print("   FaceNet is generating IDENTICAL embeddings for different images!")
    print("   This is why all faces are recognized as the same person.")
    print("\n   Root cause: FaceNet model problem")
else:
    sim12 = cosine_similarity(emb1, emb2)
    sim23 = cosine_similarity(emb2, emb3)
    sim13 = cosine_similarity(emb1, emb3)
    
    print(f"\n✓ Embeddings are DIFFERENT:")
    print(f"  Similarity 1-2: {sim12:.6f}")
    print(f"  Similarity 2-3: {sim23:.6f}")
    print(f"  Similarity 1-3: {sim13:.6f}")
    
    if sim12 < 0.3 and sim23 < 0.3 and sim13 < 0.3:
        print("\n✓ FaceNet model is working correctly!")
        print("  Different images produce different embeddings.")
    else:
        print("\n⚠️  Embeddings are unexpectedly similar")

# Test 2: Check database embeddings
print("\n" + "="*70)
print("[TEST 2] Checking stored embeddings in database...")

db = FaceDatabase()
faces = db.get_all_faces()

print(f"\nFaces in database: {len(faces)}")

if len(faces) == 0:
    print("  No faces to check")
elif len(faces) == 1:
    face = faces[0]
    print(f"\n  Name: {face['name']}")
    print(f"  Embedding sample: {face['embedding'][:5]}")
    print(f"  Embedding norm: {np.linalg.norm(face['embedding']):.6f}")
else:
    print("\nStored embeddings:")
    for i, face in enumerate(faces, 1):
        emb = face['embedding']
        print(f"\n  {i}. {face['name']}")
        print(f"     Sample values: {emb[:5]}")
        print(f"     Norm: {np.linalg.norm(emb):.6f}")
    
    # Check if stored embeddings are identical (BUG!)
    print("\n" + "-"*70)
    print("Comparing stored embeddings:")
    
    embeddings = [f['embedding'] for f in faces]
    all_same = True
    
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            name1 = faces[i]['name']
            name2 = faces[j]['name']
            
            print(f"\n  {name1} <-> {name2}")
            print(f"    Similarity: {sim:.6f}")
            
            if sim >= 0.99:
                print(f"    ❌ IDENTICAL! This is the bug!")
                print(f"    These are the SAME embedding stored twice")
            elif sim >= 0.9:
                print(f"    ❌ Almost identical! ({sim:.6f})")
            elif sim >= 0.75:
                print(f"    ⚠️  Too similar - will be confused")
                all_same = False
            else:
                print(f"    ✓ Different enough")
                all_same = False
    
    if all_same and len(faces) > 1:
        print("\n" + "="*70)
        print("❌ ROOT CAUSE FOUND!")
        print("="*70)
        print("\nAll stored embeddings are IDENTICAL or nearly identical.")
        print("This means:")
        print("  1. You registered the SAME PERSON multiple times, OR")
        print("  2. The photos were so similar they produced same embedding, OR")
        print("  3. There's a bug in the registration process")
        print("\nSOLUTION:")
        print("  1. Delete all faces: python clear_database.py")
        print("  2. Register FIRST person with good lighting")
        print("  3. Register SECOND person (DIFFERENT person!)")
        print("  4. Make sure they're actually different people!")

db.close()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

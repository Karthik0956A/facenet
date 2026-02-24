"""
Simple database check - no external dependencies except pymongo.
"""
from pymongo import MongoClient
import numpy as np

print("="*70)
print("CHECKING DATABASE EMBEDDINGS")
print("="*70)

client = MongoClient("mongodb://localhost:27017/")
db = client["facenet_db"]
collection = db["faces"]

faces = list(collection.find())

print(f"\nTotal faces in database: {len(faces)}")

if len(faces) == 0:
    print("No faces found in database.")
elif len(faces) == 1:
    face = faces[0]
    emb = np.array(face['embedding'])
    print(f"\nSingle face:")
    print(f"  Name: {face['name']}")
    print(f"  Face ID: {face['face_id']}")
    print(f"  Embedding shape: {emb.shape}")
    print(f"  First 10 values: {emb[:10]}")
    print(f"  Norm: {np.linalg.norm(emb):.6f}")
else:
    print("\nStored faces:")
    for i, face in enumerate(faces, 1):
        emb = np.array(face['embedding'])
        print(f"\n{i}. Name: {face['name']}")
        print(f"   Face ID: {face['face_id']}")
        print(f"   Embedding shape: {emb.shape}")
        print(f"   First 10 values: {emb[:10]}")
        print(f"   Norm: {np.linalg.norm(emb):.6f}")
    
    # Compare embeddings
    print("\n" + "="*70)
    print("COMPARING EMBEDDINGS")
    print("="*70)
    
    embeddings = [np.array(f['embedding']) for f in faces]
    
    # Cosine similarity
    def cosine_sim(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        return np.dot(a_norm, b_norm)
    
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            emb1 = embeddings[i]
            emb2 = embeddings[j]
            name1 = faces[i]['name']
            name2 = faces[j]['name']
            
            # Check if exactly the same
            are_identical = np.array_equal(emb1, emb2)
            sim = cosine_sim(emb1, emb2)
            
            print(f"\n{name1} vs {name2}:")
            print(f"  Identical: {are_identical}")
            print(f"  Cosine similarity: {sim:.10f}")
            
            if are_identical:
                print(f"  ❌ EMBEDDINGS ARE EXACTLY THE SAME!")
                print(f"     This is why both are recognized as the same person.")
            elif sim >= 0.99:
                print(f"  ❌ Nearly identical (>0.99)")
            elif sim >= 0.9:
                print(f"  ⚠️  Very similar (>0.9)")
            elif sim >= 0.75:
                print(f"  ⚠️  Similar enough to confuse (>0.75)")
            else:
                print(f"  ✓ Different enough")

client.close()

print("\n" + "="*70)

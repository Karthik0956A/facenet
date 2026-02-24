"""
Simple diagnostic to check why faces are being misidentified.
"""
import sys
sys.path.insert(0, r'D:\miniproj\facenet_project')

try:
    from database import FaceDatabase
    from utils import cosine_similarity
    import numpy as np
    
    print("\n" + "="*70)
    print("CHECKING REGISTERED FACES")
    print("="*70)
    
    db = FaceDatabase()
    faces = db.get_all_faces()
    
    print(f"\nTotal faces in database: {len(faces)}")
    
    if len(faces) == 0:
        print("\n✗ No faces registered yet!")
        sys.exit(1)
    
    print("\nRegistered faces:")
    for i, face in enumerate(faces, 1):
        print(f"  {i}. {face['name']}")
        print(f"     ID: {face['face_id'][:8]}...")
        print(f"     Embedding shape: {face['embedding'].shape}")
        print(f"     Embedding norm: {np.linalg.norm(face['embedding']):.6f}")
    
    if len(faces) >= 2:
        print("\n" + "="*70)
        print("SIMILARITY MATRIX")
        print("="*70)
        
        for i, face1 in enumerate(faces):
            for j, face2 in enumerate(faces):
                if i < j:
                    sim = cosine_similarity(face1['embedding'], face2['embedding'])
                    print(f"\n{face1['name']} <-> {face2['name']}")
                    print(f"  Similarity: {sim:.6f}")
                    
                    if sim >= 0.75:
                        print(f"  ⚠️  WARNING: Too similar! (threshold=0.75)")
                        print(f"     These will be CONFUSED!")
                    elif sim >= 0.6:
                        print(f"  ⚠️  CAUTION: Moderately similar")
                    else:
                        print(f"  ✓ OK: Sufficiently different")
    
    db.close()
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

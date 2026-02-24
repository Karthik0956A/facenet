"""
Debug script to test and verify face recognition accuracy.
Shows detailed similarity scores between faces.
"""
import sys
import numpy as np
from database import FaceDatabase
from utils import cosine_similarity, batch_cosine_similarity, setup_logger

logger = setup_logger(__name__)


def check_database_faces():
    """Check all faces in database and show similarity matrix."""
    print("\n" + "="*70)
    print("FACE DATABASE VERIFICATION")
    print("="*70)
    
    try:
        db = FaceDatabase()
        faces = db.get_all_faces()
        
        if not faces:
            print("\n✗ No faces in database!")
            print("  Register some faces first with: python main.py smart")
            return 1
        
        print(f"\n✓ Found {len(faces)} face(s) in database:")
        for i, face in enumerate(faces, 1):
            print(f"  {i}. {face['name']} (ID: {face['face_id'][:8]}...)")
        
        # Create similarity matrix
        print("\n" + "="*70)
        print("SIMILARITY MATRIX (Cosine Similarity)")
        print("="*70)
        print("Values range from -1 to 1, where:")
        print("  1.0  = Identical (same person)")
        print("  0.75-0.95 = Likely same person")
        print("  0.5-0.75  = Similar features (different people)")
        print("  <0.5 = Very different")
        print("-"*70)
        
        # Header
        print(f"\n{'Name':<20}", end="")
        for face in faces:
            print(f"{face['name'][:15]:<18}", end="")
        print()
        print("-"*70)
        
        # Similarity matrix
        embeddings = [face['embedding'] for face in faces]
        
        for i, face1 in enumerate(faces):
            print(f"{face1['name']:<20}", end="")
            
            for j, face2 in enumerate(faces):
                if i == j:
                    # Same face
                    print(f"{'1.0000':<18}", end="")
                else:
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    color_code = ""
                    if sim >= 0.75:
                        color_code = " ⚠️"  # Warning: might be too similar
                    print(f"{sim:.4f}{color_code:<14}", end="")
            print()
        
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        # Check for problematic similarities
        problems = []
        for i, face1 in enumerate(faces):
            for j, face2 in enumerate(faces):
                if i < j:  # Avoid duplicates
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    if sim >= 0.75:
                        problems.append((face1['name'], face2['name'], sim))
        
        if problems:
            print("\n⚠️  HIGH SIMILARITY DETECTED:")
            print("These faces are very similar and might be confused:")
            for name1, name2, sim in problems:
                print(f"  • {name1} ↔ {name2}: {sim:.4f}")
            print("\nRecommendations:")
            print("  1. If these are different people, consider:")
            print("     - Re-registering with better lighting")
            print("     - Using different angles")
            print("     - Increasing RECOGNITION_THRESHOLD in .env")
            print("  2. If these are the same person with different names:")
            print("     - Delete one: python main.py delete")
        else:
            print("\n✓ All faces are sufficiently distinct!")
            print("  No confusion expected between registered faces.")
        
        print("\n" + "="*70)
        print("CURRENT SETTINGS")
        print("="*70)
        
        from config import RECOGNITION_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD
        print(f"Recognition Threshold:      {RECOGNITION_THRESHOLD:.4f}")
        print(f"High Confidence Threshold:  {HIGH_CONFIDENCE_THRESHOLD:.4f}")
        
        print("\nThreshold Guidelines:")
        print("  • >= 0.85: Very strict (best for similar looking people)")
        print("  • 0.75-0.85: Strict (recommended default)")
        print("  • 0.6-0.75: Moderate (may confuse similar faces)")
        print("  • < 0.6: Lenient (high false positive rate)")
        
        db.close()
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_specific_faces(name1: str, name2: str):
    """Test similarity between two specific faces."""
    print(f"\n" + "="*70)
    print(f"COMPARING: {name1} vs {name2}")
    print("="*70)
    
    try:
        db = FaceDatabase()
        
        face1 = db.get_face_by_name(name1)
        face2 = db.get_face_by_name(name2)
        
        if not face1:
            print(f"\n✗ Face not found: {name1}")
            return 1
        
        if not face2:
            print(f"\n✗ Face not found: {name2}")
            return 1
        
        similarity = cosine_similarity(face1['embedding'], face2['embedding'])
        
        print(f"\nCosine Similarity: {similarity:.6f}")
        
        from config import RECOGNITION_THRESHOLD
        
        if similarity >= RECOGNITION_THRESHOLD:
            print(f"Status: ⚠️  WOULD BE CONFUSED (above threshold {RECOGNITION_THRESHOLD})")
            print("\nThese faces are too similar!")
            print("Consider:")
            print(f"  • Increasing threshold to at least {similarity + 0.05:.2f}")
            print("  • Re-registering with better photos")
            print("  • Deleting duplicate entries")
        else:
            print(f"Status: ✓ DISTINCT (below threshold {RECOGNITION_THRESHOLD})")
            print("\nThese faces can be distinguished correctly.")
        
        db.close()
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - show all faces
        sys.exit(check_database_faces())
    
    elif len(sys.argv) == 3:
        # Two arguments - compare specific faces
        sys.exit(test_specific_faces(sys.argv[1], sys.argv[2]))
    
    else:
        print("Usage:")
        print("  python verify_accuracy.py              - Check all faces")
        print("  python verify_accuracy.py NAME1 NAME2  - Compare two faces")
        sys.exit(1)

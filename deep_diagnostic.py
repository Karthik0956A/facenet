"""
COMPREHENSIVE DIAGNOSTIC SCRIPT
Detects root cause of identical embeddings bug
"""
import numpy as np
import cv2
import hashlib
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from database import FaceDatabase
from embedder import FaceEmbedder
from detector import FaceDetector
from camera import Camera
from utils import cosine_similarity, batch_cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_1_camera_frame_uniqueness():
    """Test if camera returns unique frames"""
    print("\n" + "="*80)
    print("TEST 1: Camera Frame Uniqueness")
    print("="*80)
    
    camera = Camera()
    camera.open()
    
    hashes = []
    for i in range(5):
        input(f"\nMove your face/hand, then press Enter to capture frame {i+1}/5...")
        frame = camera.read_frame()
        if frame is not None:
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
            hashes.append(frame_hash)
            print(f"Frame {i+1} hash: {frame_hash}")
    
    camera.release()
    
    unique_hashes = len(set(hashes))
    print(f"\nResult: {unique_hashes}/{len(hashes)} unique frames")
    
    if unique_hashes < len(hashes):
        print("❌ FAIL: Camera returning duplicate frames!")
        print("   This could cause identical embeddings.")
        return False
    else:
        print("✓ PASS: Camera returns unique frames")
        return True


def test_2_face_extraction_uniqueness():
    """Test if MTCNN extracts different faces"""
    print("\n" + "="*80)
    print("TEST 2: Face Extraction Uniqueness")
    print("="*80)
    
    camera = Camera()
    detector = FaceDetector()
    camera.open()
    
    faces = []
    hashes = []
    
    for i in range(3):
        input(f"\nShow face (change expression/angle), press Enter to capture {i+1}/3...")
        frame = camera.read_frame()
        if frame is not None:
            face, detection = detector.detect_and_extract_largest(frame)
            if face is not None:
                face_hash = hashlib.md5(face.tobytes()).hexdigest()
                hashes.append(face_hash)
                faces.append(face)
                print(f"Face {i+1}:")
                print(f"  Shape: {face.shape}")
                print(f"  Hash: {face_hash}")
                print(f"  BBox: {detection['box']}")
                print(f"  Mean pixel: {face.mean():.3f}")
                print(f"  Std pixel: {face.std():.3f}")
    
    camera.release()
    
    unique_hashes = len(set(hashes))
    print(f"\nResult: {unique_hashes}/{len(hashes)} unique face crops")
    
    if unique_hashes < len(hashes):
        print("❌ FAIL: Face extraction producing identical crops!")
        print("   MTCNN or extraction logic has a bug.")
        return False
    else:
        print("✓ PASS: Face extraction produces unique crops")
        return True


def test_3_preprocessing_uniqueness():
    """Test if face preprocessing produces unique outputs"""
    print("\n" + "="*80)
    print("TEST 3: Preprocessing Uniqueness")
    print("="*80)
    
    embedder = FaceEmbedder()
    camera = Camera()
    detector = FaceDetector()
    camera.open()
    
    preprocessed_faces = []
    hashes = []
    
    for i in range(3):
        input(f"\nShow face, press Enter to capture {i+1}/3...")
        frame = camera.read_frame()
        if frame is not None:
            face, _ = detector.detect_and_extract_largest(frame)
            if face is not None:
                # Preprocess
                preprocessed = embedder.preprocess_face(face)
                preprocessed_hash = hashlib.md5(preprocessed.tobytes()).hexdigest()
                hashes.append(preprocessed_hash)
                preprocessed_faces.append(preprocessed)
                
                print(f"Preprocessed face {i+1}:")
                print(f"  Shape: {preprocessed.shape}")
                print(f"  Dtype: {preprocessed.dtype}")
                print(f"  Range: [{preprocessed.min():.6f}, {preprocessed.max():.6f}]")
                print(f"  Mean: {preprocessed.mean():.6f}")
                print(f"  Std: {preprocessed.std():.6f}")
                print(f"  Hash: {preprocessed_hash}")
    
    camera.release()
    
    unique_hashes = len(set(hashes))
    print(f"\nResult: {unique_hashes}/{len(hashes)} unique preprocessed faces")
    
    if unique_hashes < len(hashes):
        print("❌ FAIL: Preprocessing producing identical outputs!")
        print("   Preprocessing pipeline has a bug.")
        return False
    else:
        print("✓ PASS: Preprocessing produces unique outputs")
        return True


def test_4_embedding_uniqueness():
    """Test if FaceNet generates unique embeddings"""
    print("\n" + "="*80)
    print("TEST 4: Embedding Uniqueness (CRITICAL)")
    print("="*80)
    
    embedder = FaceEmbedder()
    camera = Camera()
    detector = FaceDetector()
    camera.open()
    
    embeddings = []
    hashes = []
    
    for i in range(3):
        input(f"\nShow face #{i+1}, press Enter to generate embedding {i+1}/3...")
        frame = camera.read_frame()
        if frame is not None:
            face, _ = detector.detect_and_extract_largest(frame)
            if face is not None:
                # Generate embedding
                embedding = embedder.generate_embedding(face)
                if embedding is not None:
                    emb_hash = hashlib.sha256(embedding.tobytes()).hexdigest()[:16]
                    hashes.append(emb_hash)
                    embeddings.append(embedding)
                    
                    print(f"\nEmbedding {i+1}:")
                    print(f"  Shape: {embedding.shape}")
                    print(f"  Dtype: {embedding.dtype}")
                    print(f"  Norm: {np.linalg.norm(embedding):.10f}")
                    print(f"  Mean: {embedding.mean():.10f}")
                    print(f"  Std: {embedding.std():.10f}")
                    print(f"  Min: {embedding.min():.10f}")
                    print(f"  Max: {embedding.max():.10f}")
                    print(f"  First 10: {embedding[:10]}")
                    print(f"  Hash: {emb_hash}")
    
    camera.release()
    
    # Check uniqueness
    unique_hashes = len(set(hashes))
    print(f"\n{'='*80}")
    print(f"Result: {unique_hashes}/{len(hashes)} unique embeddings")
    print(f"{'='*80}")
    
    if unique_hashes < len(hashes):
        print("❌ CRITICAL FAIL: FaceNet generating IDENTICAL embeddings!")
        print("   This is the ROOT CAUSE of your bug.")
        print("\n   Possible reasons:")
        print("   1. Model returning cached output")
        print("   2. Preprocessing producing identical inputs")
        print("   3. Model weights not loaded correctly")
        print("   4. Batch dimension issue")
        return False, embeddings
    else:
        print("✓ PASS: FaceNet generates unique embeddings")
        
        # Check similarities
        if len(embeddings) >= 2:
            print("\nPairwise similarities (same person):")
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = cosine_similarity(embeddings[i], embeddings[j])
                    print(f"  Embedding {i+1} vs {j+1}: {sim:.6f}")
                    if sim >= 0.98:
                        print("    ⚠️  Too similar (might be bug if these are different people)")
        
        return True, embeddings


def test_5_database_storage_integrity():
    """Test if database preserves embedding precision"""
    print("\n" + "="*80)
    print("TEST 5: Database Storage Integrity")
    print("="*80)
    
    db = FaceDatabase()
    
    # Create test embedding
    test_embedding = np.random.randn(512).astype(np.float32)
    test_embedding = test_embedding / np.linalg.norm(test_embedding)
    
    test_id = "test_embedding_integrity_check"
    test_name = "TestPerson_DeleteMe"
    
    print("Original embedding:")
    print(f"  Shape: {test_embedding.shape}")
    print(f"  Dtype: {test_embedding.dtype}")
    print(f"  Norm: {np.linalg.norm(test_embedding):.10f}")
    print(f"  First 10: {test_embedding[:10]}")
    print(f"  Hash: {hashlib.sha256(test_embedding.tobytes()).hexdigest()[:16]}")
    
    # Store
    success = db.insert_face(test_id, test_name, test_embedding)
    
    if not success:
        print("❌ FAIL: Could not insert test face")
        db.close()
        return False
    
    # Retrieve
    retrieved = db.get_face_by_id(test_id)
    
    if retrieved is None:
        print("❌ FAIL: Could not retrieve test face")
        db.delete_face(test_id)
        db.close()
        return False
    
    retrieved_embedding = np.array(retrieved['embedding'], dtype=np.float32)
    
    print("\nRetrieved embedding:")
    print(f"  Shape: {retrieved_embedding.shape}")
    print(f"  Dtype: {retrieved_embedding.dtype}")
    print(f"  Norm: {np.linalg.norm(retrieved_embedding):.10f}")
    print(f"  First 10: {retrieved_embedding[:10]}")
    print(f"  Hash: {hashlib.sha256(retrieved_embedding.tobytes()).hexdigest()[:16]}")
    
    # Compare
    max_error = np.abs(test_embedding - retrieved_embedding).max()
    mean_error = np.abs(test_embedding - retrieved_embedding).mean()
    
    print(f"\nPrecision loss:")
    print(f"  Max error: {max_error:.15f}")
    print(f"  Mean error: {mean_error:.15f}")
    
    # Cleanup
    db.delete_face(test_id)
    db.close()
    
    if max_error > 1e-6:
        print("❌ FAIL: Significant precision loss during storage!")
        print("   Database storage/retrieval has a bug.")
        return False
    else:
        print("✓ PASS: Database preserves embedding precision")
        return True


def test_6_database_duplicate_check():
    """Check if database contains duplicate embeddings"""
    print("\n" + "="*80)
    print("TEST 6: Database Duplicate Detection")
    print("="*80)
    
    db = FaceDatabase()
    faces = db.get_all_faces()
    
    print(f"Found {len(faces)} faces in database")
    
    if len(faces) < 2:
        print("Need at least 2 faces to check for duplicates")
        db.close()
        return True
    
    # Extract embeddings
    embeddings = []
    names = []
    for face in faces:
        emb = np.array(face['embedding'], dtype=np.float32)
        embeddings.append(emb)
        names.append(face['name'])
        print(f"\n{face['name']}:")
        print(f"  Norm: {np.linalg.norm(emb):.6f}")
        print(f"  Mean: {emb.mean():.6f}")
        print(f"  Std: {emb.std():.6f}")
        print(f"  First 5: {emb[:5]}")
        print(f"  Hash: {hashlib.sha256(emb.tobytes()).hexdigest()[:16]}")
    
    # Compute pairwise similarities
    print(f"\n{'='*80}")
    print("PAIRWISE SIMILARITY MATRIX")
    print(f"{'='*80}\n")
    
    n = len(embeddings)
    duplicates_found = False
    
    for i in range(n):
        for j in range(i+1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            
            status = ""
            if sim >= 0.999:
                status = "❌ IDENTICAL (bug!)"
                duplicates_found = True
            elif sim >= 0.98:
                status = "❌ DUPLICATE"
                duplicates_found = True
            elif sim >= 0.90:
                status = "⚠️  TOO SIMILAR"
            elif sim >= 0.70:
                status = "✓ Reasonable"
            else:
                status = "✓ Different"
            
            print(f"{names[i]:20} <-> {names[j]:20}  {sim:.6f}  {status}")
    
    db.close()
    
    if duplicates_found:
        print("\n" + "="*80)
        print("❌ CRITICAL: DUPLICATE EMBEDDINGS FOUND IN DATABASE!")
        print("="*80)
        print("\nThis is the ROOT CAUSE of your recognition bug.")
        print("\nSolutions:")
        print("  1. Run: python clear_database.py")
        print("  2. Re-register each person with DIFFERENT photos")
        print("  3. Ensure faces are actually from different people")
        print("  4. Check if preprocessing/embedding generation has bugs (Tests 1-4)")
        return False
    else:
        print("\n✓ PASS: No duplicate embeddings found")
        return True


def test_7_cosine_similarity_correctness():
    """Test if cosine similarity calculation is correct"""
    print("\n" + "="*80)
    print("TEST 7: Cosine Similarity Correctness")
    print("="*80)
    
    # Test 1: Identical vectors should give 1.0
    v1 = np.array([1, 0, 0], dtype=np.float32)
    sim = cosine_similarity(v1, v1)
    print(f"Test 1 - Identical vectors: {sim:.10f} (expected: 1.0)")
    if not np.isclose(sim, 1.0):
        print("❌ FAIL")
        return False
    
    # Test 2: Orthogonal vectors should give 0.0
    v2 = np.array([1, 0, 0], dtype=np.float32)
    v3 = np.array([0, 1, 0], dtype=np.float32)
    sim = cosine_similarity(v2, v3)
    print(f"Test 2 - Orthogonal vectors: {sim:.10f} (expected: 0.0)")
    if not np.isclose(sim, 0.0):
        print("❌ FAIL")
        return False
    
    # Test 3: Opposite vectors should give 0.0 (or -1.0 if not abs)
    v4 = np.array([1, 0, 0], dtype=np.float32)
    v5 = np.array([-1, 0, 0], dtype=np.float32)
    sim = cosine_similarity(v4, v5)
    print(f"Test 3 - Opposite vectors: {sim:.10f} (expected: 0.0 or close)")
    
    # Test 4: Random normalized vectors
    v6 = np.random.randn(512).astype(np.float32)
    v6 = v6 / np.linalg.norm(v6)
    v7 = np.random.randn(512).astype(np.float32)
    v7 = v7 / np.linalg.norm(v7)
    sim = cosine_similarity(v6, v7)
    print(f"Test 4 - Random vectors: {sim:.10f} (expected: 0.0-1.0)")
    if sim < 0.0 or sim > 1.0:
        print(f"❌ FAIL: Similarity out of range!")
        return False
    
    print("✓ PASS: Cosine similarity calculation is correct")
    return True


def test_8_model_determinism():
    """Test if model returns same embedding for same input"""
    print("\n" + "="*80)
    print("TEST 8: Model Determinism")
    print("="*80)
    
    embedder = FaceEmbedder()
    
    # Create fixed test image (black square)
    test_image = np.zeros((160, 160, 3), dtype=np.uint8)
    
    print("Generating embeddings from identical input (3 times)...")
    embeddings = []
    for i in range(3):
        emb = embedder.generate_embedding(test_image)
        if emb is not None:
            embeddings.append(emb)
            print(f"\nEmbedding {i+1}:")
            print(f"  First 5: {emb[:5]}")
            print(f"  Norm: {np.linalg.norm(emb):.10f}")
            print(f"  Hash: {hashlib.sha256(emb.tobytes()).hexdigest()[:16]}")
    
    if len(embeddings) != 3:
        print("❌ FAIL: Could not generate all embeddings")
        return False
    
    # Check if all identical
    all_identical = True
    for i in range(1, len(embeddings)):
        if not np.allclose(embeddings[0], embeddings[i], atol=1e-6):
            all_identical = False
            diff = np.abs(embeddings[0] - embeddings[i]).max()
            print(f"Embedding 1 vs {i+1}: max diff = {diff:.15f}")
    
    if all_identical:
        print("✓ PASS: Model is deterministic (same input → same output)")
        return True
    else:
        print("⚠️  WARNING: Model output varies for same input")
        print("   This could be due to dropout layers or other non-deterministic operations")
        return True  # Not necessarily a fail


def run_all_tests():
    """Run comprehensive diagnostic battery"""
    print("\n" + "="*80)
    print("COMPREHENSIVE DIAGNOSTIC TEST SUITE")
    print("Debugging: Different people recognized as same with similarity ≈ 1.00")
    print("="*80)
    
    results = {}
    
    # Run all tests
    results['test_1'] = test_1_camera_frame_uniqueness()
    results['test_2'] = test_2_face_extraction_uniqueness()
    results['test_3'] = test_3_preprocessing_uniqueness()
    results['test_4'], embeddings = test_4_embedding_uniqueness()
    results['test_5'] = test_5_database_storage_integrity()
    results['test_6'] = test_6_database_duplicate_check()
    results['test_7'] = test_7_cosine_similarity_correctness()
    results['test_8'] = test_8_model_determinism()
    
    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    failures =[ k for k, v in results.items() if not v]
    
    if failures:
        print("\n" + "="*80)
        print("ROOT CAUSE IDENTIFIED:")
        print("="*80)
        for fail in failures:
            print(f"  - {fail}")
        print("\nRefer to the specific test output above for details and solutions.")
    else:
        print("\n✓ All tests passed!")
        print("If you're still experiencing the bug, it may be environmental or")
        print("related to specific faces/lighting conditions.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    run_all_tests()

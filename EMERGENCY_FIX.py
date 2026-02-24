"""
EMERGENCY DIAGNOSTIC AND FIX

This script will:
1. Show exactly what's in your database
2. Check if embeddings are identical (the bug)
3. Provide step-by-step fix instructions
"""
import numpy as np
from pymongo import MongoClient
import sys

print("="*80)
print(" "*25 + "EMERGENCY DIAGNOSTIC")
print("="*80)

# Connect to database
print("\n[1] Connecting to database...")
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client["facenet_db"]
    collection = db["faces"]
    print("    ✓ Connected to MongoDB")
except Exception as e:
    print(f"    ✗ ERROR: Cannot connect to MongoDB: {e}")
    print("\n    SOLUTION: Make sure MongoDB is running:")
    print("      - Check if mongod.exe is running in Task Manager")
    print("      - Or start it with: net start MongoDB")
    sys.exit(1)

# Get all faces
print("\n[2] Checking database contents...")
faces = list(collection.find())
print(f"    Found {len(faces)} face(s) in database")

if len(faces) == 0:
    print("\n    ✗ ERROR: Database is EMPTY!")
    print("\n    SOLUTION:")
    print("      1. Run: python start.py")
    print("      2. Capture FIRST person and enter their name (e.g., 'Alice')")
    print("      3. Run: python start.py again")
    print("      4. Capture SECOND person and enter DIFFERENT name (e.g., 'Bob')")
    sys.exit(0)

# Show all faces
print("\n[3] Listing all faces in database:")
print("    " + "-"*76)
for i, face in enumerate(faces, 1):
    emb = np.array(face['embedding'])
    print(f"\n    {i}. Name: {face['name']}")
    print(f"       Face ID: {face['face_id']}")
    print(f"       Registered: {face['created_at']}")
    print(f"       Embedding shape: {emb.shape}")
    print(f"       Embedding norm: {np.linalg.norm(emb):.6f}")
    print(f"       First 5 values: {emb[:5]}")
    print(f"       Last 5 values: {emb[-5:]}")

print("\n    " + "-"*76)

# CRITICAL CHECK: Are all embeddings identical?
if len(faces) >= 2:
    print("\n[4] 🔍 CHECKING IF EMBEDDINGS ARE IDENTICAL (THIS IS THE BUG!)...")
    print("    " + "-"*76)
    
    embeddings = [np.array(f['embedding']) for f in faces]
    names = [f['name'] for f in faces]
    
    # Helper function
    def cosine_sim(a, b):
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        return np.dot(a_norm, b_norm)
    
    bug_found = False
    
    # Compare all pairs
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            emb1 = embeddings[i]
            emb2 = embeddings[j]
            name1 = names[i]
            name2 = names[j]
            
            # Check if exactly same
            are_identical = np.array_equal(emb1, emb2)
            sim = cosine_sim(emb1, emb2)
            
            print(f"\n    Comparing: '{name1}' vs '{name2}'")
            print(f"       Arrays identical: {are_identical}")
            print(f"       Cosine similarity: {sim:.10f}")
            
            if are_identical:
                print("       ❌❌❌ BUG FOUND! ❌❌❌")
                print("       These are THE EXACT SAME EMBEDDING!")
                print("       This is why both faces are recognized as the same person.")
                bug_found = True
            elif sim >= 0.99:
                print("       ❌ Nearly identical! This will cause false matches.")
                bug_found = True
            elif sim >= 0.9:
                print("       ⚠️  Very similar - might cause confusion")
            elif sim >= 0.75:
                print("       ⚠️  Above threshold - will be confused")
            else:
                print("       ✓ Different enough (should work correctly)")
    
    print("\n    " + "-"*76)
    
    # Show the ROOT CAUSE
    print("\n" + "="*80)
    if bug_found:
        print("ROOT CAUSE IDENTIFIED!")
        print("="*80)
        print("\n❌ Your database contains identical or nearly identical embeddings!")
        print("\nThis happens when:")
        print("  1. You registered the SAME PERSON twice with different names")
        print("  2. You registered photos that are too similar")
        print("  3. The embedding generation has a bug")
        
        print("\n" + "="*80)
        print("SOLUTION - STEP BY STEP FIX:")
        print("="*80)
        print("\n1. DELETE ALL FACES (start fresh):")
        print("      python clear_database.py")
        
        print("\n2. REGISTER FIRST PERSON:")
        print("      python start.py")
        print("      - Make sure FIRST person is visible")
        print("      - Good lighting, face clearly visible")
        print("      - Enter their name (e.g., 'Alice')")
        
        print("\n3. REGISTER SECOND PERSON (DIFFERENT PERSON!):")
        print("      python start.py")
        print("      - Make sure a DIFFERENT person is visible")
        print("      - Good lighting, face clearly visible  ")
        print("      - Enter a DIFFERENT name (e.g., 'Bob')")
        
        print("\n4. VERIFY IT WORKS:")
        print("      python test_pipeline.py")
        print("      - Should show different similarity scores")
        print("      - Should recognize different people correctly")
        
        print("\n" + "="*80)
        print("\n⚠️  IMPORTANT:")
        print("   - Make sure you're capturing TWO DIFFERENT PEOPLE")
        print("   - Don't use the same person with different names")
        print("   - Ensure good lighting and clear face visibility")
        print("   - Face should be looking directly at camera")
        print("="*80)
        
    else:
        print("INTERESTING - Embeddings are DIFFERENT!")
        print("="*80)
        print("\nThe embeddings are actually different, but recognition is still failing.")
        print("\nPossible causes:")
        print("  1. Recognition threshold too low (currently 0.75)")
        print("  2. Real faces produce higher similarity than test data")
        print("  3. Face preprocessing issue")
        
        print("\nTry INCREASING the threshold:")
        print("  1. Open: .env file")
        print("  2. Change: RECOGNITION_THRESHOLD=0.90")
        print("  3. Test again with: python start.py")
        
        print("\nOr check face detection quality:")
        print("  - Ensure faces are well-lit")
        print("  - Face should be frontal, not at angle")
        print("  - Remove glasses/obstructions if possible")

else:
    print("\n[4] Only 1 face in database - cannot check for duplicates")
    print("    Register at least 2 different people to test recognition.")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)

client.close()

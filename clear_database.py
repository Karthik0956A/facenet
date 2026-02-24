"""
Clear all faces from the database and start fresh.
"""
from database import FaceDatabase

print("\n" + "="*70)
print("CLEAR ALL FACES FROM DATABASE")
print("="*70)

try:
    db = FaceDatabase()
    faces = db.get_all_faces()
    
    print(f"\nCurrent faces in database: {len(faces)}")
    
    if len(faces) == 0:
        print("\n✓ Database is already empty!")
        db.close()
        exit(0)
    
    print("\nFaces to be deleted:")
    for i, face in enumerate(faces, 1):
        print(f"  {i}. {face['name']} (ID: {face['face_id'][:8]}...)")
    
    print("\n" + "="*70)
    response = input("\n⚠️  Delete ALL faces? Type 'DELETE ALL' to confirm: ").strip()
    
    if response == "DELETE ALL":
        print("\nDeleting faces...")
        deleted_count = 0
        
        for face in faces:
            if db.delete_face(face['face_id']):
                deleted_count += 1
                print(f"  ✓ Deleted: {face['name']}")
            else:
                print(f"  ✗ Failed to delete: {face['name']}")
        
        print(f"\n✓ Deleted {deleted_count} face(s)")
        print("\nYou can now register faces fresh with:")
        print("  python start.py")
    else:
        print("\n✗ Cancelled. Database unchanged.")
    
    db.close()
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

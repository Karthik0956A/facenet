#!/usr/bin/env python
"""
Setup and verification script for FaceNet Facial Recognition System.
Checks dependencies, MongoDB connection, and system readiness.
"""
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def check_python_version():
    """Check Python version."""
    print("\n1. Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 10:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro}")
        print("   Error: Python 3.10 or higher required")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\n2. Checking Python dependencies...")
    
    required_packages = [
        'numpy',
        'cv2',
        'pymongo',
        'mtcnn',
        'tensorflow',
        'keras_facenet'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True


def check_mongodb():
    """Check MongoDB connection."""
    print("\n3. Checking MongoDB connection...")
    
    try:
        from database import FaceDatabase
        
        db = FaceDatabase()
        count = db.count_faces()
        db.close()
        
        print(f"   ✓ MongoDB connected")
        print(f"   ✓ Database: {count} face(s) registered")
        return True
        
    except Exception as e:
        print(f"   ✗ MongoDB connection failed: {e}")
        print("   Make sure MongoDB is running:")
        print("     - Windows: mongod --dbpath C:\\data\\db")
        print("     - Linux/Mac: sudo systemctl start mongodb")
        print("     - Docker: docker run -d -p 27017:27017 mongo")
        return False


def check_camera():
    """Check camera availability."""
    print("\n4. Checking camera...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print(f"   ✓ Camera accessible")
                print(f"   ✓ Frame size: {frame.shape}")
                return True
            else:
                print(f"   ⚠ Camera opened but cannot read frames")
                return False
        else:
            print(f"   ⚠ Camera not accessible (may not be available)")
            print("   Note: Camera required for capture/recognition")
            return False
            
    except Exception as e:
        print(f"   ✗ Camera check failed: {e}")
        return False


def check_directories():
    """Check required directories exist."""
    print("\n5. Checking project directories...")
    
    dirs = ['models', 'temp', 'logs', 'tests']
    all_exist = True
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"   ✓ {dir_name}/")
        else:
            print(f"   ⚠ {dir_name}/ (creating...)")
            dir_path.mkdir(exist_ok=True)
            all_exist = False
    
    return True


def check_config():
    """Check configuration."""
    print("\n6. Checking configuration...")
    
    try:
        from config import validate_config
        
        if validate_config():
            print("   ✓ Configuration valid")
            
            from config import (
                RECOGNITION_THRESHOLD,
                EMBEDDING_DIMENSION,
                CAMERA_INDEX
            )
            
            print(f"   ✓ Recognition threshold: {RECOGNITION_THRESHOLD}")
            print(f"   ✓ Embedding dimension: {EMBEDDING_DIMENSION}")
            print(f"   ✓ Camera index: {CAMERA_INDEX}")
            return True
        else:
            print("   ✗ Configuration validation failed")
            return False
            
    except Exception as e:
        print(f"   ✗ Configuration error: {e}")
        return False


def run_setup():
    """Run complete setup verification."""
    print_header("FaceNet System Setup Verification")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("MongoDB", check_mongodb),
        ("Camera", check_camera),
        ("Directories", check_directories),
        ("Configuration", check_config),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ✗ Unexpected error: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Setup Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"   {status:8} {name}")
    
    print(f"\n   Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n   ✅ System ready! You can now use:")
        print("      python main.py register")
        print("      python main.py recognize")
        print("      python main.py list")
        return 0
    else:
        print("\n   ⚠ Some checks failed. Please fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_setup())

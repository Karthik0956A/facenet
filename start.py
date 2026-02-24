"""
Quick start script for FaceNet facial recognition.
Runs smart recognition mode with auto-registration.
"""
import sys
from smart_recognition import SmartRecognition


def main():
    """Run smart recognition."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     FaceNet Facial Recognition - Smart Mode                  ║
╚══════════════════════════════════════════════════════════════╝

Welcome! This mode will:
  ✓ Recognize you if you're already registered
  ✓ Ask for your name if you're new (first time)
  ✓ Remember you for next time

Ready to start? Let's go!
""")
    
    try:
        smart = SmartRecognition()
        smart.run()
        smart.close()
        
        print("\n✓ Thank you for using FaceNet!")
        print("\nTry these commands:")
        print("  python start.py          - Run this again")
        print("  python main.py list      - See all registered faces")
        print("  python main.py continuous - Real-time recognition")
        print("  python main.py stats     - System statistics")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return 130
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Run: python setup.py")
        print("  2. Check MongoDB is running")
        print("  3. Check camera is connected")
        return 1


if __name__ == "__main__":
    sys.exit(main())

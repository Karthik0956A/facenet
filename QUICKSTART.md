# Quick Start Guide

Get up and running with FaceNet facial recognition in 5 minutes!

## Prerequisites

- Python 3.10+
- MongoDB (or Docker)
- Webcam

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MongoDB

**Option A - Docker (Easiest):**
```bash
docker run -d -p 27017:27017 --name facenet-mongo mongo:latest
```

**Option B - Local Installation:**
- Windows: Download from [mongodb.com](https://www.mongodb.com/try/download/community), then run `mongod`
- Linux: `sudo apt-get install mongodb && sudo systemctl start mongodb`
- macOS: `brew install mongodb-community && brew services start mongodb-community`

### 3. Verify Setup

```bash
python setup.py
```

✅ All checks should pass!

## 🚀 Smart Mode (Recommended!)

The easiest way to get started - automatically registers new faces and recognizes known ones!

```bash
# Simple way
python start.py

# Or using main script
python main.py smart
```

**First time (Registration):**
1. Look at the camera
2. Press **'c'** to capture
3. Enter your name when prompted
4. Done! You're registered ✓

**Second time (Recognition):**
1. Look at the camera
2. Press **'c'** to capture
3. You'll be automatically recognized! 🎉

## Basic Usage

### Register Your First Face (Manual)

```bash
python main.py register
```

1. Enter your name
2. Look at the camera
3. Press **'c'** to capture
4. Done! ✓

### Recognize a Face

```bash
python main.py recognize
```

1. Look at the camera
2. Press **'c'** to capture
3. See recognition result!

### See Live Recognition

```bash
python main.py continuous
```

- Real-time face detection with bounding boxes
- Green box = Recognized
- Red box = Unknown
- Press **'q'** to quit

## Common Commands

| Command | Description |
|---------|-------------|
| `python start.py` | **🌟 Smart mode (recommended!)** |
| `python main.py smart` | Smart mode with auto-registration |
| `python main.py register` | Register new face manually |
| `python main.py recognize` | Recognize single face |
| `python main.py continuous` | Live recognition mode |
| `python main.py list` | List all registered faces |
| `python main.py delete` | Delete a face |
| `python main.py stats` | Show system statistics |

## Troubleshooting

### Camera not working?

```bash
# Try different camera index
export CAMERA_INDEX=1
python main.py recognize
```

### MongoDB connection failed?

```bash
# Check if MongoDB is running
mongo --eval "db.adminCommand('ping')"

# Or start with Docker
docker start facenet-mongo
```

### Face not detected?

- Ensure good lighting
- Face the camera directly
- Move closer (face should be > 40x40 pixels)
- Remove obstructions (sunglasses, etc.)

### Dependencies not installing?

```bash
# Upgrade pip first
pip install --upgrade pip

# Try again
pip install -r requirements.txt

# For GPU support (optional)
pip install tensorflow[and-cuda]
```

## Next Steps

1. **Register multiple people** - Try different names and faces
2. **Test recognition** - See how accurate it is
3. **Adjust threshold** - Edit `.env` file (copy from `.env.example`)
4. **Read full docs** - Check `README.md` for advanced features

## Tips for Best Results

✅ **Good lighting** - Natural light works best  
✅ **Face the camera** - Look straight ahead  
✅ **Remove obstructions** - No sunglasses or masks  
✅ **Stay still** - Don't move during capture  
✅ **Register multiple angles** - For better recognition  

## Example Session

### Using Smart Mode (Easiest!)

```bash
# Start MongoDB
docker run -d -p 27017:27017 mongo

# Verify setup
python setup.py

# First person - will ask for name
python start.py
# → Press 'c' to capture
# → Face not recognized! Let's register...
# → Enter your name: "Alice"
# ✓ Successfully registered: Alice

# Same person again - will recognize!
python start.py
# → Press 'c' to capture
# ✓ FACE RECOGNIZED! Welcome back, Alice! 👋

# Different person - will ask for name
python start.py
# → Press 'c' to capture
# → Face not recognized! Let's register...
# → Enter your name: "Bob"
# ✓ Successfully registered: Bob
```

### Manual Registration and Recognition

```bash
# Register yourself
python main.py register
# → Enter name: "Alice"
# → Press 'c' to capture
# ✓ Successfully registered!

# Register another person
python main.py register
# → Enter name: "Bob"
# → Press 'c' to capture
# ✓ Successfully registered!

# List registered faces
python main.py list
# Alice    abc-123...    2024-02-23
# Bob      def-456...    2024-02-23

# Try recognition
python main.py recognize
# → Press 'c' to capture
# ✓ Face recognized: Alice (confidence: 0.95)

# Live recognition
python main.py continuous
# [Real-time bounding boxes appear]
# Press 'q' to quit
```

## Configuration

Create `.env` file for custom settings:

```env
RECOGNITION_THRESHOLD=0.6      # Lower = more lenient
CAMERA_INDEX=0                 # Change if using external camera
LOG_LEVEL=INFO                 # DEBUG for troubleshooting
```

## Architecture Overview

```
Webcam → Camera Module → MTCNN Detector → FaceNet Embedder 
                                              ↓
                                         MongoDB Database
                                              ↓
                                      Recognizer (Cosine Similarity)
                                              ↓
                                         Results Display
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test specific module
python camera.py
python detector.py
python embedder.py
```

## Docker Deployment

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Performance

- **Registration**: ~1-2 seconds per face
- **Recognition**: ~0.5-1 second per query
- **Database**: Handles 1000+ faces efficiently
- **Memory**: ~500MB RAM (CPU mode)

## Support

Having issues? Check:

1. ✅ This Quick Start guide
2. ✅ Full README.md (comprehensive)
3. ✅ Logs in `logs/facenet.log`
4. ✅ Run `python setup.py` for diagnostics

## What's Next?

### Production Features:
- ✅ Add web interface (FastAPI/Flask)
- ✅ Implement authentication
- ✅ Add batch processing
- ✅ Deploy to cloud (AWS/Azure/GCP)
- ✅ Enable GPU acceleration

### System Improvements:
- ✅ Fine-tune recognition threshold
- ✅ Add liveness detection
- ✅ Implement face anti-spoofing
- ✅ Add multi-face tracking
- ✅ Create dashboard UI

---

**Ready to build? Start with `python main.py register`!** 🚀

# FaceNet Facial Recognition System

A production-quality facial recognition system using FaceNet embeddings, MTCNN face detection, and MongoDB for storage. The system provides real-time face recognition via webcam with threshold-based matching to prevent false positives.

## Features

- ✅ **Real-time Face Detection**: Uses MTCNN for accurate face detection
- ✅ **FaceNet Embeddings**: Generates 512-dimensional face embeddings
- ✅ **MongoDB Storage**: Stores embeddings with metadata (name, date, summary)
- ✅ **Cosine Similarity Matching**: Threshold-based recognition to avoid false positives
- ✅ **Interactive CLI**: Easy-to-use command-line interface
- ✅ **Privacy-Focused**: Automatically deletes raw images after embedding extraction
- ✅ **Production-Ready**: Includes logging, error handling, and retry logic
- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Comprehensive Testing**: Unit and integration tests included

## Architecture

```
facenet_project/
│
├── main.py              # CLI entry point
├── start.py             # Quick start with smart mode
├── config.py            # Configuration management
├── database.py          # MongoDB operations
├── camera.py            # Webcam capture
├── detector.py          # MTCNN face detection
├── embedder.py          # FaceNet embedding generation
├── recognizer.py        # Face recognition engine
├── register.py          # Face registration
├── delete.py            # Face deletion
├── smart_recognition.py # Smart mode with auto-registration
├── utils.py             # Utility functions
│
├── tests/               # Test suite
│   ├── test_camera.py
│   ├── test_embedding.py
│   └── test_recognition.py
│
├── models/              # Model storage (auto-created)
├── temp/                # Temporary images (auto-created)
├── logs/                # Application logs (auto-created)
│
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Requirements

- **Python**: 3.10 or higher
- **MongoDB**: 4.0 or higher (running locally or remotely)
- **Webcam**: For real-time capture
- **RAM**: Minimum 4GB recommended
- **GPU**: Optional (CPU-only mode supported)

## Installation

### 1. Clone or Download Project

```bash
cd facenet_project
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will download FaceNet pretrained weights (~90MB).

### 4. Install MongoDB

#### Windows:
Download from [MongoDB Downloads](https://www.mongodb.com/try/download/community)

Start MongoDB:
```bash
mongod --dbpath C:\data\db
```

#### Linux:
```bash
sudo apt-get install -y mongodb
sudo systemctl start mongodb
```

#### macOS:
```bash
brew install mongodb-community
brew services start mongodb-community
```

#### Docker (All platforms):
```bash
docker run -d -p 27017:27017 --name facenet-mongo mongo:latest
```

### 5. Configure Environment Variables (Optional)

Create `.env` file in project root:

```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=facenet_db
MONGODB_COLLECTION=faces

# Recognition Settings
RECOGNITION_THRESHOLD=0.6
HIGH_CONFIDENCE_THRESHOLD=0.8

# Camera Settings
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480

# Logging
LOG_LEVEL=INFO
```

## Usage

### Smart Mode (Recommended!) 🌟

The quickest way to get started! Smart mode automatically:
- ✅ Recognizes your face if already registered
- ✅ Asks for your name if you're new
- ✅ Remembers you for next time

```bash
# Easiest way
python start.py

# Or using main script
python main.py smart
```

**How it works:**
1. Position your face in the frame
2. Press 'c' to capture
3. **First time**: System asks for your name → Registers you
4. **Next time**: System recognizes you automatically! 🎉

This is perfect for:
- Quick testing and demos
- Personal use
- Family members
- Small teams

### Register a New Face

```bash
python main.py register
```

Follow the interactive prompts:
1. Enter person's name
2. Add optional summary/notes
3. Position face in camera frame
4. Press 'c' to capture
5. Face is automatically processed and stored

### Recognize a Face

```bash
python main.py recognize
```

Steps:
1. Position face in camera frame
2. Press 'c' to capture
3. System searches database and displays match with confidence

### Continuous Recognition Mode

```bash
python main.py continuous
```

Real-time recognition with bounding boxes:
- Green box = Recognized face
- Red box = Unknown face
- Press 'q' to quit

### Delete a Face

```bash
python main.py delete
```

Options:
1. Delete by name
2. Delete by face recognition
3. Cancel

### List All Registered Faces

```bash
python main.py list
```

Displays table with:
- Name
- Face ID
- Registration date

### Show System Statistics

```bash
python main.py stats
```

Displays:
- Total registered faces
- Recognition thresholds
- Database connection status

## MongoDB Schema

### Collection: `faces`

```javascript
{
    "_id": ObjectId("..."),
    "face_id": "uuid-string-here",           // Unique identifier
    "name": "John Doe",                       // Person's name
    "embedding": [0.123, -0.456, ...],       // 512-dimensional array
    "summary": "Optional notes",              // Additional information
    "created_at": "2024-01-15T10:30:00",     // ISO timestamp
    "updated_at": "2024-01-15T10:30:00",     // ISO timestamp
    "embedding_dimension": 512                // Always 512 for FaceNet
}
```

### Indexes

- `face_id`: Unique index
- `name`: Non-unique index for fast lookups
- `created_at`: For time-based queries

## Configuration

All configuration is in `config.py` and can be overridden with environment variables.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RECOGNITION_THRESHOLD` | 0.6 | Minimum similarity for match (0-1) |
| `HIGH_CONFIDENCE_THRESHOLD` | 0.8 | Threshold for high confidence |
| `MTCNN_MIN_FACE_SIZE` | 40 | Minimum face size in pixels |
| `CAMERA_INDEX` | 0 | Webcam device index |
| `EMBEDDING_DIMENSION` | 512 | FaceNet embedding size |

### Adjusting Recognition Threshold

Lower threshold (e.g., 0.4):
- More lenient matching
- Higher false positive rate
- Better for varied lighting/angles

Higher threshold (e.g., 0.8):
- Stricter matching
- Lower false positive rate
- Requires consistent conditions

## Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Test Module

```bash
python -m pytest tests/test_camera.py -v
python -m pytest tests/test_embedding.py -v
python -m pytest tests/test_recognition.py -v
```

### Run with Coverage

```bash
python -m pytest tests/ --cov=. --cov-report=html
```

View coverage report: `htmlcov/index.html`

### Manual Testing

Test individual modules:

```bash
python camera.py
python detector.py
python embedder.py
python recognizer.py
```

## Troubleshooting

### Camera Not Found

**Problem**: `Failed to open camera 0`

**Solutions**:
- Check if webcam is connected
- Try different camera index: `CAMERA_INDEX=1`
- Check if another app is using camera
- On Linux, check permissions: `ls -l /dev/video*`

### MongoDB Connection Failed

**Problem**: `Failed to connect to MongoDB after 3 attempts`

**Solutions**:
- Ensure MongoDB is running: `mongod` or `brew services list`
- Check connection string in `.env`
- Verify port 27017 is not blocked
- Try: `mongo --eval "db.adminCommand('ping')"`

### FaceNet Model Download Issues

**Problem**: Model fails to download on first run

**Solutions**:
- Check internet connection
- Download manually from [keras-facenet releases](https://github.com/nyoki-mtl/keras-facenet/releases)
- Place in `~/.keras/models/`
- Retry with better connection

### No Face Detected

**Problem**: `No face detected in image`

**Solutions**:
- Ensure good lighting
- Face directly towards camera
- Move closer to camera
- Check if face is fully visible (no obstructions)
- Minimum face size is 40x40 pixels

### Low Recognition Accuracy

**Problem**: Faces not being recognized correctly

**Solutions**:
- Lower recognition threshold in config
- Register multiple images of same person
- Ensure consistent lighting during registration
- Check face detection quality (confidence > 0.9)
- Re-register faces with better quality images

### Memory Issues

**Problem**: `Out of memory` errors

**Solutions**:
- Close other applications
- Use CPU-only TensorFlow: `pip install tensorflow-cpu`
- Process images sequentially instead of batch
- Reduce camera resolution in config

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solutions**:
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.10+)
- Update pip: `pip install --upgrade pip`

## Performance Optimization

### GPU Acceleration

To use GPU (if available):

```bash
pip install tensorflow[and-cuda]  # For CUDA support
```

Check GPU usage:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Batch Processing

For processing multiple faces:

```python
from embedder import FaceEmbedder

embedder = FaceEmbedder()
embeddings = embedder.generate_embeddings_batch(face_images_list)
```

### Database Optimization

For large databases (1000+ faces):
- Use `batch_cosine_similarity` for faster matching
- Consider vector databases (Faiss, Pinecone) for scale
- Add compound indexes for frequent queries

## Security Considerations

1. **Privacy**: Raw images are deleted after embedding extraction
2. **Data Protection**: Store MongoDB credentials securely
3. **Access Control**: Implement authentication for production
4. **Encryption**: Use TLS for MongoDB connections in production
5. **Audit Logging**: All operations are logged with timestamps

## Production Deployment

### Recommended Setup

1. **Web API**: Wrap with FastAPI or Flask
2. **Message Queue**: Use Celery for async processing
3. **Load Balancer**: Distribute recognition requests
4. **Monitoring**: Integrate with Prometheus/Grafana
5. **Backup**: Regular MongoDB backups
6. **Scaling**: Separate camera nodes from processing nodes

### Example Docker Compose

```yaml
version: '3.8'
services:
  mongodb:
    image: mongo:latest
    volumes:
      - mongo_data:/data/db
  
  facenet:
    build: .
    depends_on:
      - mongodb
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/

volumes:
  mongo_data:
```

## API Reference

### Core Classes

#### `Camera`
- `open()`: Open camera connection
- `read_frame()`: Read single frame
- `capture_image()`: Capture and save image
- `show_stream()`: Interactive video stream
- `release()`: Release camera resources

#### `FaceDetector`
- `detect_faces(image)`: Detect all faces
- `detect_largest_face(image)`: Detect largest face
- `extract_face(image, detection)`: Extract face region
- `is_valid_detection(detection)`: Validate detection quality

#### `FaceEmbedder`
- `generate_embedding(face)`: Generate 512D embedding
- `generate_embeddings_batch(faces)`: Batch processing
- `verify_embedding(embedding)`: Validate embedding
- `compare_embeddings(emb1, emb2)`: Calculate similarity

#### `FaceRecognizer`
- `recognize(embedding)`: Find best match
- `recognize_top_k(embedding, k)`: Get top K matches
- `verify_identity(embedding, name)`: Verify claimed identity
- `set_threshold(threshold)`: Update threshold

#### `FaceDatabase`
- `insert_face()`: Add new face
- `get_face_by_name(name)`: Retrieve by name
- `get_all_faces()`: Get all faces
- `delete_face(face_id)`: Remove face
- `count_faces()`: Total count

## Contributing

To extend this system:

1. **Add new features** in separate modules
2. **Write tests** for new functionality
3. **Update documentation** in this README
4. **Follow coding standards**: PEP 8 for Python
5. **Log appropriately**: Use structured logging

Example custom metric:

```python
from utils import euclidean_distance

def recognize_with_euclidean(self, query_embedding):
    """Alternative recognition using Euclidean distance."""
    stored_faces = self.db.get_all_faces()
    
    best_match = None
    min_distance = float('inf')
    
    for face in stored_faces:
        distance = euclidean_distance(query_embedding, face['embedding'])
        if distance < min_distance:
            min_distance = distance
            best_match = face
    
    # Distance threshold (lower is better)
    if min_distance < 0.8:
        return True, best_match, 1.0 / (1.0 + min_distance)
    
    return False, None, 0.0
```

## License

This project is provided for educational and commercial use. Ensure compliance with FaceNet model license and local privacy regulations.

## Support

For issues or questions:
1. Check this README's troubleshooting section
2. Review logs in `logs/facenet.log`
3. Test individual modules for isolation
4. Check MongoDB connection and data

## Acknowledgments

- **FaceNet**: [Schroff et al., 2015](https://arxiv.org/abs/1503.03832)
- **MTCNN**: Multi-task Cascaded Convolutional Networks
- **keras-facenet**: Pre-trained FaceNet implementation
- **MongoDB**: Document database for embedding storage

## Version History

- **v1.0.0** (2024): Initial production release
  - Real-time recognition
  - MongoDB integration
  - CLI interface
  - Comprehensive testing

---

**Built with Python 3.10+ | MongoDB | OpenCV | TensorFlow | FaceNet**
#   f a c e n e t  
 
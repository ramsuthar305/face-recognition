# Face Recognition API Server

Real-time face recognition with REST API endpoints for uploading faces and recognizing images/videos.

## Installation

### macOS/Linux

1. **Clone and setup:**
```bash
git clone <repository-url>
cd face-recognition

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. **Download model weights:**
```bash
# Create weights directory
sh download.sh
```

## Running the Server

### Option 1: With FAISS Database (Default)
```bash
# Activate virtual environment
source venv/bin/activate

# Start API server with FAISS
python api.py
```

### Option 2: With Milvus Vector Database (Recommended for 100K+ faces)
```bash
# 1. Setup Milvus with Podman
./setup_milvus.sh

# 2. Start API server with Milvus
source venv/bin/activate
python api.py --use-milvus
```

Server starts at: `http://localhost:8080`

## Milvus Setup (Recommended for Large Scale)

Milvus provides better scalability and performance for large face databases (100K+ faces).

### Prerequisites
- Podman installed (`brew install podman` on macOS)
- At least 4GB RAM available

### Setup Steps
```bash
# Run the setup script
./setup_milvus.sh

# This will:
# - Install podman-compose
# - Start Milvus, etcd, and MinIO containers
# - Install pymilvus Python package
# - Verify services are running
```

### Milvus Services
- **Milvus**: `localhost:19530` (vector database)
- **Milvus Web UI**: `localhost:9091` (health check)
- **MinIO**: `localhost:9000` (object storage)
- **MinIO Console**: `localhost:9001` (admin interface)

### Managing Milvus
```bash
# Check status
podman-compose ps

# View logs
podman-compose logs milvus

# Stop services
podman-compose down

# Restart services
podman-compose up -d
```

## API Endpoints

### Health Check
```bash
curl -X GET http://localhost:8080/health
```

### Upload Face Images
```bash
# Upload single image
curl -X POST http://localhost:8080/upload \
  -F "person_name=john_doe" \
  -F "files=@path/to/image.jpg"

# Upload multiple images for same person
curl -X POST http://localhost:8080/upload \
  -F "person_name=jane_smith" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg"
```

### Recognize Faces in Image
```bash
curl -X POST http://localhost:8080/recognize \
  -F "file=@test_image.jpg" \
  -F "similarity_threshold=0.5"
```

### Recognize Faces in Video
```bash
curl -X POST http://localhost:8080/recognize \
  -F "file=@test_video.mp4" \
  -F "similarity_threshold=0.4"
```

### List All Persons
```bash
curl -X GET http://localhost:8080/list_persons
```

## API Response Examples

### Upload Response
```json
{
  "success": true,
  "person_name": "john_doe",
  "added_faces": 2,
  "total_faces_in_db": 15,
  "errors": []
}
```

### Recognition Response
```json
{
  "success": true,
  "type": "image",
  "faces_detected": 1,
  "results": [
    {
      "face_id": 0,
      "person_name": "john_doe",
      "similarity_score": 0.756,
      "bounding_box": {
        "x1": 100,
        "y1": 150,
        "x2": 200,
        "y2": 250
      }
    }
  ]
}
```

### Health Check Response
```json
{
  "status": "healthy",
  "models_loaded": true,
  "database_faces": 13
}
```

## Parameters

### Upload Endpoint
- `person_name` (required): Name to label the faces
- `files` (required): One or more image files (JPG, PNG, JPEG)

### Recognition Endpoint
- `file` (required): Image or video file
- `similarity_threshold` (optional): Recognition threshold (default: 0.4)
- `max_faces` (optional): Max faces to detect (default: 0 = unlimited)

## Supported File Types
- **Images**: JPG, JPEG, PNG
- **Videos**: MP4, AVI, MOV
- **Max file size**: 16MB
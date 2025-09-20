# Face Recognition API Documentation

## Overview

This API provides face recognition capabilities with two main functionalities:
1. **Upload new face images** to build/update the face database
2. **Recognize faces** in uploaded images or videos

## Starting the Server

```bash
cd /path/to/face-reidentification
source venv/bin/activate
python api.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API server and models are running properly.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "database_faces": 13
}
```

### 2. Upload Face Images

**POST** `/upload`

Upload new face images for a person to add them to the database.

**Parameters:**
- `person_name` (form data, required): Name of the person
- `files` (form data, required): One or more image files (JPG, JPEG, PNG)

**Example using curl:**
```bash
curl -X POST http://localhost:5000/upload \
  -F "person_name=john_doe" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

**Response:**
```json
{
  "success": true,
  "person_name": "john_doe",
  "added_faces": 2,
  "total_faces_in_db": 15,
  "errors": [],
  "warning": "Some files had issues: 0 errors"
}
```

### 3. Recognize Faces

**POST** `/recognize`

Recognize faces in an uploaded image or video file.

**Parameters:**
- `file` (form data, required): Image or video file (JPG, JPEG, PNG, MP4, AVI, MOV)
- `similarity_threshold` (form data, optional): Similarity threshold (default: 0.4)
- `max_faces` (form data, optional): Maximum number of faces to detect (default: 0 for unlimited)

**Example using curl (Image):**
```bash
curl -X POST http://localhost:5000/recognize \
  -F "file=@test_image.jpg" \
  -F "similarity_threshold=0.5"
```

**Example using curl (Video):**
```bash
curl -X POST http://localhost:5000/recognize \
  -F "file=@test_video.mp4" \
  -F "similarity_threshold=0.4" \
  -F "max_faces=10"
```

**Response for Image:**
```json
{
  "success": true,
  "type": "image",
  "faces_detected": 2,
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
      },
      "confidence": 0.95
    },
    {
      "face_id": 1,
      "person_name": "Unknown",
      "similarity_score": 0.234,
      "bounding_box": {
        "x1": 300,
        "y1": 180,
        "x2": 400,
        "y2": 280
      },
      "confidence": 0.89
    }
  ]
}
```

**Response for Video:**
```json
{
  "success": true,
  "type": "video",
  "total_frames": 1800,
  "processed_frames": 60,
  "unique_persons": [
    {
      "person_name": "john_doe",
      "max_similarity": 0.823,
      "appearances": 15,
      "first_seen_frame": 30,
      "last_seen_frame": 1650
    }
  ],
  "frame_by_frame_results": [
    {
      "frame_number": 30,
      "timestamp": 1.0,
      "faces_detected": 1,
      "faces": [
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
  ]
}
```

### 4. List Persons

**GET** `/list_persons`

List all persons currently in the face database.

**Response:**
```json
{
  "success": true,
  "total_faces": 13,
  "unique_persons": 7,
  "persons": [
    {
      "name": "Joey",
      "face_count": 1
    },
    {
      "name": "ram",
      "face_count": 7
    }
  ]
}
```

## Python Client Example

```python
import requests

# Health check
response = requests.get("http://localhost:5000/health")
print(response.json())

# Upload faces
files = [
    ('files', ('image1.jpg', open('image1.jpg', 'rb'), 'image/jpeg')),
    ('files', ('image2.jpg', open('image2.jpg', 'rb'), 'image/jpeg'))
]
data = {'person_name': 'john_doe'}
response = requests.post("http://localhost:5000/upload", data=data, files=files)
print(response.json())

# Recognize faces in image
files = {'file': ('test.jpg', open('test.jpg', 'rb'), 'image/jpeg')}
data = {'similarity_threshold': 0.5}
response = requests.post("http://localhost:5000/recognize", data=data, files=files)
print(response.json())

# List all persons
response = requests.get("http://localhost:5000/list_persons")
print(response.json())
```

## JavaScript/Fetch Example

```javascript
// Upload faces
const formData = new FormData();
formData.append('person_name', 'john_doe');
formData.append('files', fileInput.files[0]);

fetch('http://localhost:5000/upload', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Recognize faces
const formData2 = new FormData();
formData2.append('file', imageFile);
formData2.append('similarity_threshold', '0.5');

fetch('http://localhost:5000/recognize', {
  method: 'POST',
  body: formData2
})
.then(response => response.json())
.then(data => console.log(data));
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing parameters, invalid files)
- `500`: Server error

Error responses include an `error` field:
```json
{
  "error": "No file provided"
}
```

## File Limitations

- **Maximum file size**: 16MB
- **Supported image formats**: PNG, JPG, JPEG
- **Supported video formats**: MP4, AVI, MOV
- **Video processing**: Processes every 30th frame to optimize performance

## Performance Notes

- **Image recognition**: Fast, processes all detected faces
- **Video recognition**: Processes every 30th frame for efficiency
- **Batch processing**: Uses parallel processing for multiple faces
- **Database**: Automatically saves after adding new faces

## Security Considerations

- Files are temporarily stored during processing and then deleted
- No authentication implemented (add as needed for production)
- CORS not configured (add as needed for web applications)
- Consider rate limiting for production use

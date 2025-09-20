import os
import cv2
import json
import numpy as np
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging
from typing import List, Dict, Any, Tuple

from database import FaceDatabase
from models import SCRFD, ArcFace
from utils.logging import setup_logging

# Setup logging
setup_logging(log_to_file=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models and database
detector = None
recognizer = None
face_db = None

# Configuration
WEIGHTS_DIR = "./weights"
DB_PATH = "./database/face_database"
FACES_DIR = "./assets/faces"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    video_extensions = {'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

def initialize_models():
    """Initialize face detection and recognition models"""
    global detector, recognizer, face_db
    
    try:
        detector = SCRFD(
            os.path.join(WEIGHTS_DIR, "det_10g.onnx"), 
            input_size=(640, 640), 
            conf_thres=0.5
        )
        recognizer = ArcFace(os.path.join(WEIGHTS_DIR, "w600k_mbf.onnx"))
        
        # Load existing face database
        face_db = FaceDatabase(db_path=DB_PATH, max_workers=4)
        face_db.load()
        
        logging.info("Models and database initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': detector is not None and recognizer is not None,
        'database_faces': face_db.index.ntotal if face_db else 0
    })

@app.route('/upload', methods=['POST'])
def upload_face():
    """
    Upload new face images for a person
    
    Expected form data:
    - person_name: str - Name of the person
    - files: List of image files
    
    Returns:
    - JSON response with success status and details
    """
    try:
        if 'person_name' not in request.form:
            return jsonify({'error': 'person_name is required'}), 400
        
        person_name = request.form['person_name'].strip()
        if not person_name:
            return jsonify({'error': 'person_name cannot be empty'}), 400
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Create person directory
        person_dir = os.path.join(FACES_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        added_faces = 0
        errors = []
        
        for file in files:
            if file and allowed_file(file.filename) and not is_video_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(person_dir, filename)
                    file.save(filepath)
                    
                    # Process the image
                    image = cv2.imread(filepath)
                    if image is None:
                        errors.append(f"Could not read image: {filename}")
                        continue
                    
                    # Detect face and get embedding
                    bboxes, kpss = detector.detect(image, max_num=1)
                    
                    if len(kpss) == 0:
                        errors.append(f"No face detected in: {filename}")
                        continue
                    
                    embedding = recognizer.get_embedding(image, kpss[0])
                    face_db.add_face(embedding, person_name)
                    added_faces += 1
                    
                    logging.info(f"Added face for {person_name} from {filename}")
                    
                except Exception as e:
                    errors.append(f"Error processing {file.filename}: {str(e)}")
                    logging.error(f"Error processing {file.filename}: {e}")
        
        # Save updated database
        if added_faces > 0:
            face_db.save()
        
        response = {
            'success': True,
            'person_name': person_name,
            'added_faces': added_faces,
            'total_faces_in_db': face_db.index.ntotal,
            'errors': errors
        }
        
        if errors:
            response['warning'] = f"Some files had issues: {len(errors)} errors"
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Upload endpoint error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/recognize', methods=['POST'])
def recognize_faces():
    """
    Recognize faces in uploaded image or video
    
    Expected form data:
    - file: Image or video file
    - similarity_threshold: float (optional, default 0.4)
    - max_faces: int (optional, default 0 for unlimited)
    
    Returns:
    - JSON response with recognition results
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get optional parameters
        similarity_threshold = float(request.form.get('similarity_threshold', 0.4))
        max_faces = int(request.form.get('max_faces', 0))
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_filepath = temp_file.name
        
        try:
            if is_video_file(file.filename):
                results = process_video(temp_filepath, similarity_threshold, max_faces)
            else:
                results = process_image(temp_filepath, similarity_threshold, max_faces)
            
            return jsonify(results), 200
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
        
    except Exception as e:
        logging.error(f"Recognition endpoint error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def process_image(image_path: str, similarity_threshold: float, max_faces: int) -> Dict[str, Any]:
    """Process a single image for face recognition"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image file")
        
        logging.info(f"Processing image: {image_path}, max_faces: {max_faces}, threshold: {similarity_threshold}")
        
        # Detect faces
        bboxes, kpss = detector.detect(image, max_num=max_faces if max_faces > 0 else 0)
        logging.info(f"Detected {len(bboxes)} faces")
    except Exception as e:
        logging.error(f"Error in process_image: {e}")
        raise
    
    if len(bboxes) == 0:
        return {
            'success': True,
            'type': 'image',
            'faces_detected': 0,
            'results': []
        }
    
    # Get embeddings for all faces
    embeddings = []
    face_boxes = []
    
    for bbox, kps in zip(bboxes, kpss):
        try:
            embedding = recognizer.get_embedding(image, kps)
            embeddings.append(embedding)
            # Convert bbox to list for JSON serialization
            face_boxes.append(bbox.tolist())
        except Exception as e:
            logging.warning(f"Error processing face embedding: {e}")
            continue
    
    if not embeddings:
        return {
            'success': True,
            'type': 'image',
            'faces_detected': 0,
            'results': []
        }
    
    # Batch search for all faces
    search_results = face_db.batch_search(embeddings, similarity_threshold)
    
    # Format results
    results = []
    for i, ((name, similarity), bbox) in enumerate(zip(search_results, face_boxes)):
        results.append({
            'face_id': i,
            'person_name': name,
            'similarity_score': float(similarity),
            'bounding_box': {
                'x1': int(bbox[0]),
                'y1': int(bbox[1]),
                'x2': int(bbox[2]),
                'y2': int(bbox[3])
            },
            'confidence': float(bbox[4]) if len(bbox) > 4 else None
        })
    
    return {
        'success': True,
        'type': 'image',
        'faces_detected': len(results),
        'results': results
    }

def process_video(video_path: str, similarity_threshold: float, max_faces: int) -> Dict[str, Any]:
    """Process a video file for face recognition"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    frame_results = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 30th frame to reduce processing time
            if frame_count % 30 == 0:
                # Detect faces
                bboxes, kpss = detector.detect(frame, max_num=max_faces if max_faces > 0 else 0)
                
                if len(bboxes) > 0:
                    # Get embeddings for all faces
                    embeddings = []
                    face_boxes = []
                    
                    for bbox, kps in zip(bboxes, kpss):
                        try:
                            embedding = recognizer.get_embedding(frame, kps)
                            embeddings.append(embedding)
                            face_boxes.append(bbox.tolist())
                        except Exception as e:
                            logging.warning(f"Error processing face embedding in frame {frame_count}: {e}")
                            continue
                    
                    if embeddings:
                        # Batch search for all faces
                        search_results = face_db.batch_search(embeddings, similarity_threshold)
                        
                        # Format frame results
                        faces = []
                        for i, ((name, similarity), bbox) in enumerate(zip(search_results, face_boxes)):
                            faces.append({
                                'face_id': i,
                                'person_name': name,
                                'similarity_score': float(similarity),
                                'bounding_box': {
                                    'x1': int(bbox[0]),
                                    'y1': int(bbox[1]),
                                    'x2': int(bbox[2]),
                                    'y2': int(bbox[3])
                                }
                            })
                        
                        frame_results.append({
                            'frame_number': frame_count,
                            'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                            'faces_detected': len(faces),
                            'faces': faces
                        })
            
            frame_count += 1
    
    finally:
        cap.release()
    
    # Aggregate results - find unique persons across all frames
    all_persons = {}
    for frame_result in frame_results:
        for face in frame_result['faces']:
            person_name = face['person_name']
            if person_name != 'Unknown':
                if person_name not in all_persons:
                    all_persons[person_name] = {
                        'person_name': person_name,
                        'max_similarity': face['similarity_score'],
                        'appearances': 1,
                        'first_seen_frame': frame_result['frame_number'],
                        'last_seen_frame': frame_result['frame_number']
                    }
                else:
                    all_persons[person_name]['max_similarity'] = max(
                        all_persons[person_name]['max_similarity'],
                        face['similarity_score']
                    )
                    all_persons[person_name]['appearances'] += 1
                    all_persons[person_name]['last_seen_frame'] = frame_result['frame_number']
    
    return {
        'success': True,
        'type': 'video',
        'total_frames': frame_count,
        'processed_frames': len(frame_results),
        'unique_persons': list(all_persons.values()),
        'frame_by_frame_results': frame_results
    }

@app.route('/list_persons', methods=['GET'])
def list_persons():
    """List all persons in the database"""
    try:
        unique_names = list(set(face_db.metadata))
        name_counts = {name: face_db.metadata.count(name) for name in unique_names}
        
        return jsonify({
            'success': True,
            'total_faces': face_db.index.ntotal,
            'unique_persons': len(unique_names),
            'persons': [
                {
                    'name': name,
                    'face_count': count
                }
                for name, count in name_counts.items()
            ]
        }), 200
    except Exception as e:
        logging.error(f"List persons error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    if not initialize_models():
        print("Failed to initialize models. Exiting.")
        exit(1)
    
    print("Face Recognition API Server Starting...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /upload - Upload face images")
    print("  POST /recognize - Recognize faces in image/video")
    print("  GET  /list_persons - List all persons in database")
    
    app.run(host='0.0.0.0', port=8080, debug=True)

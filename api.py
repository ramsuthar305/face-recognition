import os
import cv2
import json
import numpy as np
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from milvus_db import MilvusFaceDatabase
from models import SCRFD, ArcFace
from utils.logging import setup_logging
from spaces_manager import SpacesManager
from embedding_enhancer import enhance_embedding_to_1024, enhance_embedding_to_768

# Load environment variables
load_dotenv('config.env')

# Setup logging
setup_logging(log_to_file=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB max file size

# Global variables for models and database
detector = None
recognizer = None
face_db = None
spaces_manager = None

# Configuration from environment variables
WEIGHTS_DIR = os.getenv('WEIGHTS_DIR', './weights')
DB_PATH = os.getenv('DB_PATH', './database/face_database')
FACES_DIR = os.getenv('FACES_DIR', './assets/faces')
EMBEDDING_SIZE = int(os.getenv('EMBEDDING_SIZE', 1024))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 8))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.4))
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = int(os.getenv('MILVUS_PORT', 19530))
MILVUS_USER = os.getenv('MILVUS_USER', '')
MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD', '')
MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME', 'face_embeddings')
API_PORT = int(os.getenv('API_PORT', 8080))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    video_extensions = {'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

# Remove the get_database function - we'll use face_db directly

def initialize_models():
    """Initialize face detection and recognition models"""
    global detector, recognizer, face_db, spaces_manager

    detector = SCRFD(
        os.path.join(WEIGHTS_DIR, "det_10g.onnx"), 
        input_size=(640, 640), 
        conf_thres=CONFIDENCE_THRESHOLD
    )
    recognizer = ArcFace(os.path.join(WEIGHTS_DIR, "w600k_mbf.onnx"))
    
    # Initialize Milvus database
    face_db = MilvusFaceDatabase(
        collection_name=MILVUS_COLLECTION_NAME,
        embedding_size=EMBEDDING_SIZE,
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        user=MILVUS_USER,
        password=MILVUS_PASSWORD,
        max_workers=MAX_WORKERS
    )
    logging.info(f"Using Milvus vector database at {MILVUS_HOST}:{MILVUS_PORT}")
    
    # Initialize Spaces manager
    spaces_manager = SpacesManager()
    if spaces_manager.check_connection():
        logging.info("Spaces manager initialized successfully")
    else:
        logging.warning("Spaces connection failed - uploads will be local only")
        spaces_manager = None
    
    logging.info("Models and database initialized successfully")
    return True

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check models
        models_healthy = detector is not None and recognizer is not None
        
        # Check database
        db_healthy = face_db is not None
        
        # Overall health status
        overall_healthy = models_healthy and db_healthy
        
        return jsonify({
            'success': True,
            'response_code': 'FIS0001',
            'message': 'System health check completed successfully',
            'status': 'healthy' if overall_healthy else 'degraded',
            'models_loaded': models_healthy,
            'database_connected': db_healthy,
            'database_type': 'Milvus',
            'total_faces_in_db': face_db.index.ntotal if face_db else 0
        })
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({
            'success': False,
            'response_code': 'FIE0001',
            'message': 'Health check failed',
            'error': str(e),
            'status': 'unhealthy'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_face():
    """
    Upload new face images for a person
    
    Expected form data:
    - name: str - Name of the person
    - id: str - Unique ID of the person  
    - files: List of image files
    
    Returns:
    - JSON response with success status and details
    """
    try:
        # Validate required fields
        if 'name' not in request.form:
            return jsonify({'success': False, 'response_code': 'FIE1001', 'message': 'Missing required parameter: name', 'error': 'name is required'}), 400
        if 'id' not in request.form:
            return jsonify({'success': False, 'response_code': 'FIE1002', 'message': 'Missing required parameter: id', 'error': 'id is required'}), 400
        
        name = request.form['name'].strip()
        person_id = request.form['id'].strip()
        
        if not name:
            return jsonify({'success': False, 'response_code': 'FIE1003', 'message': 'Invalid parameter: name is empty', 'error': 'name cannot be empty'}), 400
        if not person_id:
            return jsonify({'success': False, 'response_code': 'FIE1004', 'message': 'Invalid parameter: id is empty', 'error': 'id cannot be empty'}), 400
        
        if 'files' not in request.files:
            return jsonify({'success': False, 'response_code': 'FIE1005', 'message': 'No files in upload request', 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'success': False, 'response_code': 'FIE1006', 'message': 'Empty file list in upload request', 'error': 'No files selected'}), 400
        
        # Create identifier by concatenating name and id
        identifier = f"{secure_filename(name)}_{secure_filename(person_id)}"
        
        # Process and store images
        added_faces = 0
        errors = []
        local_files = []
        embeddings = []
        
        # Create temporary directory for processing
        temp_dir = os.path.join(FACES_DIR, "temp", identifier)
        os.makedirs(temp_dir, exist_ok=True)
        # Process each image file
        duplicates_found = []
        
        # First pass: Validate all images have faces before processing any
        valid_files = []
        invalid_files = []
        
        logging.info(f"Starting validation for {len(files)} files")
        for i, file in enumerate(files):
            logging.info(f"Processing file {i+1}: {file.filename}")
            if file and allowed_file(file.filename) and not is_video_file(file.filename):
                # Save temporarily for validation
                filename = f"{identifier}_{i+1}_{secure_filename(file.filename)}"
                temp_filepath = os.path.join(temp_dir, filename)
                file.save(temp_filepath)
                logging.info(f"Saved temp file: {temp_filepath}")
                
                # Validate the image
                image = cv2.imread(temp_filepath)
                if image is None:
                    invalid_files.append({
                        'filename': file.filename,
                        'error': 'Could not read image file',
                        'filepath': temp_filepath
                    })
                    continue
                
                # Check if face is detected
                logging.info(f"Validating face in {file.filename}, image shape: {image.shape}")
                bboxes, kpss = detector.detect(image, max_num=1)
                logging.info(f"Face detection result for {file.filename}: {len(bboxes)} bboxes, {len(kpss)} keypoints")
                
                if len(kpss) == 0:
                    invalid_files.append({
                        'filename': file.filename,
                        'error': 'No face detected in image',
                        'filepath': temp_filepath
                    })
                    logging.warning(f"No face detected in {file.filename}")
                    continue
                
                logging.info(f"Face detected successfully in {file.filename}")
                
                # Crop the face from the image
                bbox = bboxes[0]  # Use first detected face
                x1, y1, x2, y2 = bbox[:4].astype(int)
                
                # Add padding around the face (10% on each side)
                height, width = image.shape[:2]
                padding = 0.1
                face_width = x2 - x1
                face_height = y2 - y1
                
                pad_x = int(face_width * padding)
                pad_y = int(face_height * padding)
                
                # Calculate crop coordinates with padding
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(width, x2 + pad_x)
                crop_y2 = min(height, y2 + pad_y)
                
                # Crop the face
                face_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Save the cropped face
                crop_filename = f"cropped_{filename}"
                crop_filepath = os.path.join(temp_dir, crop_filename)
                cv2.imwrite(crop_filepath, face_crop)
                
                logging.info(f"Cropped face from {file.filename}: {face_crop.shape} -> {crop_filepath}")
                
                # Valid image with face detected and cropped
                valid_files.append({
                    'filename': file.filename,
                    'filepath': temp_filepath,  # Original image path
                    'crop_filepath': crop_filepath,  # Cropped face path
                    'image': image,  # Original image
                    'face_crop': face_crop,  # Cropped face
                    'kpss': kpss[0],  # Use first detected face
                    'bbox': bbox,  # Bounding box info
                    'crop_coords': (crop_x1, crop_y1, crop_x2, crop_y2)
                })
                local_files.extend([temp_filepath, crop_filepath])
        
        # If ANY files are invalid, return error and don't process ANY files
        if invalid_files:
            return jsonify({
                'success': False,
                'response_code': 'FIE1007',
                'message': 'Face validation failed - one or more images have no detectable faces',
                'error': 'Face validation failed',
                'details': 'One or more uploaded images have no faces or are invalid. Upload cancelled.',
                'invalid_files': [
                    {
                        'filename': item['filename'],
                        'error': item['error']
                    } for item in invalid_files
                ],
                'valid_files': len(valid_files),
                'invalid_files_count': len(invalid_files),
                'total_files': len(files)
            }), 400
        
        # Check if this person ID already exists in the database
        existing_person_check = None
        existing_identifiers = [name for name in face_db.metadata if name == identifier]
        if existing_identifiers:
            existing_person_check = identifier
            logging.info(f"Person {identifier} already exists in database with {len(existing_identifiers)} images")
        else:
            logging.info(f"Person {identifier} is new - not found in database")
        
        # Second pass: Process only valid files with faces
        for valid_file in valid_files:
            try:
                image = valid_file['image']  # Original image for embedding extraction
                face_crop = valid_file['face_crop']  # Cropped face for storage
                kps = valid_file['kpss']
                filename = valid_file['filename']
                
                logging.info(f"Processing {filename}: original {image.shape}, cropped {face_crop.shape}")
                
                # Get embedding from the cropped face image
                # For cropped face, we need to adjust keypoints or use center-based approach
                # Since the face is already cropped and centered, we can use a simplified approach
                crop_height, crop_width = face_crop.shape[:2]
                
                # Create approximate keypoints for the cropped face (centered)
                # ArcFace typically expects 5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
                center_x, center_y = crop_width // 2, crop_height // 2
                
                # Approximate keypoint positions for a centered face crop
                adjusted_kps = np.array([
                    [center_x - crop_width * 0.15, center_y - crop_height * 0.1],  # left eye
                    [center_x + crop_width * 0.15, center_y - crop_height * 0.1],  # right eye  
                    [center_x, center_y],                                          # nose
                    [center_x - crop_width * 0.1, center_y + crop_height * 0.15], # left mouth
                    [center_x + crop_width * 0.1, center_y + crop_height * 0.15]  # right mouth
                ])
                
                # Generate embedding from cropped face
                embedding = recognizer.get_embedding(face_crop, adjusted_kps)
                logging.info(f"Generated embedding from cropped face: {embedding.shape}")
                
                # Enhance embedding to target dimensions if needed
                if EMBEDDING_SIZE == 1024 and embedding.shape[0] == 512:
                    embedding = enhance_embedding_to_1024(embedding)
                elif EMBEDDING_SIZE == 768 and embedding.shape[0] == 512:
                    embedding = enhance_embedding_to_768(embedding)
                
                # Check behavior based on whether person exists
                if existing_person_check:
                    # Person exists - check similarity with existing images
                    logging.info(f"Checking similarity for existing person {identifier}")
                    duplicate_check = face_db.search(embedding, threshold=0.7)  # 70% threshold for existing person
                    duplicate_identifier, similarity = duplicate_check
                    logging.info(f"Existing person similarity check for {filename}: {duplicate_identifier} (similarity: {similarity:.3f})")
                    
                    if duplicate_identifier == identifier and similarity > 0.7:
                        # High similarity with same person - allow update
                        embeddings.append(embedding)
                        added_faces += 1
                        logging.info(f"UPDATE ALLOWED: High similarity ({similarity:.3f}) for existing person {identifier}")
                    else:
                        # Low similarity or different person - reject update
                        return jsonify({
                            'success': False,
                            'response_code': 'FIE1011',
                            'message': 'Update rejected - insufficient similarity with existing images',
                            'error': f'New image has {similarity:.1%} similarity with existing images for {identifier}. Minimum 70% required for updates.',
                            'identifier': identifier,
                            'filename': filename,
                            'similarity_score': float(similarity),
                            'required_similarity': 0.7
                        }), 400
                else:
                    # New person - check for duplicates with other people
                    logging.info(f"Checking for duplicates with other people for new person {identifier}")
                    duplicate_check = face_db.search(embedding, threshold=0.5)  # 50% threshold for new person
                    duplicate_identifier, similarity = duplicate_check
                    logging.info(f"New person duplicate check for {filename}: {duplicate_identifier} (similarity: {similarity:.3f})")
                    
                    if duplicate_identifier != "Unknown" and similarity > 0.5:
                        # Different person duplicate detected
                        if "_" in duplicate_identifier:
                            parts = duplicate_identifier.rsplit("_", 1)
                            if len(parts) == 2:
                                duplicate_name = parts[0].replace("_", " ")
                                duplicate_id = parts[1]
                            else:
                                duplicate_name = duplicate_identifier
                                duplicate_id = None
                        else:
                            duplicate_name = duplicate_identifier
                            duplicate_id = None
                        
                        # Get image URLs from Spaces if available
                        duplicate_images = []
                        if spaces_manager:
                            duplicate_images = spaces_manager.get_images_urls(duplicate_identifier)
                        
                        duplicate_info = {
                            'filename': filename,
                            'similarity_score': float(similarity),
                            'duplicate_with': {
                                'identifier': duplicate_identifier,
                                'person_name': duplicate_name,
                                'person_id': duplicate_id,
                                'image_urls': duplicate_images
                            }
                        }
                        duplicates_found.append(duplicate_info)
                        logging.info(f"NEW PERSON DUPLICATE: {filename} matches existing person {duplicate_identifier} (similarity: {similarity:.3f})")
                    else:
                        # No duplicate found for new person, add to embeddings
                        embeddings.append(embedding)
                        added_faces += 1
                        logging.info(f"NEW PERSON: Added new face for {identifier} from {filename} (no duplicates found)")
                    
            except Exception as e:
                errors.append(f"Error processing {valid_file['filename']}: {str(e)}")
                logging.error(f"Error processing {valid_file['filename']}: {e}")
        
        # If duplicates found, consolidate and return duplicate information
        if duplicates_found:
            # Group duplicates by identifier to avoid multiple entries for same person
            grouped_duplicates = {}
            
            for duplicate in duplicates_found:
                dup_identifier = duplicate['duplicate_with']['identifier']
                
                if dup_identifier not in grouped_duplicates:
                    # First occurrence of this duplicate person
                    grouped_duplicates[dup_identifier] = {
                        'duplicate_with': duplicate['duplicate_with'],
                        'uploaded_files': [duplicate['filename']],
                        'max_similarity': duplicate['similarity_score'],
                        'min_similarity': duplicate['similarity_score'],
                        'avg_similarity': duplicate['similarity_score'],
                        'duplicate_count': 1
                    }
                else:
                    # Additional files matching same person
                    existing = grouped_duplicates[dup_identifier]
                    existing['uploaded_files'].append(duplicate['filename'])
                    existing['max_similarity'] = max(existing['max_similarity'], duplicate['similarity_score'])
                    existing['min_similarity'] = min(existing['min_similarity'], duplicate['similarity_score'])
                    existing['duplicate_count'] += 1
                    
                    # Recalculate average
                    all_similarities = [dup['similarity_score'] for dup in duplicates_found 
                                      if dup['duplicate_with']['identifier'] == dup_identifier]
                    existing['avg_similarity'] = sum(all_similarities) / len(all_similarities)
            
            # Convert to list format
            consolidated_duplicates = []
            for identifier, duplicate_info in grouped_duplicates.items():
                consolidated_duplicates.append({
                    'duplicate_with': duplicate_info['duplicate_with'],
                    'uploaded_files': duplicate_info['uploaded_files'],
                    'similarity_stats': {
                        'max_similarity': duplicate_info['max_similarity'],
                        'min_similarity': duplicate_info['min_similarity'], 
                        'avg_similarity': duplicate_info['avg_similarity']
                    },
                    'files_matched': duplicate_info['duplicate_count']
                })
            
            return jsonify({
                'success': False,
                'response_code': 'FIE1008',
                'message': 'Duplicate faces detected - faces already exist for different person(s)',
                'duplicate_detected': True,
                'details': f'Duplicate faces detected for {len(grouped_duplicates)} existing person(s). Upload cancelled.',
                'duplicates': consolidated_duplicates,
                'unique_duplicates': len(grouped_duplicates),
                'total_files_matched': len(duplicates_found)
            }), 409  # 409 Conflict status code
        
        # Upload cropped faces to Spaces if available and we have processed images
        spaces_upload_result = None
        if spaces_manager and embeddings:  # Only upload if we have successfully processed embeddings
            # Upload only the cropped face images to Spaces
            file_objects = []
            for valid_file in valid_files:
                if 'crop_filepath' in valid_file and os.path.exists(valid_file['crop_filepath']):
                    with open(valid_file['crop_filepath'], 'rb') as f:
                        file_content = f.read()
                        # Create a file-like object
                        from io import BytesIO
                        file_obj = BytesIO(file_content)
                        # Use original filename for the cropped version
                        original_name, ext = os.path.splitext(valid_file['filename'])
                        file_obj.filename = f"cropped_{original_name}{ext}"
                        file_obj.seek(0)
                        file_objects.append(file_obj)
            
            if file_objects:
                spaces_upload_result = spaces_manager.upload_images(identifier, file_objects)
                
                if spaces_upload_result['success']:
                    logging.info(f"Uploaded {spaces_upload_result['uploaded_count']} cropped face images to Spaces for {identifier}")
                else:
                    errors.append(f"Spaces upload failed: {spaces_upload_result.get('error', 'Unknown error')}")
            else:
                logging.warning("No cropped images to upload to Spaces")
                    
        
        # Add embeddings to vector database
        if embeddings:
            # Add all embeddings for this person
            for embedding in embeddings:
                face_db.add_face(embedding, identifier)
            
            # Save updated database
            face_db.save()
            logging.info(f"Added {len(embeddings)} embeddings to vector database for {identifier}")
        
        # Clean up local temporary files
        for local_file in local_files:
            if os.path.exists(local_file):
                os.remove(local_file)
        # Remove temp directory if empty
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
        logging.info(f"Cleaned up {len(local_files)} temporary files for {identifier}")
        
        # Determine if this was a creation or update
        is_new_person = not existing_person_check
        status_code = 201 if is_new_person else 200
        
        # Prepare response
        response = {
            'success': True,
            'response_code': 'FIS1001' if is_new_person else 'FIS1002',
            'message': 'New person created successfully' if is_new_person else 'Existing person updated successfully',
            'operation': 'created' if is_new_person else 'updated',
            'identifier': identifier,
            'name': name,
            'id': person_id,
            'added_faces': added_faces,
            'total_faces_in_db': 0,  # Will be updated with actual count from database
            'errors': errors
        }
        
        # Add Spaces upload info if available
        if spaces_upload_result:
            response['spaces_upload'] = {
                'success': spaces_upload_result['success'],
                'uploaded_count': spaces_upload_result.get('uploaded_count', 0),
                'folder_path': spaces_upload_result.get('folder_path', ''),
                'urls': [file['url'] for file in spaces_upload_result.get('uploaded_files', [])],
                'image_type': 'cropped_faces'
            }
        
        # Add crop information
        
        if errors:
            response['warning'] = f"Some files had issues: {len(errors)} errors"
        
        return jsonify(response), status_code
        
    except Exception as e:
        logging.error(f"Upload endpoint error: {e}")
        return jsonify({
            'success': False,
            'response_code': 'FIE1010',
            'message': 'Upload operation failed due to server error',
            'error': f'Server error: {str(e)}'
        }), 500
        
    finally:
        # Ensure cleanup happens even if there's an error
        for local_file in local_files:
            if os.path.exists(local_file):
                os.remove(local_file)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

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
    if 'file' not in request.files:
        return jsonify({'success': False, 'response_code': 'FIE2001', 'message': 'No file in recognition request', 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'response_code': 'FIE2002', 'message': 'Empty filename in recognition request', 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'response_code': 'FIE2003', 'message': 'Unsupported file format for recognition', 'error': 'Invalid file type'}), 400
    
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

def process_image(image_path: str, similarity_threshold: float, max_faces: int) -> Dict[str, Any]:
    """Process a single image for face recognition"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image file")
    
    logging.info(f"Processing image: {image_path}, max_faces: {max_faces}, threshold: {similarity_threshold}")
    
    # Detect faces
    bboxes, kpss = detector.detect(image, max_num=max_faces if max_faces > 0 else 0)
    logging.info(f"Detected {len(bboxes)} faces")
    
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
        # Crop face for consistent embedding generation
        x1, y1, x2, y2 = bbox[:4].astype(int)
        
        # Add padding around the face (10% on each side)
        height, width = image.shape[:2]
        padding = 0.1
        face_width = x2 - x1
        face_height = y2 - y1
        
        pad_x = int(face_width * padding)
        pad_y = int(face_height * padding)
        
        # Calculate crop coordinates with padding
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(width, x2 + pad_x)
        crop_y2 = min(height, y2 + pad_y)
        
        # Crop the face
        face_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Create adjusted keypoints for cropped face
        crop_height, crop_width = face_crop.shape[:2]
        center_x, center_y = crop_width // 2, crop_height // 2
        
        adjusted_kps = np.array([
            [center_x - crop_width * 0.15, center_y - crop_height * 0.1],  # left eye
            [center_x + crop_width * 0.15, center_y - crop_height * 0.1],  # right eye  
            [center_x, center_y],                                          # nose
            [center_x - crop_width * 0.1, center_y + crop_height * 0.15], # left mouth
            [center_x + crop_width * 0.1, center_y + crop_height * 0.15]  # right mouth
        ])
        
        # Generate embedding from cropped face
        embedding = recognizer.get_embedding(face_crop, adjusted_kps)
        
        # Enhance embedding to target dimensions if needed
        if EMBEDDING_SIZE == 1024 and embedding.shape[0] == 512:
            embedding = enhance_embedding_to_1024(embedding)
        elif EMBEDDING_SIZE == 768 and embedding.shape[0] == 512:
            embedding = enhance_embedding_to_768(embedding)
        
        embeddings.append(embedding)
        # Convert bbox to list for JSON serialization
        face_boxes.append(bbox.tolist())
    
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
    for i, ((identifier, similarity), bbox) in enumerate(zip(search_results, face_boxes)):
        # Parse identifier to extract name and id
        if identifier != "Unknown" and "_" in identifier:
            # Split identifier back to name and id
            parts = identifier.rsplit("_", 1)  # Split from right to handle names with underscores
            if len(parts) == 2:
                person_name = parts[0].replace("_", " ")  # Convert underscores back to spaces in name
                person_id = parts[1]
            else:
                person_name = identifier
                person_id = None
        else:
            person_name = identifier
            person_id = None
        
        result = {
            'face_id': i,
            'identifier': identifier,
            'person_name': person_name,
            'person_id': person_id,
            'similarity_score': float(similarity),
            'bounding_box': {
                'x1': int(bbox[0]),
                'y1': int(bbox[1]),
                'x2': int(bbox[2]),
                'y2': int(bbox[3])
            },
            'confidence': float(bbox[4]) if len(bbox) > 4 else None
        }
        results.append(result)
    
    return {
        'success': True,
        'response_code': 'FIS2001',
        'message': 'Image face recognition completed successfully',
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
                        # Crop face for consistent embedding generation
                        x1, y1, x2, y2 = bbox[:4].astype(int)
                        
                        # Add padding around the face (10% on each side)
                        height, width = frame.shape[:2]
                        padding = 0.1
                        face_width = x2 - x1
                        face_height = y2 - y1
                        
                        pad_x = int(face_width * padding)
                        pad_y = int(face_height * padding)
                        
                        # Calculate crop coordinates with padding
                        crop_x1 = max(0, x1 - pad_x)
                        crop_y1 = max(0, y1 - pad_y)
                        crop_x2 = min(width, x2 + pad_x)
                        crop_y2 = min(height, y2 + pad_y)
                        
                        # Crop the face
                        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        # Create adjusted keypoints for cropped face
                        crop_height, crop_width = face_crop.shape[:2]
                        center_x, center_y = crop_width // 2, crop_height // 2
                        
                        adjusted_kps = np.array([
                            [center_x - crop_width * 0.15, center_y - crop_height * 0.1],  # left eye
                            [center_x + crop_width * 0.15, center_y - crop_height * 0.1],  # right eye  
                            [center_x, center_y],                                          # nose
                            [center_x - crop_width * 0.1, center_y + crop_height * 0.15], # left mouth
                            [center_x + crop_width * 0.1, center_y + crop_height * 0.15]  # right mouth
                        ])
                        
                        # Generate embedding from cropped face
                        embedding = recognizer.get_embedding(face_crop, adjusted_kps)
                        
                        # Enhance embedding to target dimensions if needed
                        if EMBEDDING_SIZE == 1024 and embedding.shape[0] == 512:
                            embedding = enhance_embedding_to_1024(embedding)
                        elif EMBEDDING_SIZE == 768 and embedding.shape[0] == 512:
                            embedding = enhance_embedding_to_768(embedding)
                        
                        embeddings.append(embedding)
                        face_boxes.append(bbox.tolist())
                    
                    if embeddings:
                        # Batch search for all faces
                        search_results = face_db.batch_search(embeddings, similarity_threshold)
                        
                        # Format frame results
                        faces = []
                        for i, ((identifier, similarity), bbox) in enumerate(zip(search_results, face_boxes)):
                            # Parse identifier to extract name and id
                            if identifier != "Unknown" and "_" in identifier:
                                # Split identifier back to name and id
                                parts = identifier.rsplit("_", 1)  # Split from right to handle names with underscores
                                if len(parts) == 2:
                                    person_name = parts[0].replace("_", " ")  # Convert underscores back to spaces in name
                                    person_id = parts[1]
                                else:
                                    person_name = identifier
                                    person_id = None
                            else:
                                person_name = identifier
                                person_id = None
                            
                            faces.append({
                                'face_id': i,
                                'identifier': identifier,
                                'person_name': person_name,
                                'person_id': person_id,
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
            identifier = face['identifier']
            if identifier != 'Unknown':
                if identifier not in all_persons:
                    all_persons[identifier] = {
                        'identifier': identifier,
                        'person_name': face['person_name'],
                        'person_id': face['person_id'],
                        'max_similarity': face['similarity_score'],
                        'appearances': 1,
                        'first_seen_frame': frame_result['frame_number'],
                        'last_seen_frame': frame_result['frame_number']
                    }
                else:
                    all_persons[identifier]['max_similarity'] = max(
                        all_persons[identifier]['max_similarity'],
                        face['similarity_score']
                    )
                    all_persons[identifier]['appearances'] += 1
                    all_persons[identifier]['last_seen_frame'] = frame_result['frame_number']
    
    return {
        'success': True,
        'response_code': 'FIS2002',
        'message': 'Video face recognition completed successfully',
        'type': 'video',
        'total_frames': frame_count,
        'processed_frames': len(frame_results),
        'unique_persons': list(all_persons.values()),
        'frame_by_frame_results': frame_results
    }

@app.route('/validate_faces', methods=['POST'])
def validate_faces():
    """
    Test endpoint to validate if uploaded images contain faces
    Returns detailed face detection information without storing anything
    """
    try:
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'response_code': 'FIE3001',
                'message': 'No files provided for validation',
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({
                'success': False,
                'response_code': 'FIE3002',
                'message': 'No files selected for validation',
                'error': 'No files selected'
            }), 400
        
        validation_results = []
        
        for i, file in enumerate(files):
            if file and allowed_file(file.filename) and not is_video_file(file.filename):
                # Save temporarily for validation
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                    file.save(temp_file.name)
                    temp_filepath = temp_file.name
                
                try:
                    # Read and validate image
                    image = cv2.imread(temp_filepath)
                    if image is None:
                        validation_results.append({
                            'filename': file.filename,
                            'valid': False,
                            'error': 'Could not read image file',
                            'faces_detected': 0
                        })
                        continue
                    
                    # Detect faces
                    bboxes, kpss = detector.detect(image, max_num=0)  # Detect all faces
                    
                    face_info = []
                    for j, (bbox, kps) in enumerate(zip(bboxes, kpss)):
                        face_info.append({
                            'face_id': j,
                            'bounding_box': {
                                'x1': int(bbox[0]),
                                'y1': int(bbox[1]),
                                'x2': int(bbox[2]),
                                'y2': int(bbox[3])
                            },
                            'confidence': float(bbox[4]) if len(bbox) > 4 else None
                        })
                    
                    validation_results.append({
                        'filename': file.filename,
                        'valid': len(kpss) > 0,
                        'faces_detected': len(kpss),
                        'face_details': face_info,
                        'image_size': f"{image.shape[1]}x{image.shape[0]}"
                    })
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_filepath):
                        os.unlink(temp_filepath)
        
        valid_count = sum(1 for result in validation_results if result['valid'])
        
        return jsonify({
            'success': True,
            'response_code': 'FIS3002',
            'message': 'Face validation completed',
            'total_files': len(validation_results),
            'valid_files': valid_count,
            'invalid_files': len(validation_results) - valid_count,
            'validation_results': validation_results
        }), 200
        
    except Exception as e:
        logging.error(f"Face validation error: {e}")
        return jsonify({
            'success': False,
            'response_code': 'FIE3003',
            'message': 'Face validation operation failed',
            'error': f'Validation error: {str(e)}'
        }), 500

@app.route('/list_persons', methods=['GET'])
def list_persons():
    """List all persons in the database"""
    unique_names = list(set(face_db.metadata))
    name_counts = {name: face_db.metadata.count(name) for name in unique_names}
    
    return jsonify({
        'success': True,
        'response_code': 'FIS3001',
        'message': 'Person list retrieved successfully',
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

@app.route('/flush_database', methods=['DELETE'])
def flush_database():
    """
    Flush/delete everything from the database
    
    WARNING: This will permanently delete all face data!
    """
    try:
        initial_count = face_db.index.ntotal
        
        # For Milvus - use the flush_all method
        face_db.flush_all()
        
        return jsonify({
            'success': True,
            'response_code': 'FIS4001',
            'message': 'Database flush operation completed successfully',
            'details': 'All face data has been permanently deleted from the database',
            'deleted_faces': initial_count,
            'remaining_faces': 0
        }), 200
        
    except Exception as e:
        logging.error(f"Failed to flush database: {e}")
        return jsonify({
            'success': False,
            'response_code': 'FIE4001',
            'message': 'Database flush operation failed',
            'error': f'Failed to flush database: {str(e)}'
        }), 500

@app.route('/delete_record/<person_id>', methods=['DELETE'])
def delete_record(person_id):
    """
    Delete a specific person record from both database and DigitalOcean Spaces
    
    This endpoint accepts just the person ID (e.g., "12345") and finds the matching identifier
    
    Args:
        person_id: Person ID (e.g., "12345") - will search for identifiers ending with "_12345"
    
    Returns:
        - JSON response with detailed deletion results
        - Information about what was deleted from database and Spaces
    """
    try:
        logging.info(f"Starting deletion process for person ID: {person_id}")
        
        # Find matching identifier(s) that end with the person_id
        matching_identifiers = []
        initial_db_count = 0
        
        try:
            # Get initial count from database
            initial_db_count = face_db.index.ntotal
            
            # Search for identifiers that end with the person_id
            if hasattr(face_db, 'metadata'):
                for identifier in face_db.metadata:
                    if identifier.endswith(f"_{person_id}"):
                        matching_identifiers.append(identifier)
                
        except Exception as e:
            logging.warning(f"Could not check database for person existence: {e}")
        
        if not matching_identifiers:
                return jsonify({
                    'success': False,
                'response_code': 'FIE5003',
                    'message': 'Person not found in database',
                'error': f'No person found with ID {person_id}',
                'person_id': person_id,
                'database_deleted': 0,
                'spaces_deleted': 0
                }), 404
            
        # If multiple matches, log warning but proceed with deletion
        if len(matching_identifiers) > 1:
            logging.warning(f"Multiple identifiers found for ID {person_id}: {matching_identifiers}")
            logging.info(f"Will delete all {len(matching_identifiers)} matching identifiers")
        
        logging.info(f"Found matching identifier(s): {matching_identifiers}")
        
        # Delete from Milvus database - handle all matching identifiers
        db_deleted_count = 0
        db_success = True
        db_deletion_details = {}
        
        for identifier_to_delete in matching_identifiers:
            try:
                deleted_count = face_db.delete_person(identifier_to_delete)
                db_deleted_count += deleted_count
                db_deletion_details[identifier_to_delete] = deleted_count
                logging.info(f"Successfully deleted {deleted_count} records from database for {identifier_to_delete}")
            except Exception as e:
                logging.error(f"Failed to delete {identifier_to_delete} from database: {e}")
                db_success = False
                db_deletion_details[identifier_to_delete] = 0
        
        # Delete from DigitalOcean Spaces - handle all matching identifiers
        spaces_deleted_count = 0
        spaces_success = True
        spaces_details = {}
        
        if spaces_manager:
            for identifier_to_delete in matching_identifiers:
                try:
                    spaces_result = spaces_manager.delete_images(identifier_to_delete)
                    if spaces_result.get('success', False):
                        spaces_deleted_count += spaces_result.get('deleted_count', 0)
                        spaces_details[identifier_to_delete] = spaces_result
                        logging.info(f"Spaces deletion result for {identifier_to_delete}: {spaces_result}")
                    else:
                        spaces_success = False
                        spaces_details[identifier_to_delete] = spaces_result
                except Exception as e:
                    logging.error(f"Failed to delete {identifier_to_delete} from Spaces: {e}")
                    spaces_success = False
                    spaces_details[identifier_to_delete] = {'success': False, 'error': str(e)}
        else:
            logging.warning("Spaces manager not available - skipping Spaces deletion")
        
        # Determine overall success
        overall_success = db_success and (spaces_success or not spaces_manager)
        
        # Get final database count
        final_db_count = 0
        try:
            final_db_count = face_db.index.ntotal
        except Exception as e:
            logging.warning(f"Could not get final database count: {e}")
        
        # Prepare response
        response = {
            'success': overall_success,
            'response_code': 'FIS5002' if overall_success else 'FIE5004',
            'message': 'Record deletion completed successfully' if overall_success else 'Record deletion completed with errors',
            'person_id': person_id,
            'matching_identifiers': matching_identifiers,
            'deletion_summary': {
                'database': {
                    'success': db_success,
                    'deleted_count': db_deleted_count,
                    'initial_count': initial_db_count,
                    'final_count': final_db_count,
                    'per_identifier': db_deletion_details
                },
                'spaces': {
                    'success': spaces_success,
                    'deleted_count': spaces_deleted_count,
                    'manager_available': spaces_manager is not None,
                    'per_identifier': spaces_details
                }
            },
            'total_deleted': {
                'database_records': db_deleted_count,
                'spaces_images': spaces_deleted_count,
                'combined': db_deleted_count + spaces_deleted_count
            }
        }
        
        # Add warnings if there were partial failures
        warnings = []
        if not db_success:
            warnings.append("Database deletion failed")
        if spaces_manager and not spaces_success:
            warnings.append("Spaces deletion failed")
        
        if warnings:
            response['warnings'] = warnings
        
        status_code = 200 if overall_success else 207  # 207 Multi-Status for partial success
        
        logging.info(f"Deletion completed for person ID {person_id}: DB={db_deleted_count}, Spaces={spaces_deleted_count}")
        
        return jsonify(response), status_code
        
    except Exception as e:
        logging.error(f"Unexpected error during deletion of person ID {person_id}: {e}")
        return jsonify({
            'success': False,
            'response_code': 'FIE5005',
            'message': 'Unexpected error during deletion',
            'error': f'Deletion failed: {str(e)}',
            'person_id': person_id
        }), 500

@app.route('/delete_records', methods=['POST'])
def delete_records():
    """
    Bulk delete multiple person records from both database and DigitalOcean Spaces
    
    Expected JSON payload:
    {
        "person_ids": ["12345", "67890", "11111"]
    }
    
    Returns:
        - JSON response with detailed deletion results for each person ID
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'response_code': 'FIE5006',
                'message': 'Request must be JSON',
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        if 'person_ids' not in data:
            return jsonify({
                'success': False,
                'response_code': 'FIE5007',
                'message': 'Missing required field: person_ids',
                'error': 'person_ids array is required'
            }), 400
        
        person_ids = data['person_ids']
        
        if not isinstance(person_ids, list):
            return jsonify({
                'success': False,
                'response_code': 'FIE5008',
                'message': 'Invalid person_ids format',
                'error': 'person_ids must be an array'
            }), 400
        
        if not person_ids:
            return jsonify({
                'success': False,
                'response_code': 'FIE5009',
                'message': 'Empty person_ids list',
                'error': 'At least one person ID must be provided'
            }), 400
        
        if len(person_ids) > 100:  # Limit bulk operations
            return jsonify({
                'success': False,
                'response_code': 'FIE5010',
                'message': 'Too many person IDs',
                'error': 'Maximum 100 person IDs allowed per request'
            }), 400
        
        logging.info(f"Starting bulk deletion for {len(person_ids)} person IDs: {person_ids}")
        
        # Process each person ID
        results = []
        total_db_deleted = 0
        total_spaces_deleted = 0
        successful_deletions = 0
        failed_deletions = 0
        
        for person_id in person_ids:
            try:
                # Find matching identifiers for this person ID
                matching_identifiers = []
                if hasattr(face_db, 'metadata'):
                    for identifier in face_db.metadata:
                        if identifier.endswith(f"_{person_id}"):
                            matching_identifiers.append(identifier)
                
                if not matching_identifiers:
                    results.append({
                        'person_id': person_id,
                        'success': False,
                        'error': 'Person not found in database',
                        'matching_identifiers': [],
                        'database_deleted': 0,
                        'spaces_deleted': 0
                    })
                    failed_deletions += 1
                    continue
                
                # Delete from database - handle all matching identifiers
                db_deleted = 0
                db_success = True
                db_details = {}
                
                for identifier in matching_identifiers:
                    try:
                        deleted_count = face_db.delete_person(identifier)
                        db_deleted += deleted_count
                        db_details[identifier] = deleted_count
                        total_db_deleted += deleted_count
                    except Exception as e:
                        logging.error(f"Failed to delete {identifier} from database: {e}")
                        db_success = False
                        db_details[identifier] = 0
                
                # Delete from Spaces - handle all matching identifiers
                spaces_deleted = 0
                spaces_success = True
                spaces_details = {}
                
                if spaces_manager:
                    for identifier in matching_identifiers:
                        try:
                            spaces_result = spaces_manager.delete_images(identifier)
                            if spaces_result.get('success', False):
                                spaces_deleted += spaces_result.get('deleted_count', 0)
                                total_spaces_deleted += spaces_result.get('deleted_count', 0)
                                spaces_details[identifier] = spaces_result
                            else:
                                spaces_success = False
                                spaces_details[identifier] = spaces_result
                        except Exception as e:
                            logging.error(f"Failed to delete {identifier} from Spaces: {e}")
                            spaces_success = False
                            spaces_details[identifier] = {'success': False, 'error': str(e)}
                else:
                    spaces_success = True  # No Spaces manager, consider success
                
                # Determine individual success
                individual_success = db_success and spaces_success
                
                if individual_success:
                    successful_deletions += 1
                else:
                    failed_deletions += 1
                
                results.append({
                    'person_id': person_id,
                    'success': individual_success,
                    'matching_identifiers': matching_identifiers,
                    'database_deleted': db_deleted,
                    'spaces_deleted': spaces_deleted,
                    'database_success': db_success,
                    'spaces_success': spaces_success,
                    'database_details': db_details,
                    'spaces_details': spaces_details
                })
                
            except Exception as e:
                logging.error(f"Error processing person ID {person_id}: {e}")
                results.append({
                    'person_id': person_id,
                    'success': False,
                    'error': str(e),
                    'matching_identifiers': [],
                    'database_deleted': 0,
                    'spaces_deleted': 0
                })
                failed_deletions += 1
        
        # Determine overall success
        overall_success = failed_deletions == 0
        
        response = {
            'success': overall_success,
            'response_code': 'FIS5003' if overall_success else 'FIE5011',
            'message': f'Bulk deletion completed: {successful_deletions} successful, {failed_deletions} failed',
            'summary': {
                'total_requested': len(person_ids),
                'successful_deletions': successful_deletions,
                'failed_deletions': failed_deletions,
                'total_database_deleted': total_db_deleted,
                'total_spaces_deleted': total_spaces_deleted,
                'total_combined_deleted': total_db_deleted + total_spaces_deleted
            },
            'results': results
        }
        
        status_code = 200 if overall_success else 207  # 207 Multi-Status for partial success
        
        logging.info(f"Bulk deletion completed: {successful_deletions}/{len(person_ids)} successful")
        
        return jsonify(response), status_code
        
    except Exception as e:
        logging.error(f"Unexpected error during bulk deletion: {e}")
        return jsonify({
            'success': False,
            'response_code': 'FIE5012',
            'message': 'Unexpected error during bulk deletion',
            'error': f'Bulk deletion failed: {str(e)}'
        }), 500

@app.route('/delete_person/<identifier>', methods=['DELETE'])
def delete_person(identifier):
    """
    Delete a specific person from the database
    
    Args:
        identifier: Person identifier (e.g., "John_Doe_12345")
    """
    try:
        # For Milvus - use the delete_person method
        deleted_count = face_db.delete_person(identifier)
        
        # Also delete from Spaces if available
        spaces_deleted = 0
        if spaces_manager:
            spaces_result = spaces_manager.delete_images(identifier)
            if spaces_result['success']:
                spaces_deleted = spaces_result['deleted_count']
        
        return jsonify({
            'success': True,
            'response_code': 'FIS5001',
            'message': 'Person deletion completed successfully',
            'details': f'Person {identifier} deleted successfully',
            'deleted_faces': deleted_count,
            'deleted_images_from_spaces': spaces_deleted,
            'remaining_faces': 0  # Will be updated with actual count from database
        }), 200
        
    except Exception as e:
        logging.error(f"Failed to delete person {identifier}: {e}")
        return jsonify({
            'success': False,
            'response_code': 'FIE5002',
            'message': 'Person deletion operation failed',
            'error': f'Failed to delete person: {str(e)}'
        }), 500

if __name__ == '__main__':
    if not initialize_models():
        print("Failed to initialize models. Exiting.")
        exit(1)
    
    print("Face Recognition API Server Starting...")
    print(f"Configuration loaded from: config.env")
    print(f"Embedding size: {EMBEDDING_SIZE}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Database backend: Milvus")
    print(f"Milvus: {MILVUS_HOST}:{MILVUS_PORT} (collection: {MILVUS_COLLECTION_NAME})")
    print(f"Spaces manager: {'Enabled' if spaces_manager else 'Disabled'}")
    print("Available endpoints:")
    print("  GET    /health - Health check")
    print("  POST   /upload - Upload face images")
    print("  POST   /recognize - Recognize faces in image/video")
    print("  POST   /validate_faces - Test face detection in images")
    print("  GET    /list_persons - List all persons in database")
    print("  DELETE /flush_database - Delete all faces from database")
    print("  DELETE /delete_person/<identifier> - Delete specific person (legacy)")
    print("  DELETE /delete_record/<person_id> - Delete specific person from DB and Spaces (by ID)")
    print("  POST   /delete_records - Bulk delete multiple persons from DB and Spaces (by IDs)")
    
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
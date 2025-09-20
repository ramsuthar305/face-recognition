#!/usr/bin/env python3
"""
Real-time CCTV Face Recognition System
Supports multiple camera sources: webcam, IP cameras, RTSP streams
"""

import cv2
import time
import threading
import queue
import argparse
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json
import os

from database import FaceDatabase
from models import SCRFD, ArcFace
from utils.logging import setup_logging
from utils.helpers import draw_bbox_info, draw_bbox

# Setup logging
setup_logging(log_to_file=True)

class CCTVFaceRecognition:
    def __init__(self, 
                 det_weight: str = "./weights/det_10g.onnx",
                 rec_weight: str = "./weights/w600k_mbf.onnx",
                 db_path: str = "./database/face_database",
                 confidence_thresh: float = 0.5,
                 similarity_thresh: float = 0.4,
                 max_faces: int = 0,
                 frame_skip: int = 1,
                 save_detections: bool = False,
                 detection_log: str = "./detections.json"):
        
        self.confidence_thresh = confidence_thresh
        self.similarity_thresh = similarity_thresh
        self.max_faces = max_faces
        self.frame_skip = frame_skip
        self.save_detections = save_detections
        self.detection_log = detection_log
        
        # Initialize models
        logging.info("Initializing face detection and recognition models...")
        self.detector = SCRFD(det_weight, input_size=(640, 640), conf_thres=confidence_thresh)
        self.recognizer = ArcFace(rec_weight)
        
        # Initialize face database
        self.face_db = FaceDatabase(db_path=db_path, max_workers=4)
        self.face_db.load()
        logging.info(f"Loaded face database with {self.face_db.index.ntotal} faces")
        
        # Threading setup
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Colors for different persons
        self.colors = {}
        
        # Detection history for logging
        self.detections = []
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0

    def get_camera_source(self, source: str) -> cv2.VideoCapture:
        """
        Get camera source from various inputs:
        - Integer: Webcam index (0, 1, 2, etc.)
        - IP Camera: http://ip:port/stream
        - RTSP Stream: rtsp://username:password@ip:port/path
        - File: path to video file
        """
        try:
            # Try to convert to integer (webcam)
            source_int = int(source)
            logging.info(f"Using webcam source: {source_int}")
            return cv2.VideoCapture(source_int)
        except ValueError:
            # String source (IP camera, RTSP, file)
            if source.startswith(('http://', 'https://', 'rtsp://')):
                logging.info(f"Using network camera source: {source}")
            else:
                logging.info(f"Using file source: {source}")
            return cv2.VideoCapture(source)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame for face detection and recognition"""
        try:
            # Detect faces
            bboxes, kpss = self.detector.detect(frame, max_num=self.max_faces if self.max_faces > 0 else 0)
            
            if len(bboxes) == 0:
                return frame, []
            
            # Get embeddings for all faces
            embeddings = []
            processed_bboxes = []
            
            for bbox, kps in zip(bboxes, kpss):
                try:
                    *bbox_coords, conf_score = bbox.astype(np.int32)
                    embedding = self.recognizer.get_embedding(frame, kps)
                    embeddings.append(embedding)
                    processed_bboxes.append((bbox_coords, conf_score))
                except Exception as e:
                    logging.warning(f"Error processing face embedding: {e}")
                    continue
            
            if not embeddings:
                return frame, []
            
            # Batch search for all faces
            results = self.face_db.batch_search(embeddings, self.similarity_thresh)
            
            # Process results and draw bounding boxes
            detections = []
            for i, ((name, similarity), (bbox_coords, conf_score)) in enumerate(zip(results, processed_bboxes)):
                detection = {
                    'timestamp': datetime.now().isoformat(),
                    'person_name': name,
                    'similarity_score': float(similarity),
                    'confidence_score': float(conf_score),
                    'bounding_box': {
                        'x1': int(bbox_coords[0]),
                        'y1': int(bbox_coords[1]),
                        'x2': int(bbox_coords[2]),
                        'y2': int(bbox_coords[3])
                    }
                }
                detections.append(detection)
                
                # Draw bounding box and info
                if name != "Unknown":
                    if name not in self.colors:
                        self.colors[name] = (
                            np.random.randint(0, 255),
                            np.random.randint(0, 255),
                            np.random.randint(0, 255)
                        )
                    draw_bbox_info(frame, bbox_coords, similarity=similarity, name=name, color=self.colors[name])
                else:
                    draw_bbox(frame, bbox_coords, (255, 0, 0))  # Red for unknown
            
            return frame, detections
            
        except Exception as e:
            logging.error(f"Error in frame processing: {e}")
            return frame, []

    def frame_capture_thread(self, cap: cv2.VideoCapture):
        """Thread for capturing frames from camera"""
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Skip frames if queue is full
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                # Remove oldest frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except queue.Empty:
                    pass

    def frame_process_thread(self):
        """Thread for processing frames"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                # Skip frames based on frame_skip setting
                if self.frame_count % (self.frame_skip + 1) != 0:
                    self.frame_count += 1
                    continue
                
                processed_frame, detections = self.process_frame(frame)
                
                # Log detections if enabled
                if self.save_detections and detections:
                    self.detections.extend(detections)
                    # Save detections every 100 frames to avoid too frequent I/O
                    if len(self.detections) >= 100:
                        self.save_detections_to_file()
                
                # Put processed frame in result queue
                if not self.result_queue.full():
                    self.result_queue.put((processed_frame, detections))
                
                self.frame_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in frame processing thread: {e}")

    def save_detections_to_file(self):
        """Save detection results to JSON file"""
        try:
            # Load existing detections if file exists
            existing_detections = []
            if os.path.exists(self.detection_log):
                with open(self.detection_log, 'r') as f:
                    existing_detections = json.load(f)
            
            # Append new detections
            existing_detections.extend(self.detections)
            
            # Save to file
            with open(self.detection_log, 'w') as f:
                json.dump(existing_detections, f, indent=2)
            
            logging.info(f"Saved {len(self.detections)} detections to {self.detection_log}")
            self.detections = []  # Clear the buffer
            
        except Exception as e:
            logging.error(f"Error saving detections: {e}")

    def calculate_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def run(self, source: str, display: bool = True, save_video: bool = False, output_path: str = "cctv_output.mp4"):
        """Main run loop for CCTV processing"""
        # Initialize camera
        cap = self.get_camera_source(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera source: {source}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cap = cap.get(cv2.CAP_PROP_FPS)
        
        logging.info(f"Camera resolution: {width}x{height}, FPS: {fps_cap}")
        
        # Initialize video writer if saving
        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps_cap, (width, height))
            logging.info(f"Saving output to: {output_path}")
        
        # Start processing threads
        self.running = True
        
        capture_thread = threading.Thread(target=self.frame_capture_thread, args=(cap,))
        process_thread = threading.Thread(target=self.frame_process_thread)
        
        capture_thread.start()
        process_thread.start()
        
        logging.info("Starting real-time face recognition...")
        logging.info("Press 'q' to quit, 's' to save current detections")
        
        try:
            while self.running:
                try:
                    processed_frame, detections = self.result_queue.get(timeout=1.0)
                    
                    # Calculate FPS
                    self.calculate_fps()
                    
                    # Add FPS and info overlay
                    cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Faces in DB: {self.face_db.index.ntotal}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add detection count
                    if detections:
                        known_faces = sum(1 for d in detections if d['person_name'] != 'Unknown')
                        cv2.putText(processed_frame, f"Known faces: {known_faces}/{len(detections)}", (10, 110), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save frame if video output is enabled
                    if out is not None:
                        out.write(processed_frame)
                    
                    # Display frame
                    if display:
                        cv2.imshow('CCTV Face Recognition', processed_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s') and self.save_detections:
                            self.save_detections_to_file()
                            logging.info("Manual save triggered")
                
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        
        finally:
            # Cleanup
            self.running = False
            
            # Wait for threads to finish
            capture_thread.join(timeout=2.0)
            process_thread.join(timeout=2.0)
            
            # Save any remaining detections
            if self.save_detections and self.detections:
                self.save_detections_to_file()
            
            # Release resources
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            
            logging.info("CCTV processing stopped")

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time CCTV Face Recognition")
    
    parser.add_argument("--source", type=str, default="0", 
                       help="Camera source: webcam index (0,1,2) or IP camera URL or RTSP stream")
    parser.add_argument("--det-weight", type=str, default="./weights/det_10g.onnx", 
                       help="Path to detection model")
    parser.add_argument("--rec-weight", type=str, default="./weights/w600k_mbf.onnx", 
                       help="Path to recognition model")
    parser.add_argument("--db-path", type=str, default="./database/face_database", 
                       help="Path to face database")
    parser.add_argument("--confidence-thresh", type=float, default=0.5, 
                       help="Face detection confidence threshold")
    parser.add_argument("--similarity-thresh", type=float, default=0.4, 
                       help="Face recognition similarity threshold")
    parser.add_argument("--max-faces", type=int, default=0, 
                       help="Maximum faces to detect per frame (0 for unlimited)")
    parser.add_argument("--frame-skip", type=int, default=0, 
                       help="Skip N frames between processing (0 for no skip)")
    parser.add_argument("--no-display", action="store_true", 
                       help="Don't display video window")
    parser.add_argument("--save-video", action="store_true", 
                       help="Save processed video to file")
    parser.add_argument("--output", type=str, default="cctv_output.mp4", 
                       help="Output video file path")
    parser.add_argument("--save-detections", action="store_true", 
                       help="Save detection results to JSON file")
    parser.add_argument("--detection-log", type=str, default="./detections.json", 
                       help="Path to detection log file")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Convert source to int if it's a number
        try:
            args.source = int(args.source)
        except ValueError:
            pass  # Keep as string for IP cameras/RTSP
        
        # Initialize CCTV system
        cctv = CCTVFaceRecognition(
            det_weight=args.det_weight,
            rec_weight=args.rec_weight,
            db_path=args.db_path,
            confidence_thresh=args.confidence_thresh,
            similarity_thresh=args.similarity_thresh,
            max_faces=args.max_faces,
            frame_skip=args.frame_skip,
            save_detections=args.save_detections,
            detection_log=args.detection_log
        )
        
        # Run CCTV processing
        cctv.run(
            source=args.source,
            display=not args.no_display,
            save_video=args.save_video,
            output_path=args.output
        )
        
    except Exception as e:
        logging.error(f"Error in CCTV system: {e}")
        raise

if __name__ == "__main__":
    main()

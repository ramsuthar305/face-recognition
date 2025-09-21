"""
DigitalOcean Spaces Manager for Face Recognition System
Handles image storage and management in DigitalOcean Spaces
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError
from werkzeug.utils import secure_filename
import bleach
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

class SpacesManager:
    """
    DigitalOcean Spaces manager for storing and managing face images
    """
    
    def __init__(self, 
                 access_id: str = None,
                 secret_key: str = None,
                 endpoint: str = None,
                 bucket_name: str = None,
                 cdn_endpoint: str = None):
        """
        Initialize Spaces manager
        
        Args:
            access_id: DigitalOcean Spaces access ID
            secret_key: DigitalOcean Spaces secret key
            endpoint: Spaces endpoint URL
            bucket_name: Bucket/Space name
            cdn_endpoint: CDN endpoint for public URLs
        """
        self.access_id = access_id or os.getenv('SPACES_ACCESS_ID')
        self.secret_key = secret_key or os.getenv('SPACES_SECRET_KEY')
        self.endpoint = endpoint or os.getenv('SPACES_ENDPOINT')
        self.bucket_name = bucket_name or os.getenv('SPACES_BUCKET_NAME', 'gurujiapp')
        self.cdn_endpoint = cdn_endpoint or os.getenv('SPACES_BLOB_URL_ENDPOINT')
        
        if not all([self.access_id, self.secret_key, self.endpoint]):
            raise ValueError("Missing required Spaces credentials")
        
        # Initialize boto3 session
        self.session = boto3.Session()
        self.client = self.session.client(
            's3',
            region_name='sgp1',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_id,
            aws_secret_access_key=self.secret_key
        )
        
        logging.info("Spaces manager initialized successfully")

    def create_identifier_folder(self, identifier: str) -> str:
        """
        Create a folder path for the given identifier
        
        Args:
            identifier: Unique identifier (e.g., "john_doe_123")
            
        Returns:
            Cleaned folder path
        """
        # Clean the identifier to make it safe for file paths
        clean_identifier = bleach.clean(secure_filename(identifier))
        folder_path = f"face_images/{clean_identifier}"
        return folder_path

    def upload_images(self, 
                     identifier: str, 
                     image_files: List[Any], 
                     make_public: bool = True) -> Dict[str, Any]:
        """
        Upload multiple images for a person identifier
        
        Args:
            identifier: Person identifier (e.g., "john_doe_123")
            image_files: List of file objects to upload
            make_public: Whether to make files publicly accessible
            
        Returns:
            Dictionary with upload results
        """
        try:
            folder_path = self.create_identifier_folder(identifier)
            uploaded_files = []
            errors = []
            
            for i, file_obj in enumerate(image_files):
                try:
                    # Generate unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_extension = self._get_file_extension(file_obj.filename)
                    filename = f"{identifier}_{timestamp}_{i+1}{file_extension}"
                    
                    # Create full path
                    file_path = f"{folder_path}/{secure_filename(filename)}"
                    
                    # Determine content type
                    content_type = self._get_content_type(file_extension)
                    
                    # Upload to Spaces
                    upload_params = {
                        'Body': file_obj,
                        'Bucket': self.bucket_name,
                        'Key': file_path,
                        'ContentType': content_type
                    }
                    
                    if make_public:
                        upload_params['ACL'] = 'public-read'
                    
                    self.client.put_object(**upload_params)
                    
                    # Generate public URL
                    public_url = f"{self.cdn_endpoint}/{file_path}" if self.cdn_endpoint else f"{self.endpoint}/{self.bucket_name}/{file_path}"
                    
                    uploaded_files.append({
                        'filename': filename,
                        'path': file_path,
                        'url': public_url,
                        'size': self._get_file_size(file_obj)
                    })
                    
                    logging.info(f"Uploaded image: {file_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to upload file {i+1}: {str(e)}"
                    errors.append(error_msg)
                    logging.error(error_msg)
            
            return {
                'success': True,
                'identifier': identifier,
                'folder_path': folder_path,
                'uploaded_files': uploaded_files,
                'uploaded_count': len(uploaded_files),
                'errors': errors,
                'total_files': len(image_files)
            }
            
        except Exception as e:
            logging.error(f"Failed to upload images for {identifier}: {e}")
            return {
                'success': False,
                'identifier': identifier,
                'error': str(e),
                'uploaded_files': [],
                'uploaded_count': 0,
                'errors': [str(e)]
            }

    def delete_images(self, identifier: str) -> Dict[str, Any]:
        """
        Delete all images for a specific identifier
        
        Args:
            identifier: Person identifier
            
        Returns:
            Dictionary with deletion results
        """
        try:
            folder_path = self.create_identifier_folder(identifier)
            
            # List all objects in the folder
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=folder_path
            )
            
            if 'Contents' not in response:
                return {
                    'success': True,
                    'identifier': identifier,
                    'deleted_count': 0,
                    'message': 'No files found to delete'
                }
            
            # Delete all objects
            objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
            
            delete_response = self.client.delete_objects(
                Bucket=self.bucket_name,
                Delete={'Objects': objects_to_delete}
            )
            
            deleted_count = len(delete_response.get('Deleted', []))
            errors = delete_response.get('Errors', [])
            
            logging.info(f"Deleted {deleted_count} files for identifier: {identifier}")
            
            return {
                'success': True,
                'identifier': identifier,
                'deleted_count': deleted_count,
                'errors': errors
            }
            
        except Exception as e:
            logging.error(f"Failed to delete images for {identifier}: {e}")
            return {
                'success': False,
                'identifier': identifier,
                'error': str(e),
                'deleted_count': 0
            }

    def get_images_urls(self, identifier: str) -> List[str]:
        """
        Get all image URLs for a specific identifier
        
        Args:
            identifier: Person identifier
            
        Returns:
            List of public URLs
        """
        try:
            folder_path = self.create_identifier_folder(identifier)
            
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=folder_path
            )
            
            if 'Contents' not in response:
                return []
            
            urls = []
            for obj in response['Contents']:
                url = f"{self.cdn_endpoint}/{obj['Key']}" if self.cdn_endpoint else f"{self.endpoint}/{self.bucket_name}/{obj['Key']}"
                urls.append(url)
            
            return urls
            
        except Exception as e:
            logging.error(f"Failed to get image URLs for {identifier}: {e}")
            return []

    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        if not filename:
            return '.jpg'  # Default extension
        return os.path.splitext(filename)[1].lower() or '.jpg'

    def _get_content_type(self, file_extension: str) -> str:
        """Get content type based on file extension"""
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return content_types.get(file_extension.lower(), 'image/jpeg')

    def _get_file_size(self, file_obj) -> Optional[int]:
        """Get file size from file object"""
        try:
            # Save current position
            current_pos = file_obj.tell()
            # Seek to end to get size
            file_obj.seek(0, 2)
            size = file_obj.tell()
            # Restore position
            file_obj.seek(current_pos)
            return size
        except:
            return None

    def check_connection(self) -> bool:
        """
        Check if connection to Spaces is working
        
        Returns:
            True if connection is successful
        """
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as e:
            logging.error(f"Spaces connection failed: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error checking Spaces connection: {e}")
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage stats
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='face_images/'
            )
            
            if 'Contents' not in response:
                return {
                    'total_files': 0,
                    'total_size': 0,
                    'folders': 0
                }
            
            total_files = len(response['Contents'])
            total_size = sum(obj['Size'] for obj in response['Contents'])
            
            # Count unique folders (identifiers)
            folders = set()
            for obj in response['Contents']:
                path_parts = obj['Key'].split('/')
                if len(path_parts) >= 2:
                    folders.add(path_parts[1])  # face_images/{identifier}/...
            
            return {
                'total_files': total_files,
                'total_size': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'unique_identifiers': len(folders),
                'folders': list(folders)
            }
            
        except Exception as e:
            logging.error(f"Failed to get storage stats: {e}")
            return {
                'error': str(e),
                'total_files': 0,
                'total_size': 0,
                'folders': 0
            }

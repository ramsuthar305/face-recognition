#!/usr/bin/env python3
"""
WSGI entry point for Face Recognition API
Production deployment with Gunicorn, uWSGI, or other WSGI servers
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
load_dotenv('.env')

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

def create_app():
    """
    Application factory for creating Flask app instance
    """
    try:
        # Import after setting up path and environment
        from api import app, initialize_models
        
        # Initialize models and database (always uses Milvus)
        if not initialize_models():
            logging.error("Failed to initialize models for production")
            raise RuntimeError("Model initialization failed")
        
        logging.info("Face Recognition API initialized for production")
        logging.info("Database backend: Milvus")
        
        return app
        
    except Exception as e:
        logging.error(f"Failed to create application: {e}")
        raise

# Create application instance
application = create_app()

# For compatibility with different WSGI servers
app = application

if __name__ == "__main__":
    # This runs only when executed directly (not via WSGI server)
    port = int(os.getenv('API_PORT', 8080))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    print("Starting Face Recognition API in development mode...")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    
    # Force debug=False to prevent double initialization
    application.run(host='0.0.0.0', port=port, debug=False)

"""
Gunicorn configuration for Face Recognition API
Production-ready settings for high-performance deployment with optimized connection pooling
"""

import os
import multiprocessing

# Load environment variables
from dotenv import load_dotenv
load_dotenv('.env')

# Server socket - optimized for high concurrent connections
bind = f"0.0.0.0:{os.getenv('API_PORT', 8080)}"
backlog = int(os.getenv('GUNICORN_BACKLOG', 2048))

# Worker processes - optimized for CPU and I/O bound tasks
cpu_count = multiprocessing.cpu_count()
workers = int(os.getenv('GUNICORN_WORKERS', cpu_count * 2 + 1))

# Use async workers for better concurrency with I/O operations
worker_class = os.getenv('GUNICORN_WORKER_CLASS', 'gevent')

# Connection pooling settings
worker_connections = int(os.getenv('GUNICORN_WORKER_CONNECTIONS', 2000))
max_requests = int(os.getenv('GUNICORN_MAX_REQUESTS', 2000))
max_requests_jitter = int(os.getenv('GUNICORN_MAX_REQUESTS_JITTER', 100))

# Timeout settings optimized for face processing
timeout = int(os.getenv('GUNICORN_TIMEOUT', 180))  # 3 minutes for face processing
keepalive = int(os.getenv('GUNICORN_KEEPALIVE', 5))
graceful_timeout = int(os.getenv('GUNICORN_GRACEFUL_TIMEOUT', 60))

# Memory management
preload_app = True
worker_tmp_dir = "/dev/shm"  # Use memory for temporary files (if available)

# Connection pool settings
max_keepalive_requests = int(os.getenv('GUNICORN_MAX_KEEPALIVE_REQUESTS', 100))

# Logging
accesslog = os.getenv('GUNICORN_ACCESS_LOG', 'access.log')
errorlog = os.getenv('GUNICORN_ERROR_LOG', 'error.log')
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s %(p)s'

# Process naming
proc_name = "face-recognition-api"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Connection limits
limit_conn = int(os.getenv('GUNICORN_LIMIT_CONN', 10000))

# Thread pool settings (for sync workers)
threads = int(os.getenv('GUNICORN_THREADS', 4)) if worker_class == 'sync' else None

# SSL (if needed)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

def when_ready(server):
    """Called just after the server is started"""
    server.log.info("Face Recognition API server is ready to accept connections")

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT"""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked"""
    server.log.info(f"Worker {worker.pid} is being forked")

def post_fork(server, worker):
    """Called just after a worker has been forked"""
    server.log.info(f"Worker {worker.pid} has been forked")

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal"""
    worker.log.info(f"Worker {worker.pid} received SIGABRT signal")

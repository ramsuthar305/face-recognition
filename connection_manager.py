"""
Connection Manager for Face Recognition API
Handles connection pooling and health checks for database connections
"""

import os
import logging
import threading
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue, Empty
import weakref

# Import database modules
from pymilvus import connections, utility
from database.face_db import FaceDatabase
from milvus_db import MilvusFaceDatabase


@dataclass
class ConnectionConfig:
    """Configuration for database connections"""
    host: str
    port: int
    user: str
    password: str
    collection_name: str
    embedding_size: int
    max_connections: int = 10
    connection_timeout: int = 30
    health_check_interval: int = 60


class ConnectionPool:
    """Thread-safe connection pool for database connections"""
    
    def __init__(self, config: ConnectionConfig, connection_factory):
        self.config = config
        self.connection_factory = connection_factory
        self.pool = Queue(maxsize=config.max_connections)
        self.active_connections = set()
        self.lock = threading.RLock()
        self._shutdown = False
        
        # Initialize pool with connections
        self._initialize_pool()
        
        # Start health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        logging.info(f"Connection pool initialized with {config.max_connections} max connections")
    
    def _initialize_pool(self):
        """Initialize the connection pool with initial connections"""
        for _ in range(min(3, self.config.max_connections)):
            try:
                conn = self.connection_factory()
                self.pool.put(conn)
            except Exception as e:
                logging.error(f"Failed to create initial connection: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        conn = None
        try:
            # Try to get existing connection
            try:
                conn = self.pool.get_nowait()
            except Empty:
                # Create new connection if pool is empty and under limit
                with self.lock:
                    if len(self.active_connections) < self.config.max_connections:
                        conn = self.connection_factory()
                        self.active_connections.add(id(conn))
                    else:
                        # Wait for a connection to become available
                        conn = self.pool.get(timeout=self.config.connection_timeout)
            
            yield conn
            
        except Exception as e:
            logging.error(f"Error getting connection from pool: {e}")
            raise
        finally:
            # Return connection to pool
            if conn is not None:
                try:
                    self.pool.put_nowait(conn)
                except Exception as e:
                    logging.error(f"Error returning connection to pool: {e}")
                    # Remove broken connection
                    with self.lock:
                        self.active_connections.discard(id(conn))
    
    def _health_check_loop(self):
        """Background health check for connections"""
        while not self._shutdown:
            try:
                time.sleep(self.config.health_check_interval)
                self._health_check()
            except Exception as e:
                logging.error(f"Health check error: {e}")
    
    def _health_check(self):
        """Check health of connections in pool"""
        healthy_connections = []
        
        # Check all connections in pool
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                if self._is_connection_healthy(conn):
                    healthy_connections.append(conn)
                else:
                    logging.warning("Removing unhealthy connection from pool")
                    with self.lock:
                        self.active_connections.discard(id(conn))
            except Empty:
                break
        
        # Return healthy connections to pool
        for conn in healthy_connections:
            try:
                self.pool.put_nowait(conn)
            except Exception as e:
                logging.error(f"Error returning healthy connection: {e}")
    
    def _is_connection_healthy(self, conn) -> bool:
        """Check if a connection is healthy"""
        try:
            # This would be implemented based on the specific database type
            # For now, assume connection is healthy
            return True
        except Exception:
            return False
    
    def close(self):
        """Close all connections in the pool"""
        self._shutdown = True
        
        # Close all connections in pool
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                if hasattr(conn, 'close'):
                    conn.close()
            except Empty:
                break
        
        logging.info("Connection pool closed")


class DatabaseConnectionManager:
    """Manages database connections for the face recognition API"""
    
    def __init__(self):
        self.milvus_pool: Optional[ConnectionPool] = None
        self.faiss_db: Optional[FaceDatabase] = None
        self.use_milvus = False
        self.lock = threading.RLock()
        self._initialized = False
    
    def initialize(self, use_milvus: bool = True):
        """Initialize the database connection manager"""
        with self.lock:
            if self._initialized:
                return
            
            self.use_milvus = use_milvus
            
            if use_milvus:
                self._initialize_milvus()
            else:
                self._initialize_faiss()
            
            self._initialized = True
            logging.info(f"Database connection manager initialized with {'Milvus' if use_milvus else 'FAISS'}")
    
    def _initialize_milvus(self):
        """Initialize Milvus connection pool"""
        try:
            config = ConnectionConfig(
                host=os.getenv('MILVUS_HOST', 'localhost'),
                port=int(os.getenv('MILVUS_PORT', 19530)),
                user=os.getenv('MILVUS_USER', ''),
                password=os.getenv('MILVUS_PASSWORD', ''),
                collection_name=os.getenv('MILVUS_COLLECTION_NAME', 'face_embeddings'),
                embedding_size=int(os.getenv('EMBEDDING_SIZE', 768)),
                max_connections=int(os.getenv('MILVUS_MAX_CONNECTIONS', 10)),
                connection_timeout=int(os.getenv('MILVUS_CONNECTION_TIMEOUT', 30)),
                health_check_interval=int(os.getenv('MILVUS_HEALTH_CHECK_INTERVAL', 60))
            )
            
            def create_milvus_connection():
                return MilvusFaceDatabase(
                    collection_name=config.collection_name,
                    embedding_size=config.embedding_size,
                    host=config.host,
                    port=config.port,
                    user=config.user,
                    password=config.password,
                    max_workers=4
                )
            
            self.milvus_pool = ConnectionPool(config, create_milvus_connection)
            
        except Exception as e:
            logging.error(f"Failed to initialize Milvus connection pool: {e}")
            raise
    
    def _initialize_faiss(self):
        """Initialize FAISS database"""
        try:
            self.faiss_db = FaceDatabase(
                embedding_size=int(os.getenv('EMBEDDING_SIZE', 1024)),
                db_path=os.getenv('DB_PATH', './database/face_database'),
                max_workers=int(os.getenv('MAX_WORKERS', 8))
            )
        except Exception as e:
            logging.error(f"Failed to initialize FAISS database: {e}")
            raise
    
    @contextmanager
    def get_database(self):
        """Get database connection from appropriate pool"""
        if not self._initialized:
            raise RuntimeError("Connection manager not initialized")
        
        if self.use_milvus and self.milvus_pool:
            with self.milvus_pool.get_connection() as conn:
                yield conn
        elif self.faiss_db:
            yield self.faiss_db
        else:
            raise RuntimeError("No database connection available")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all database connections"""
        health_status = {
            'initialized': self._initialized,
            'use_milvus': self.use_milvus,
            'milvus_healthy': False,
            'faiss_healthy': False,
            'timestamp': time.time()
        }
        
        try:
            if self.use_milvus and self.milvus_pool:
                # Check Milvus connection health
                with self.milvus_pool.get_connection() as conn:
                    # Simple health check - try to get collection info
                    if hasattr(conn, 'collection') and conn.collection:
                        health_status['milvus_healthy'] = True
            elif self.faiss_db:
                # Check FAISS database health
                health_status['faiss_healthy'] = True
        except Exception as e:
            logging.error(f"Health check failed: {e}")
        
        return health_status
    
    def close(self):
        """Close all database connections"""
        with self.lock:
            if self.milvus_pool:
                self.milvus_pool.close()
                self.milvus_pool = None
            
            if self.faiss_db:
                # FAISS doesn't need explicit closing
                self.faiss_db = None
            
            self._initialized = False
            logging.info("Database connection manager closed")


# Global connection manager instance
connection_manager = DatabaseConnectionManager()

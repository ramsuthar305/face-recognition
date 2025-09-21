import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusException
)
import json
from datetime import datetime

class MilvusFaceDatabase:
    """
    Milvus-based face database for storing and searching face embeddings
    """
    
    def __init__(self, 
                 collection_name: str = "face_embeddings",
                 embedding_size: int = 768,
                 host: str = "localhost",
                 port: int = 19530,
                 user: str = "",
                 password: str = "",
                 max_workers: int = 4):
        """
        Initialize Milvus face database
        
        Args:
            collection_name: Name of the Milvus collection
            embedding_size: Dimension of face embeddings
            host: Milvus server host
            port: Milvus server port
            user: Username for authentication
            password: Password for authentication
            max_workers: Number of worker threads (for compatibility)
        """
        self.collection_name = collection_name
        self.embedding_size = embedding_size
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.max_workers = max_workers
        self.collection = None
        self._connected = False
        
        # Connect to Milvus
        self._connect()
        
        # Create collection if it doesn't exist
        self._create_collection()
        
        logging.info(f"Milvus face database initialized: {collection_name}")

    def _connect(self):
        """Connect to Milvus server with authentication"""
        try:
            # Prepare connection parameters
            conn_params = {
                "host": self.host,
                "port": self.port,
                "timeout": 30,  # 30 second timeout
                "secure": True if "zillizcloud.com" in self.host else False  # Use TLS for cloud
            }
            
            # Add authentication if provided
            if self.user and self.password:
                conn_params["user"] = self.user
                conn_params["password"] = self.password
                logging.info(f"Connecting to Milvus with authentication: {self.user}@{self.host}:{self.port}")
            else:
                logging.info(f"Connecting to Milvus without authentication: {self.host}:{self.port}")
            
            # Add token-based auth for Zilliz Cloud if needed
            if "zillizcloud.com" in self.host and self.user and self.password:
                # For Zilliz Cloud, try token format: user:password
                conn_params["token"] = f"{self.user}:{self.password}"
                # Remove user/password as token takes precedence
                del conn_params["user"]
                del conn_params["password"]
                logging.info(f"Using token authentication for Zilliz Cloud")
            
            connections.connect("default", **conn_params)
            self._connected = True
            logging.info(f"Successfully connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logging.error(f"Failed to connect to Milvus: {e}")
            logging.error(f"Connection params: host={self.host}, port={self.port}, user={self.user}")
            raise

    def _create_collection(self):
        """Create the face embeddings collection"""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logging.info(f"Loaded existing collection: {self.collection_name}")
                return
            
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="person_name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_size),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1000)
            ]
            
            schema = CollectionSchema(fields, "Face embeddings collection")
            
            # Create collection
            self.collection = Collection(self.collection_name, schema)
            
            # Create index for vector search
            index_params = {
                "metric_type": "IP",  # Inner Product for cosine similarity
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            
            self.collection.create_index("embedding", index_params)
            
            logging.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            logging.error(f"Failed to create collection: {e}")
            raise

    def add_face(self, embedding: np.ndarray, name: str, image_path: str = "", metadata: Dict = None):
        """
        Add a face embedding to the database
        
        Args:
            embedding: Face embedding vector
            name: Person name
            image_path: Path to the source image
            metadata: Additional metadata
        """
        try:
            # Normalize embedding for cosine similarity
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            # Prepare data
            entities = [
                [name],  # person_name
                [normalized_embedding.tolist()],  # embedding
                [image_path],  # image_path
                [datetime.now().isoformat()],  # created_at
                [json.dumps(metadata or {})]  # metadata
            ]
            
            # Insert data
            insert_result = self.collection.insert(entities)
            
            logging.info(f"Added face for {name}, ID: {insert_result.primary_keys[0]}")
            
        except Exception as e:
            logging.error(f"Failed to add face for {name}: {e}")
            raise

    def add_faces_batch(self, embeddings: List[np.ndarray], names: List[str], 
                       image_paths: List[str] = None, metadata_list: List[Dict] = None):
        """
        Add multiple faces to the database in batch
        
        Args:
            embeddings: List of face embeddings
            names: List of person names
            image_paths: List of image paths
            metadata_list: List of metadata dictionaries
        """
        try:
            if not embeddings or not names:
                return
            
            # Normalize embeddings
            normalized_embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]
            
            # Prepare default values
            if image_paths is None:
                image_paths = [""] * len(embeddings)
            if metadata_list is None:
                metadata_list = [{}] * len(embeddings)
            
            # Prepare data
            entities = [
                names,  # person_name
                [emb.tolist() for emb in normalized_embeddings],  # embedding
                image_paths,  # image_path
                [datetime.now().isoformat()] * len(embeddings),  # created_at
                [json.dumps(meta) for meta in metadata_list]  # metadata
            ]
            
            # Insert data
            insert_result = self.collection.insert(entities)
            
            logging.info(f"Added {len(embeddings)} faces in batch")
            
        except Exception as e:
            logging.error(f"Failed to add faces in batch: {e}")
            raise

    def search(self, embedding: np.ndarray, threshold: float = 0.4, top_k: int = 1) -> Tuple[str, float]:
        """
        Search for the closest face in the database
        
        Args:
            embedding: Query face embedding
            threshold: Similarity threshold
            top_k: Number of top results to return
            
        Returns:
            Tuple of (person_name, similarity_score)
        """
        try:
            # Load collection to memory
            self.collection.load()
            
            # Normalize query embedding
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            # Search parameters
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = self.collection.search(
                data=[normalized_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["person_name"]
            )
            
            if results and len(results[0]) > 0:
                hit = results[0][0]
                similarity = float(hit.score)
                person_name = hit.entity.get("person_name")
                
                if similarity > threshold:
                    return person_name, similarity
                else:
                    return "Unknown", similarity
            
            return "Unknown", 0.0
            
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return "Unknown", 0.0

    def search_with_person_filter(self, embedding: np.ndarray, 
                                 exclude_person_id: str = None,
                                 threshold: float = 0.4,
                                 top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search with person filtering for better accuracy
        
        Args:
            embedding: Query embedding
            exclude_person_id: Person ID to exclude from search
            threshold: Similarity threshold
            top_k: Number of results to return
            
        Returns:
            List of (person_id, person_name, similarity) tuples
        """
        try:
            self.collection.load()
            
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            # Build filter expression
            filter_expr = None
            if exclude_person_id:
                filter_expr = f'person_name != "{exclude_person_id}"'
            
            # Search parameters optimized for multiple results
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 32}
            }
            
            # Perform search
            results = self.collection.search(
                data=[normalized_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["person_name"]
            )
            
            search_results = []
            if results and len(results[0]) > 0:
                for hit in results[0]:
                    similarity = float(hit.score)
                    if similarity > threshold:
                        person_name = hit.entity.get("person_name")
                        search_results.append((person_name, person_name, similarity))  # For compatibility
            
            return search_results
            
        except Exception as e:
            logging.error(f"Search with filter failed: {e}")
            return []

    def batch_search(self, embeddings: List[np.ndarray], threshold: float = 0.4) -> List[Tuple[str, float]]:
        """
        Perform batch search for multiple embeddings
        
        Args:
            embeddings: List of face embeddings
            threshold: Similarity threshold
            
        Returns:
            List of (person_name, similarity_score) tuples
        """
        try:
            if not embeddings:
                return []
            
            # Load collection to memory
            self.collection.load()
            
            # Normalize embeddings
            normalized_embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]
            
            # Search parameters
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            # Perform batch search
            results = self.collection.search(
                data=[emb.tolist() for emb in normalized_embeddings],
                anns_field="embedding",
                param=search_params,
                limit=1,
                output_fields=["person_name"]
            )
            
            # Process results
            search_results = []
            for result in results:
                if result and len(result) > 0:
                    hit = result[0]
                    similarity = float(hit.score)
                    person_name = hit.entity.get("person_name")
                    
                    if similarity > threshold:
                        search_results.append((person_name, similarity))
                    else:
                        search_results.append(("Unknown", similarity))
                else:
                    search_results.append(("Unknown", 0.0))
            
            return search_results
            
        except Exception as e:
            logging.error(f"Batch search failed: {e}")
            return [("Unknown", 0.0)] * len(embeddings)

    def get_person_count(self) -> Dict[str, int]:
        """Get count of faces per person"""
        try:
            self.collection.load()
            
            # Query all person names
            results = self.collection.query(
                expr="id >= 0",
                output_fields=["person_name"]
            )
            
            # Count occurrences
            person_counts = {}
            for result in results:
                name = result["person_name"]
                person_counts[name] = person_counts.get(name, 0) + 1
            
            return person_counts
            
        except Exception as e:
            logging.error(f"Failed to get person count: {e}")
            return {}

    def delete_person(self, person_name: str) -> int:
        """
        Delete all faces for a specific person
        
        Args:
            person_name: Name of person to delete
            
        Returns:
            Number of deleted records
        """
        try:
            # Delete by person name
            expr = f'person_name == "{person_name}"'
            delete_result = self.collection.delete(expr)
            
            logging.info(f"Deleted {delete_result.delete_count} faces for {person_name}")
            return delete_result.delete_count
            
        except Exception as e:
            logging.error(f"Failed to delete person {person_name}: {e}")
            return 0

    def save(self):
        """Flush data to disk (Milvus handles persistence automatically)"""
        try:
            self.collection.flush()
            logging.info("Flushed data to Milvus")
        except Exception as e:
            logging.error(f"Failed to flush data: {e}")

    def load(self) -> bool:
        """Load collection (for compatibility with existing interface)"""
        try:
            self.collection.load()
            return True
        except Exception as e:
            logging.error(f"Failed to load collection: {e}")
            return False

    def flush_all(self):
        """Flush all data from the collection"""
        try:
            from pymilvus import utility, Collection
            
            # Drop existing collection
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                collection.drop()
                logging.info(f"Dropped collection: {self.collection_name}")
            
            # Recreate the collection
            self._create_collection()
            logging.info(f"Recreated empty collection: {self.collection_name}")
            
        except Exception as e:
            logging.error(f"Failed to flush collection: {e}")
            raise

    def close(self):
        """Close connection to Milvus"""
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")
            self._connected = False
            logging.info("Disconnected from Milvus")
        except Exception as e:
            logging.error(f"Error closing Milvus connection: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass

    @property
    def ntotal(self) -> int:
        """Get total number of vectors in collection"""
        try:
            self.collection.load()
            return self.collection.num_entities
        except:
            return 0

    # For compatibility with existing FAISS interface
    @property
    def index(self):
        """Compatibility property to mimic FAISS interface"""
        class IndexCompat:
            def __init__(self, collection):
                self.collection = collection
            
            @property
            def ntotal(self):
                try:
                    self.collection.load()
                    return self.collection.num_entities
                except:
                    return 0
        
        return IndexCompat(self.collection)

    @property
    def metadata(self) -> List[str]:
        """Get all person names (for compatibility)"""
        try:
            self.collection.load()
            results = self.collection.query(
                expr="id >= 0",
                output_fields=["person_name"]
            )
            return [result["person_name"] for result in results]
        except:
            return []

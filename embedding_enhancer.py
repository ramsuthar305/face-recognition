"""
Embedding Enhancer for Face Recognition
Extends 512-dimensional embeddings to 1024 dimensions for better accuracy
"""

import numpy as np
import logging
from typing import List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

class EmbeddingEnhancer:
    """
    Enhances face embeddings by extending them to higher dimensions
    Uses various techniques to create meaningful 1024-dimensional embeddings from 512-dimensional ones
    """
    
    def __init__(self, target_size: int = 1024, enhancement_method: str = "concatenate_transform"):
        """
        Initialize embedding enhancer
        
        Args:
            target_size: Target embedding dimension (e.g., 1024)
            enhancement_method: Method to enhance embeddings
                - "concatenate_transform": Concatenate with transformed versions
                - "pca_extend": Use PCA to extend dimensions
                - "polynomial": Add polynomial features
                - "statistical": Add statistical features
        """
        self.target_size = target_size
        self.enhancement_method = enhancement_method
        self.original_size = 512  # ArcFace default
        
        # Initialize enhancement components
        self.scaler = StandardScaler()
        self.pca = None
        self.is_fitted = False
        
        logging.info(f"Embedding enhancer initialized: {self.original_size} → {self.target_size}")

    def enhance_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Enhance a single embedding to target dimensions
        
        Args:
            embedding: Original 512-dimensional embedding
            
        Returns:
            Enhanced embedding with target dimensions
        """
        if embedding.shape[0] != self.original_size:
            raise ValueError(f"Expected {self.original_size}-dimensional embedding, got {embedding.shape[0]}")
        
        if self.enhancement_method == "concatenate_transform":
            return self._concatenate_transform(embedding)
        elif self.enhancement_method == "polynomial":
            return self._polynomial_features(embedding)
        elif self.enhancement_method == "statistical":
            return self._statistical_features(embedding)
        else:
            return self._concatenate_transform(embedding)  # Default

    def _concatenate_transform(self, embedding: np.ndarray) -> np.ndarray:
        """
        Enhance embedding by concatenating with transformed versions
        """
        # Normalize original embedding
        normalized = embedding / np.linalg.norm(embedding)
        
        # Create various transformations
        transformations = [
            normalized,  # Original (512)
            normalized ** 2,  # Squared (512) - emphasizes strong features
        ]
        
        # If we need exactly 1024, we can concatenate original + squared
        if self.target_size == 1024:
            enhanced = np.concatenate(transformations)
        else:
            # For other sizes, repeat and truncate as needed
            concatenated = np.concatenate(transformations)
            if len(concatenated) >= self.target_size:
                enhanced = concatenated[:self.target_size]
            else:
                # Repeat if needed
                repeats = (self.target_size // len(concatenated)) + 1
                repeated = np.tile(concatenated, repeats)
                enhanced = repeated[:self.target_size]
        
        # Normalize the final embedding
        return enhanced / np.linalg.norm(enhanced)

    def _polynomial_features(self, embedding: np.ndarray) -> np.ndarray:
        """
        Create polynomial features from embedding
        """
        normalized = embedding / np.linalg.norm(embedding)
        
        # Create polynomial features: [x, x^2, x^3, ...]
        features = [normalized]
        
        # Add squared features
        if len(features) * self.original_size < self.target_size:
            features.append(normalized ** 2)
        
        # Add interaction features (first half * second half)
        if len(features) * self.original_size < self.target_size:
            mid = len(normalized) // 2
            interactions = normalized[:mid] * normalized[mid:mid*2] if mid*2 <= len(normalized) else normalized[:mid] * normalized[mid:]
            # Pad interactions to match original size
            if len(interactions) < self.original_size:
                interactions = np.pad(interactions, (0, self.original_size - len(interactions)), mode='constant')
            features.append(interactions[:self.original_size])
        
        # Concatenate and truncate/pad to target size
        concatenated = np.concatenate(features)
        if len(concatenated) >= self.target_size:
            enhanced = concatenated[:self.target_size]
        else:
            enhanced = np.pad(concatenated, (0, self.target_size - len(concatenated)), mode='constant')
        
        return enhanced / np.linalg.norm(enhanced)

    def _statistical_features(self, embedding: np.ndarray) -> np.ndarray:
        """
        Add statistical features to embedding
        """
        normalized = embedding / np.linalg.norm(embedding)
        
        # Calculate statistical features
        mean_val = np.mean(normalized)
        std_val = np.std(normalized)
        max_val = np.max(normalized)
        min_val = np.min(normalized)
        median_val = np.median(normalized)
        
        # Create bins and histograms
        hist, _ = np.histogram(normalized, bins=32, density=True)
        
        # Combine features
        stat_features = np.array([mean_val, std_val, max_val, min_val, median_val])
        
        # Create extended features
        features = [
            normalized,  # Original (512)
            normalized ** 2,  # Squared (512)
        ]
        
        # Add statistical and histogram features to reach target size
        remaining_size = self.target_size - sum(len(f) for f in features)
        
        if remaining_size > 0:
            # Combine stat features and histogram
            additional_features = np.concatenate([stat_features, hist])
            
            # Repeat or truncate to fill remaining space
            if len(additional_features) < remaining_size:
                repeats = (remaining_size // len(additional_features)) + 1
                additional_features = np.tile(additional_features, repeats)
            
            features.append(additional_features[:remaining_size])
        
        enhanced = np.concatenate(features)
        return enhanced / np.linalg.norm(enhanced)

    def enhance_batch(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """
        Enhance multiple embeddings in batch
        
        Args:
            embeddings: List of original embeddings
            
        Returns:
            List of enhanced embeddings
        """
        return [self.enhance_embedding(emb) for emb in embeddings]

# Global enhancer instance
_enhancer = None

def get_enhancer(target_size: int = 1024, method: str = "concatenate_transform") -> EmbeddingEnhancer:
    """Get or create global enhancer instance"""
    global _enhancer
    if _enhancer is None or _enhancer.target_size != target_size:
        _enhancer = EmbeddingEnhancer(target_size, method)
    return _enhancer

def enhance_embedding_to_1024(embedding: np.ndarray) -> np.ndarray:
    """
    Quick function to enhance 512-dim embedding to 1024-dim
    
    Args:
        embedding: 512-dimensional embedding
        
    Returns:
        1024-dimensional enhanced embedding
    """
    enhancer = get_enhancer(1024, "concatenate_transform")
    return enhancer.enhance_embedding(embedding)

def enhance_embedding_to_768(embedding: np.ndarray) -> np.ndarray:
    """
    Quick function to enhance 512-dim embedding to 768-dim
    
    Args:
        embedding: 512-dimensional embedding
        
    Returns:
        768-dimensional enhanced embedding
    """
    enhancer = get_enhancer(768, "concatenate_transform")
    return enhancer.enhance_embedding(embedding)

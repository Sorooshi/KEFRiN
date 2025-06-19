"""
KEFRiN: Community Partitioning over Feature-Rich Networks Using an Extended K-Means Method
Optimized Production Implementation

This module implements the three versions of KEFRiN algorithm:
- KEFRiNe: Euclidean distance
- KEFRiNc: Cosine distance  
- KEFRiNm: Manhattan distance

Authors: Soroosh Shalileh and Boris Mirkin
Optimized for production use with large datasets
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class DistanceMetric(Enum):
    """Enumeration of supported distance metrics"""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"


class PreprocessingMethod(Enum):
    """Enumeration of preprocessing methods"""
    NONE = "none"
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"
    RANGE = "range"


@dataclass
class KEFRiNConfig:
    """Configuration class for KEFRiN algorithm"""
    n_clusters: int = 5
    rho: float = 1.0  # Feature coefficient
    xi: float = 1.0   # Network coefficient
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    max_iterations: int = 1000
    tolerance: float = 1e-6
    kmeans_plus_plus: bool = True
    random_state: Optional[int] = None
    n_init: int = 10  # Number of random initializations
    preprocessing_y: PreprocessingMethod = PreprocessingMethod.Z_SCORE
    preprocessing_p: PreprocessingMethod = PreprocessingMethod.NONE


class OptimizedDistance:
    """Optimized distance computation methods"""
    
    @staticmethod
    def euclidean_batch(data_points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between all data points and centroids efficiently
        
        Args:
            data_points: (n_samples, n_features)
            centroids: (n_clusters, n_features)
            
        Returns:
            distances: (n_samples, n_clusters)
        """
        # Use broadcasting for efficient computation
        distances = np.sqrt(np.sum((data_points[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))
        return distances
    
    @staticmethod
    def cosine_batch(data_points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute Cosine distances between all data points and centroids efficiently
        
        Args:
            data_points: (n_samples, n_features)
            centroids: (n_clusters, n_features)
            
        Returns:
            distances: (n_samples, n_clusters)
        """
        # Normalize data points and centroids
        eps = 1e-10
        data_norm = np.linalg.norm(data_points, axis=1, keepdims=True) + eps
        centroids_norm = np.linalg.norm(centroids, axis=1, keepdims=True) + eps
        
        data_normalized = data_points / data_norm
        centroids_normalized = centroids / centroids_norm
        
        # Compute cosine similarity using matrix multiplication
        cosine_sim = np.dot(data_normalized, centroids_normalized.T)
        cosine_dist = 1 - cosine_sim
        
        return cosine_dist
    
    @staticmethod
    def manhattan_batch(data_points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute Manhattan distances between all data points and centroids efficiently
        
        Args:
            data_points: (n_samples, n_features)
            centroids: (n_clusters, n_features)
            
        Returns:
            distances: (n_samples, n_clusters)
        """
        distances = np.sum(np.abs(data_points[:, np.newaxis, :] - centroids[np.newaxis, :, :]), axis=2)
        return distances


class DataPreprocessor:
    """Data preprocessing utilities for KEFRiN"""
    
    @staticmethod
    def preprocess_features(y: np.ndarray, method: PreprocessingMethod) -> Tuple[np.ndarray, dict]:
        """
        Preprocess feature data using specified method
        
        Args:
            y: Feature matrix (n_samples, n_features)
            method: Preprocessing method to apply
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_metadata)
        """
        if method == PreprocessingMethod.NONE:
            return y.copy(), {'method': 'none'}
        
        elif method == PreprocessingMethod.Z_SCORE:
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            y_std[y_std == 0] = 1  # Avoid division by zero
            y_processed = (y - y_mean) / y_std
            
            metadata = {
                'method': 'z_score',
                'mean': y_mean,
                'std': y_std
            }
            return y_processed, metadata
        
        elif method == PreprocessingMethod.MIN_MAX:
            y_min = np.min(y, axis=0)
            y_max = np.max(y, axis=0)
            y_range = y_max - y_min
            y_range[y_range == 0] = 1  # Avoid division by zero
            y_processed = (y - y_min) / y_range
            
            metadata = {
                'method': 'min_max',
                'min': y_min,
                'max': y_max,
                'range': y_range
            }
            return y_processed, metadata
        
        elif method == PreprocessingMethod.RANGE:
            # Range normalization: scale to [-1, 1]
            y_min = np.min(y, axis=0)
            y_max = np.max(y, axis=0)
            y_range = y_max - y_min
            y_range[y_range == 0] = 1  # Avoid division by zero
            y_processed = 2 * (y - y_min) / y_range - 1
            
            metadata = {
                'method': 'range',
                'min': y_min,
                'max': y_max,
                'range': y_range
            }
            return y_processed, metadata
        
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    @staticmethod
    def preprocess_network(p: np.ndarray, method: PreprocessingMethod) -> Tuple[np.ndarray, dict]:
        """
        Preprocess network data using specified method
        
        Args:
            p: Network matrix (n_samples, n_samples)
            method: Preprocessing method to apply
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_metadata)
        """
        if method == PreprocessingMethod.NONE:
            return p.copy(), {'method': 'none'}
        
        elif method == PreprocessingMethod.Z_SCORE:
            # For network data, apply z-score to each row
            p_processed = np.zeros_like(p)
            metadata = {'method': 'z_score', 'row_means': [], 'row_stds': []}
            
            for i in range(p.shape[0]):
                row_mean = np.mean(p[i])
                row_std = np.std(p[i])
                if row_std == 0:
                    row_std = 1
                p_processed[i] = (p[i] - row_mean) / row_std
                metadata['row_means'].append(row_mean)
                metadata['row_stds'].append(row_std)
            
            metadata['row_means'] = np.array(metadata['row_means'])
            metadata['row_stds'] = np.array(metadata['row_stds'])
            return p_processed, metadata
        
        elif method == PreprocessingMethod.MIN_MAX:
            p_min = np.min(p)
            p_max = np.max(p)
            p_range = p_max - p_min
            if p_range == 0:
                p_range = 1
            p_processed = (p - p_min) / p_range
            
            metadata = {
                'method': 'min_max',
                'min': p_min,
                'max': p_max,
                'range': p_range
            }
            return p_processed, metadata
        
        elif method == PreprocessingMethod.RANGE:
            p_min = np.min(p)
            p_max = np.max(p)
            p_range = p_max - p_min
            if p_range == 0:
                p_range = 1
            p_processed = 2 * (p - p_min) / p_range - 1
            
            metadata = {
                'method': 'range',
                'min': p_min,
                'max': p_max,
                'range': p_range
            }
            return p_processed, metadata
        
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")


class KEFRiN:
    """
    Optimized KEFRiN algorithm implementation
    
    Supports three distance metrics:
    - Euclidean (KEFRiNe)
    - Cosine (KEFRiNc) 
    - Manhattan (KEFRiNm)
    """
    
    def __init__(self, config: KEFRiNConfig):
        self.config = config
        self.distance_computer = OptimizedDistance()
        self.preprocessor = DataPreprocessor()
        
        # Set random state for reproducibility
        if config.random_state is not None:
            np.random.seed(config.random_state)
        
        # Initialize attributes
        self.y_processed_ = None
        self.p_processed_ = None
        self.y_metadata_ = None
        self.p_metadata_ = None
        self.centroids_y_ = None
        self.centroids_p_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
    
    def _compute_distances(self, y: np.ndarray, p: np.ndarray, 
                          centroids_y: np.ndarray, centroids_p: np.ndarray) -> np.ndarray:
        """
        Compute combined distances using the specified metric
        """
        # Compute feature distances
        if self.config.distance_metric == DistanceMetric.EUCLIDEAN:
            dist_y = self.distance_computer.euclidean_batch(y, centroids_y)
            dist_p = self.distance_computer.euclidean_batch(p, centroids_p)
        elif self.config.distance_metric == DistanceMetric.COSINE:
            dist_y = self.distance_computer.cosine_batch(y, centroids_y)
            dist_p = self.distance_computer.cosine_batch(p, centroids_p)
        elif self.config.distance_metric == DistanceMetric.MANHATTAN:
            dist_y = self.distance_computer.manhattan_batch(y, centroids_y)
            dist_p = self.distance_computer.manhattan_batch(p, centroids_p)
        else:
            raise ValueError(f"Unsupported distance metric: {self.config.distance_metric}")
        
        # Combine distances with weights
        combined_distances = self.config.rho * dist_y + self.config.xi * dist_p
        return combined_distances
    
    def _kmeans_plus_plus_init(self, y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """K-means++ initialization for better centroid selection"""
        n_samples = y.shape[0]
        centroids_y = np.zeros((self.config.n_clusters, y.shape[1]))
        centroids_p = np.zeros((self.config.n_clusters, p.shape[1]))
        
        # Choose first centroid randomly
        first_idx = np.random.randint(0, n_samples)
        centroids_y[0] = y[first_idx]
        centroids_p[0] = p[first_idx]
        
        # Choose remaining centroids
        for i in range(1, self.config.n_clusters):
            # Compute distances to existing centroids
            distances = self._compute_distances(y, p, centroids_y[:i], centroids_p[:i])
            min_distances = np.min(distances, axis=1)
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = min_distances ** 2
            probabilities /= np.sum(probabilities)
            
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids_y[i] = y[next_idx]
            centroids_p[i] = p[next_idx]
        
        return centroids_y, centroids_p
    
    def _random_init(self, y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random initialization of centroids"""
        n_samples = y.shape[0]
        indices = np.random.choice(n_samples, size=self.config.n_clusters, replace=False)
        
        centroids_y = y[indices].copy()
        centroids_p = p[indices].copy()
        
        return centroids_y, centroids_p
    
    def _update_centroids(self, y: np.ndarray, p: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update centroids based on current cluster assignments"""
        centroids_y = np.zeros((self.config.n_clusters, y.shape[1]))
        centroids_p = np.zeros((self.config.n_clusters, p.shape[1]))
        
        for k in range(self.config.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                centroids_y[k] = np.mean(y[mask], axis=0)
                centroids_p[k] = np.mean(p[mask], axis=0)
            else:
                # Handle empty clusters by reinitializing randomly
                random_idx = np.random.randint(0, y.shape[0])
                centroids_y[k] = y[random_idx]
                centroids_p[k] = p[random_idx]
        
        return centroids_y, centroids_p
    
    def _single_run(self, y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        """Single run of the KEFRiN algorithm"""
        # Initialize centroids
        if self.config.kmeans_plus_plus:
            centroids_y, centroids_p = self._kmeans_plus_plus_init(y, p)
        else:
            centroids_y, centroids_p = self._random_init(y, p)
        
        # Main iteration loop
        prev_inertia = float('inf')
        for iteration in range(self.config.max_iterations):
            # Compute distances and assign labels
            distances = self._compute_distances(y, p, centroids_y, centroids_p)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids_y, new_centroids_p = self._update_centroids(y, p, labels)
            
            # Check for convergence
            inertia = np.sum([np.sum(distances[labels == k, k]) for k in range(self.config.n_clusters)])
            
            if np.abs(prev_inertia - inertia) < self.config.tolerance:
                break
            
            centroids_y, centroids_p = new_centroids_y, new_centroids_p
            prev_inertia = inertia
        
        return centroids_y, centroids_p, labels, inertia, iteration + 1
    
    def fit(self, y: np.ndarray, p: np.ndarray) -> 'KEFRiN':
        """Fit the KEFRiN model to data"""
        logger.info(f"Starting KEFRiN fitting with {self.config.distance_metric.value} distance")
        logger.info(f"Preprocessing: Y={self.config.preprocessing_y.value}, P={self.config.preprocessing_p.value}")
        start_time = time.time()
        
        # Preprocess data using configured methods
        self.y_processed_, self.y_metadata_ = self.preprocessor.preprocess_features(
            y, self.config.preprocessing_y
        )
        self.p_processed_, self.p_metadata_ = self.preprocessor.preprocess_network(
            p, self.config.preprocessing_p
        )
        
        # Run algorithm multiple times and select best result
        best_inertia = float('inf')
        best_result = None
        
        for run in range(self.config.n_init):
            centroids_y, centroids_p, labels, inertia, n_iter = self._single_run(
                self.y_processed_, self.p_processed_
            )
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_result = (centroids_y, centroids_p, labels, inertia, n_iter)
        
        # Store best result
        self.centroids_y_, self.centroids_p_, self.labels_, self.inertia_, self.n_iter_ = best_result
        
        end_time = time.time()
        logger.info(f"KEFRiN fitting completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Final inertia: {self.inertia_:.4f}, Iterations: {self.n_iter_}")
        
        return self
    
    def preprocess_data(self, y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data using configured methods without fitting the model
        
        Args:
            y: Feature matrix (n_samples, n_features)
            p: Network matrix (n_samples, n_samples)
            
        Returns:
            Tuple of (preprocessed_y, preprocessed_p)
        """
        y_processed, self.y_metadata_ = self.preprocessor.preprocess_features(
            y, self.config.preprocessing_y
        )
        p_processed, self.p_metadata_ = self.preprocessor.preprocess_network(
            p, self.config.preprocessing_p
        )
        
        self.y_processed_ = y_processed
        self.p_processed_ = p_processed
        
        return y_processed, p_processed
    
    def fit_predict(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels"""
        self.fit(y, p)
        return self.labels_
    
    def apply_kefrin(self) -> np.ndarray:
        """Legacy method name for compatibility"""
        return self.labels_


# Convenience functions for the three versions
def KEFRiNe(y: np.ndarray, p: np.ndarray, rho: float = 1.0, xi: float = 1.0, 
           n_clusters: int = 5, max_iteration: int = 1000, kmean_pp: bool = True,
           preprocessing_y: str = 'z_score', preprocessing_p: str = 'none') -> np.ndarray:
    """KEFRiN with Euclidean distance (KEFRiNe)
    
    Args:
        y: Feature matrix (n_samples, n_features)
        p: Network matrix (n_samples, n_samples)
        rho: Feature importance weight
        xi: Network importance weight
        n_clusters: Number of clusters
        max_iteration: Maximum iterations
        kmean_pp: Use K-means++ initialization
        preprocessing_y: Feature preprocessing method ('none', 'z_score', 'min_max', 'range')
        preprocessing_p: Network preprocessing method ('none', 'z_score', 'min_max', 'range')
    """
    # Convert string preprocessing methods to enums
    preproc_y = PreprocessingMethod(preprocessing_y)
    preproc_p = PreprocessingMethod(preprocessing_p)
    
    config = KEFRiNConfig(
        n_clusters=n_clusters,
        rho=rho,
        xi=xi,
        distance_metric=DistanceMetric.EUCLIDEAN,
        max_iterations=max_iteration,
        kmeans_plus_plus=kmean_pp,
        preprocessing_y=preproc_y,
        preprocessing_p=preproc_p
    )
    
    model = KEFRiN(config)
    return model.fit_predict(y, p)


def KEFRiNc(y: np.ndarray, p: np.ndarray, rho: float = 1.0, xi: float = 1.0,
           n_clusters: int = 5, max_iteration: int = 1000, kmean_pp: bool = True,
           preprocessing_y: str = 'z_score', preprocessing_p: str = 'none') -> np.ndarray:
    """KEFRiN with Cosine distance (KEFRiNc)
    
    Args:
        y: Feature matrix (n_samples, n_features)
        p: Network matrix (n_samples, n_samples)
        rho: Feature importance weight
        xi: Network importance weight
        n_clusters: Number of clusters
        max_iteration: Maximum iterations
        kmean_pp: Use K-means++ initialization
        preprocessing_y: Feature preprocessing method ('none', 'z_score', 'min_max', 'range')
        preprocessing_p: Network preprocessing method ('none', 'z_score', 'min_max', 'range')
    """
    # Convert string preprocessing methods to enums
    preproc_y = PreprocessingMethod(preprocessing_y)
    preproc_p = PreprocessingMethod(preprocessing_p)
    
    config = KEFRiNConfig(
        n_clusters=n_clusters,
        rho=rho,
        xi=xi,
        distance_metric=DistanceMetric.COSINE,
        max_iterations=max_iteration,
        kmeans_plus_plus=kmean_pp,
        preprocessing_y=preproc_y,
        preprocessing_p=preproc_p
    )
    
    model = KEFRiN(config)
    return model.fit_predict(y, p)


def KEFRiNm(y: np.ndarray, p: np.ndarray, rho: float = 1.0, xi: float = 1.0,
           n_clusters: int = 5, max_iteration: int = 1000, kmean_pp: bool = True,
           preprocessing_y: str = 'z_score', preprocessing_p: str = 'none') -> np.ndarray:
    """KEFRiN with Manhattan distance (KEFRiNm)
    
    Args:
        y: Feature matrix (n_samples, n_features)
        p: Network matrix (n_samples, n_samples)
        rho: Feature importance weight
        xi: Network importance weight
        n_clusters: Number of clusters
        max_iteration: Maximum iterations
        kmean_pp: Use K-means++ initialization
        preprocessing_y: Feature preprocessing method ('none', 'z_score', 'min_max', 'range')
        preprocessing_p: Network preprocessing method ('none', 'z_score', 'min_max', 'range')
    """
    # Convert string preprocessing methods to enums
    preproc_y = PreprocessingMethod(preprocessing_y)
    preproc_p = PreprocessingMethod(preprocessing_p)
    
    config = KEFRiNConfig(
        n_clusters=n_clusters,
        rho=rho,
        xi=xi,
        distance_metric=DistanceMetric.MANHATTAN,
        max_iterations=max_iteration,
        kmeans_plus_plus=kmean_pp,
        preprocessing_y=preproc_y,
        preprocessing_p=preproc_p
    )
    
    model = KEFRiN(config)
    return model.fit_predict(y, p)


# Legacy compatibility class
class KEFRiN_Legacy:
    """Legacy compatibility wrapper for the original interface"""
    
    def __init__(self, y: np.ndarray, p: np.ndarray, rho: float = 1.0, xi: float = 1.0,
                 n_clusters: int = 5, kmean_pp: bool = True, euclidean: int = 1,
                 cosine: int = 0, manhattan: int = 0, max_iteration: int = 1000):
        
        # Determine distance metric from legacy parameters
        if euclidean == 1:
            distance_metric = DistanceMetric.EUCLIDEAN
        elif cosine == 1:
            distance_metric = DistanceMetric.COSINE
        elif manhattan == 1:
            distance_metric = DistanceMetric.MANHATTAN
        else:
            distance_metric = DistanceMetric.EUCLIDEAN
        
        config = KEFRiNConfig(
            n_clusters=n_clusters,
            rho=rho,
            xi=xi,
            distance_metric=distance_metric,
            max_iterations=max_iteration,
            kmeans_plus_plus=kmean_pp
        )
        
        self.model = KEFRiN(config)
        self.y = y
        self.p = p
    
    def apply_kefrin(self) -> np.ndarray:
        """Apply KEFRiN algorithm (legacy interface)"""
        return self.model.fit_predict(self.y, self.p)


# For backward compatibility, create an alias
KEFRiN_Original = KEFRiN_Legacy

# Export main classes and functions
__all__ = [
    'DistanceMetric',
    'PreprocessingMethod', 
    'KEFRiNConfig',
    'DataPreprocessor',
    'OptimizedDistance',
    'KEFRiN',
    'KEFRiNe',
    'KEFRiNc', 
    'KEFRiNm',
    'KEFRiN_Legacy',
    'KEFRiN_Original'
]

# Export main classes and functions
__all__ = [
    'DistanceMetric',
    'PreprocessingMethod', 
    'KEFRiNConfig',
    'DataPreprocessor',
    'OptimizedDistance',
    'KEFRiN',
    'KEFRiNe',
    'KEFRiNc', 
    'KEFRiNm',
    'KEFRiN_Legacy',
    'KEFRiN_Original'
] 
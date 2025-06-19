"""
Optimized Processing Tools for KEFRiN
Handles data preprocessing with improved performance for large datasets
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from scipy import sparse
import logging

logger = logging.getLogger(__name__)


class OptimizedPreprocessor:
    """
    Optimized preprocessing class for feature and network data
    Designed to handle large datasets efficiently
    """
    
    @staticmethod
    def preprocess_features(y: np.ndarray, method: str = 'z-score', 
                           handle_categorical: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess feature matrix Y with various methods
        
        Args:
            y: Feature matrix (n_samples, n_features)
            method: Preprocessing method ('z-score', 'range', 'min-max', 'none')
            handle_categorical: Whether to apply one-hot encoding for categorical features
            
        Returns:
            Preprocessed matrix and metadata
        """
        metadata = {'original_shape': y.shape, 'method': method}
        
        if method == 'none':
            return y.copy(), metadata
        
        # Handle categorical features if specified
        if handle_categorical and y.dtype in ['object', 'category']:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            y_encoded = encoder.fit_transform(y)
            metadata['encoder'] = encoder
            y = y_encoded
        
        if method == 'z-score':
            # Use in-place operations where possible for memory efficiency
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            y_std[y_std == 0] = 1.0  # Avoid division by zero
            
            y_processed = (y - y_mean) / y_std
            metadata.update({'mean': y_mean, 'std': y_std})
            
        elif method == 'range':
            y_mean = np.mean(y, axis=0)
            y_range = np.ptp(y, axis=0)  # Peak-to-peak (max - min)
            y_range[y_range == 0] = 1.0
            
            y_processed = (y - y_mean) / y_range
            metadata.update({'mean': y_mean, 'range': y_range})
            
        elif method == 'min-max':
            y_min = np.min(y, axis=0)
            y_max = np.max(y, axis=0)
            y_range = y_max - y_min
            y_range[y_range == 0] = 1.0
            
            y_processed = (y - y_min) / y_range
            metadata.update({'min': y_min, 'max': y_max, 'range': y_range})
            
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
        
        return y_processed, metadata
    
    @staticmethod
    def preprocess_network(p: np.ndarray, method: str = 'modularity') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess network/adjacency matrix P
        
        Args:
            p: Network matrix (n_samples, n_samples) 
            method: Preprocessing method ('modularity', 'uniform', 'laplacian', 'none')
            
        Returns:
            Preprocessed matrix and metadata
        """
        metadata = {'original_shape': p.shape, 'method': method}
        n = p.shape[0]
        
        if method == 'none':
            return p.copy(), metadata
        
        # Ensure symmetry for undirected networks
        if not np.allclose(p, p.T):
            logger.warning("Network matrix is not symmetric. Making it symmetric.")
            p = (p + p.T) / 2
        
        if method == 'uniform':
            # Subtract constant random interaction (mean)
            p_mean = np.mean(p)
            p_processed = p - p_mean
            metadata['mean'] = p_mean
            
        elif method == 'modularity':
            # Modularity-based preprocessing
            p_row_sum = np.sum(p, axis=1)
            p_col_sum = np.sum(p, axis=0)
            p_total = np.sum(p)
            
            if p_total > 0:
                # Compute expected edges under null model
                expected = np.outer(p_row_sum, p_col_sum) / p_total
                p_processed = p - expected
            else:
                p_processed = p.copy()
            
            metadata.update({
                'total_weight': p_total,
                'row_sums': p_row_sum,
                'col_sums': p_col_sum
            })
            
        elif method == 'laplacian':
            # Laplacian transformation using NetworkX for efficiency
            if sparse.issparse(p):
                G = nx.from_scipy_sparse_matrix(p)
            else:
                G = nx.from_numpy_array(p)
            
            # Compute normalized Laplacian
            laplacian = nx.normalized_laplacian_matrix(G, nodelist=range(n))
            p_processed = laplacian.toarray() if sparse.issparse(laplacian) else laplacian
            
            metadata['graph_nodes'] = G.number_of_nodes()
            metadata['graph_edges'] = G.number_of_edges()
            
        else:
            raise ValueError(f"Unknown network preprocessing method: {method}")
        
        return p_processed, metadata


class OptimizedMetrics:
    """
    Optimized clustering evaluation metrics
    """
    
    @staticmethod
    def compute_cluster_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive clustering evaluation metrics
        
        Args:
            labels_true: True cluster labels
            labels_pred: Predicted cluster labels
            
        Returns:
            Dictionary of metrics
        """
        from sklearn import metrics
        
        metrics_dict = {
            'adjusted_rand_score': metrics.adjusted_rand_score(labels_true, labels_pred),
            'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(labels_true, labels_pred),
            'normalized_mutual_info_score': metrics.normalized_mutual_info_score(labels_true, labels_pred),
            'homogeneity_score': metrics.homogeneity_score(labels_true, labels_pred),
            'completeness_score': metrics.completeness_score(labels_true, labels_pred),
            'v_measure_score': metrics.v_measure_score(labels_true, labels_pred),
        }
        
        # Add precision, recall, and F1 if applicable
        try:
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                labels_true, labels_pred, average='weighted'
            )
            metrics_dict.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        except:
            pass
        
        return metrics_dict
    
    @staticmethod
    def compute_silhouette_analysis(X: np.ndarray, labels: np.ndarray, 
                                   sample_size: Optional[int] = 10000) -> Dict[str, float]:
        """
        Compute silhouette analysis with sampling for large datasets
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            sample_size: Maximum number of samples to use for computation
            
        Returns:
            Silhouette metrics
        """
        from sklearn import metrics
        
        n_samples = X.shape[0]
        
        # Sample data if too large
        if sample_size is not None and n_samples > sample_size:
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
        
        try:
            silhouette_avg = metrics.silhouette_score(X_sample, labels_sample)
            silhouette_samples = metrics.silhouette_samples(X_sample, labels_sample)
            
            return {
                'silhouette_score': silhouette_avg,
                'silhouette_std': np.std(silhouette_samples),
                'silhouette_min': np.min(silhouette_samples),
                'silhouette_max': np.max(silhouette_samples)
            }
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            return {'silhouette_score': -1.0}


class DataLoader:
    """
    Optimized data loading utilities for various formats
    """
    
    @staticmethod
    def load_feature_network_data(y_path: str, p_path: str, 
                                 gt_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load feature matrix, network matrix, and optionally ground truth labels
        
        Args:
            y_path: Path to feature matrix
            p_path: Path to network matrix  
            gt_path: Optional path to ground truth labels
            
        Returns:
            Feature matrix, network matrix, and optional ground truth
        """
        # Load feature matrix
        if y_path.endswith('.npy'):
            Y = np.load(y_path)
        elif y_path.endswith('.npz'):
            Y = np.load(y_path)['arr_0']
        elif y_path.endswith('.csv'):
            Y = np.loadtxt(y_path, delimiter=',')
        elif y_path.endswith('.dat'):
            Y = np.loadtxt(y_path)
        else:
            raise ValueError(f"Unsupported file format for features: {y_path}")
        
        # Load network matrix
        if p_path.endswith('.npy'):
            P = np.load(p_path)
        elif p_path.endswith('.npz'):
            P = np.load(p_path)['arr_0']
        elif p_path.endswith('.mtx'):
            from scipy.io import mmread
            P = mmread(p_path).toarray()
        elif p_path.endswith('.csv'):
            P = np.loadtxt(p_path, delimiter=',')
        elif p_path.endswith('.dat'):
            P = np.loadtxt(p_path)
        else:
            raise ValueError(f"Unsupported file format for network: {p_path}")
        
        # Load ground truth if provided
        GT = None
        if gt_path is not None:
            if gt_path.endswith('.npy'):
                GT = np.load(gt_path)
            elif gt_path.endswith('.csv'):
                GT = np.loadtxt(gt_path, delimiter=',', dtype=int)
            elif gt_path.endswith('.dat'):
                GT = np.loadtxt(gt_path, dtype=int)
            else:
                raise ValueError(f"Unsupported file format for ground truth: {gt_path}")
        
        logger.info(f"Loaded data: Y{Y.shape}, P{P.shape}" + 
                   (f", GT{GT.shape}" if GT is not None else ""))
        
        return Y, P, GT
    
    @staticmethod
    def validate_data(Y: np.ndarray, P: np.ndarray, GT: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate input data and return summary statistics
        
        Args:
            Y: Feature matrix
            P: Network matrix
            GT: Optional ground truth labels
            
        Returns:
            Validation summary
        """
        validation = {
            'n_samples': Y.shape[0],
            'n_features': Y.shape[1],
            'network_shape': P.shape,
            'is_symmetric': np.allclose(P, P.T),
            'has_self_loops': np.any(np.diag(P) != 0),
            'feature_dtypes': str(Y.dtype),
            'network_dtype': str(P.dtype),
            'feature_missing': np.any(np.isnan(Y)) or np.any(np.isinf(Y)),
            'network_missing': np.any(np.isnan(P)) or np.any(np.isinf(P)),
        }
        
        if GT is not None:
            validation.update({
                'n_true_clusters': len(np.unique(GT)),
                'gt_shape': GT.shape,
                'gt_dtype': str(GT.dtype)
            })
        
        # Check for potential issues
        issues = []
        if Y.shape[0] != P.shape[0]:
            issues.append("Mismatch between number of samples in Y and P")
        if P.shape[0] != P.shape[1]:
            issues.append("Network matrix P is not square")
        if validation['feature_missing']:
            issues.append("Missing values found in feature matrix")
        if validation['network_missing']:
            issues.append("Missing values found in network matrix")
        
        validation['issues'] = issues
        
        return validation


# Legacy compatibility functions
def flat_cluster_results(cluster_results: Dict[int, list]) -> Tuple[np.ndarray, list]:
    """
    Convert cluster results dictionary to flat arrays (legacy compatibility)
    
    Args:
        cluster_results: Dictionary mapping cluster IDs to node lists
        
    Returns:
        Labels array and indices list
    """
    labels_pred_indices = []
    for k, v in cluster_results.items():
        labels_pred_indices.extend(v)
    
    labels_pred = np.zeros(len(labels_pred_indices), dtype=int)
    for k, v in cluster_results.items():
        for node_id in v:
            labels_pred[node_id] = k
    
    return labels_pred, labels_pred_indices


def flat_ground_truth(ground_truth: list) -> Tuple[list, list]:
    """
    Convert ground truth cluster sizes to flat labels (legacy compatibility)
    
    Args:
        ground_truth: List of cluster sizes
        
    Returns:
        Labels list and indices list
    """
    labels_true = []
    labels_true_indices = []
    
    current_id = 0
    for cluster_id, cluster_size in enumerate(ground_truth):
        cluster_indices = list(range(current_id, current_id + cluster_size))
        labels_true.extend([cluster_id] * cluster_size)
        labels_true_indices.extend(cluster_indices)
        current_id += cluster_size
    
    return labels_true, labels_true_indices


# Backward compatibility aliases
preprocess_y = OptimizedPreprocessor.preprocess_features
preprocess_p = OptimizedPreprocessor.preprocess_network 
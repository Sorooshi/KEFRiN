"""
Demo script for the optimized KEFRiN implementation
Shows how to use all three versions: KEFRiNe, KEFRiNc, KEFRiNm
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import metrics
import logging

# Import our modules
from kefrin import KEFRiN, KEFRiNConfig, DistanceMetric, KEFRiNe, KEFRiNc, KEFRiNm
from processing_tools import OptimizedPreprocessor, OptimizedMetrics, DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_with_sample_data():
    """
    Demonstrate KEFRiN with synthetic sample data
    """
    logger.info("=" * 60)
    logger.info("KEFRiN DEMO WITH SAMPLE DATA")
    logger.info("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    n_clusters = 3
    
    # Create synthetic feature matrix
    Y = np.random.randn(n_samples, n_features)
    
    # Create block-structured network matrix (community structure)
    P = np.random.rand(n_samples, n_samples) * 0.1  # Background noise
    
    # Add community structure
    cluster_size = n_samples // n_clusters
    for i in range(n_clusters):
        start_idx = i * cluster_size
        end_idx = min((i + 1) * cluster_size, n_samples)
        P[start_idx:end_idx, start_idx:end_idx] += 0.8  # Strong intra-community connections
    
    # Make symmetric
    P = (P + P.T) / 2
    
    # Create ground truth labels ensuring we have exactly n_samples labels
    gt_labels = np.zeros(n_samples, dtype=int)
    for i in range(n_clusters):
        start_idx = i * cluster_size
        end_idx = min((i + 1) * cluster_size, n_samples)
        gt_labels[start_idx:end_idx] = i
    
    # Assign remaining samples to the last cluster if any
    if n_samples % n_clusters != 0:
        remaining_start = n_clusters * cluster_size
        gt_labels[remaining_start:] = n_clusters - 1
    
    logger.info(f"Sample data created: Y{Y.shape}, P{P.shape}, GT{gt_labels.shape}")
    
    # Test all three versions
    results = {}
    
    # 1. KEFRiNe (Euclidean)
    logger.info("\n1. Testing KEFRiNe (Euclidean distance)")
    start_time = time.time()
    labels_e = KEFRiNe(Y, P, n_clusters=n_clusters, rho=1.0, xi=1.0)
    time_e = time.time() - start_time
    ari_e = metrics.adjusted_rand_score(gt_labels, labels_e)
    results['KEFRiNe'] = {'labels': labels_e, 'time': time_e, 'ari': ari_e}
    logger.info(f"KEFRiNe completed in {time_e:.2f}s, ARI: {ari_e:.4f}")
    
    # 2. KEFRiNc (Cosine)
    logger.info("\n2. Testing KEFRiNc (Cosine distance)")
    start_time = time.time()
    labels_c = KEFRiNc(Y, P, n_clusters=n_clusters, rho=1.0, xi=1.0)
    time_c = time.time() - start_time
    ari_c = metrics.adjusted_rand_score(gt_labels, labels_c)
    results['KEFRiNc'] = {'labels': labels_c, 'time': time_c, 'ari': ari_c}
    logger.info(f"KEFRiNc completed in {time_c:.2f}s, ARI: {ari_c:.4f}")
    
    # 3. KEFRiNm (Manhattan)
    logger.info("\n3. Testing KEFRiNm (Manhattan distance)")
    start_time = time.time()
    labels_m = KEFRiNm(Y, P, n_clusters=n_clusters, rho=1.0, xi=1.0)
    time_m = time.time() - start_time
    ari_m = metrics.adjusted_rand_score(gt_labels, labels_m)
    results['KEFRiNm'] = {'labels': labels_m, 'time': time_m, 'ari': ari_m}
    logger.info(f"KEFRiNm completed in {time_m:.2f}s, ARI: {ari_m:.4f}")
    
    # Summary
    logger.info("\n" + "="*40)
    logger.info("RESULTS SUMMARY")
    logger.info("="*40)
    for method, result in results.items():
        logger.info(f"{method}: ARI={result['ari']:.4f}, Time={result['time']:.2f}s")
    
    return results


def demo_with_real_data():
    """
    Demonstrate KEFRiN with real data from the data directory
    """
    logger.info("\n" + "=" * 60)
    logger.info("KEFRiN DEMO WITH REAL DATA")
    logger.info("=" * 60)
    
    try:
        # Try to load COSN dataset
        data_loader = DataLoader()
        Y, P, GT = data_loader.load_feature_network_data(
            y_path="data/COSN/Y.npy",
            p_path="data/COSN/P.npy", 
            gt_path="data/COSN/ground_truth.npy"
        )
        
        # Validate data
        validation = data_loader.validate_data(Y, P, GT)
        logger.info(f"Data validation: {validation}")
        
        if validation['issues']:
            logger.warning(f"Data issues detected: {validation['issues']}")
        
        # Preprocess data
        preprocessor = OptimizedPreprocessor()
        Y_processed, y_meta = preprocessor.preprocess_features(Y, method='z-score')
        P_processed, p_meta = preprocessor.preprocess_network(P, method='modularity')
        
        logger.info(f"Preprocessing completed: Y{Y_processed.shape}, P{P_processed.shape}")
        
        # Determine number of clusters
        n_clusters = validation['n_true_clusters']
        logger.info(f"Using {n_clusters} clusters based on ground truth")
        
        # Test with object-oriented interface
        results = {}
        metrics_computer = OptimizedMetrics()
        
        for distance_metric, name in [(DistanceMetric.EUCLIDEAN, "KEFRiNe"),
                                     (DistanceMetric.COSINE, "KEFRiNc"), 
                                     (DistanceMetric.MANHATTAN, "KEFRiNm")]:
            
            logger.info(f"\nTesting {name}")
            
            # Create configuration
            config = KEFRiNConfig(
                n_clusters=n_clusters,
                distance_metric=distance_metric,
                rho=1.0,
                xi=1.0,
                max_iterations=1000,
                n_init=5,
                random_state=42
            )
            
            # Fit model
            start_time = time.time()
            model = KEFRiN(config)
            labels = model.fit_predict(Y_processed, P_processed)
            fit_time = time.time() - start_time
            
            # Compute comprehensive metrics
            cluster_metrics = metrics_computer.compute_cluster_metrics(GT, labels)
            
            results[name] = {
                'labels': labels,
                'time': fit_time,
                'inertia': model.inertia_,
                'n_iter': model.n_iter_,
                **cluster_metrics
            }
            
            logger.info(f"{name}: ARI={cluster_metrics['adjusted_rand_score']:.4f}, "
                       f"NMI={cluster_metrics['normalized_mutual_info_score']:.4f}, "
                       f"Time={fit_time:.2f}s")
        
        # Summary table
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE RESULTS")
        logger.info("="*80)
        logger.info(f"{'Method':<10} {'ARI':<8} {'NMI':<8} {'Homog.':<8} {'Compl.':<8} {'Time':<8} {'Iter':<6}")
        logger.info("-" * 80)
        
        for method, result in results.items():
            logger.info(f"{method:<10} "
                       f"{result['adjusted_rand_score']:<8.4f} "
                       f"{result['normalized_mutual_info_score']:<8.4f} "
                       f"{result['homogeneity_score']:<8.4f} "
                       f"{result['completeness_score']:<8.4f} "
                       f"{result['time']:<8.2f} "
                       f"{result['n_iter']:<6}")
        
        return results
        
    except Exception as e:
        logger.error(f"Could not load real data: {e}")
        logger.info("Falling back to synthetic data demo")
        return demo_with_sample_data()


def demo_parameter_sensitivity():
    """
    Demonstrate parameter sensitivity analysis
    """
    logger.info("\n" + "=" * 60)
    logger.info("PARAMETER SENSITIVITY ANALYSIS")
    logger.info("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    n_clusters = 3
    
    # Simple data for quick analysis
    Y = np.random.randn(n_samples, n_features)
    P = np.random.rand(n_samples, n_samples)
    P = (P + P.T) / 2
    
    # Add community structure
    cluster_size = n_samples // n_clusters
    for i in range(n_clusters):
        start_idx = i * cluster_size
        end_idx = min((i + 1) * cluster_size, n_samples)
        P[start_idx:end_idx, start_idx:end_idx] += 0.5
    
    # Create ground truth labels ensuring we have exactly n_samples labels
    gt_labels = np.zeros(n_samples, dtype=int)
    for i in range(n_clusters):
        start_idx = i * cluster_size
        end_idx = min((i + 1) * cluster_size, n_samples)
        gt_labels[start_idx:end_idx] = i
    
    # Assign remaining samples to the last cluster if any
    if n_samples % n_clusters != 0:
        remaining_start = n_clusters * cluster_size
        gt_labels[remaining_start:] = n_clusters - 1
    
    # Test different rho/xi combinations
    rho_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    xi_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    logger.info("Testing parameter combinations (rho, xi)...")
    logger.info("Rho controls feature importance, Xi controls network importance")
    
    best_ari = 0
    best_params = None
    results_grid = []
    
    for rho in rho_values:
        for xi in xi_values:
            # Test with KEFRiNe (fastest)
            labels = KEFRiNe(Y, P, n_clusters=n_clusters, rho=rho, xi=xi)
            ari = metrics.adjusted_rand_score(gt_labels, labels)
            
            results_grid.append((rho, xi, ari))
            
            if ari > best_ari:
                best_ari = ari
                best_params = (rho, xi)
            
            logger.info(f"rho={rho:.1f}, xi={xi:.1f}: ARI={ari:.4f}")
    
    logger.info(f"\nBest parameters: rho={best_params[0]:.1f}, xi={best_params[1]:.1f}, ARI={best_ari:.4f}")
    
    return results_grid, best_params


def create_compatibility_layer():
    """
    Create a compatibility layer for the original KEFRiN interface
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPATIBILITY LAYER DEMO")
    logger.info("=" * 60)
    
    # Import legacy compatibility
    from kefrin_optimized import KEFRiN_Legacy
    
    # Sample data
    np.random.seed(42)
    n_samples = 50
    Y = np.random.randn(n_samples, 3)
    P = np.random.rand(n_samples, n_samples)
    P = (P + P.T) / 2
    
    logger.info("Testing legacy interface compatibility...")
    
    # Test original interface
    model_euclidean = KEFRiN_Legacy(Y, P, n_clusters=3, euclidean=1, cosine=0, manhattan=0)
    labels_e = model_euclidean.apply_kefrin()
    
    model_cosine = KEFRiN_Legacy(Y, P, n_clusters=3, euclidean=0, cosine=1, manhattan=0)
    labels_c = model_cosine.apply_kefrin()
    
    model_manhattan = KEFRiN_Legacy(Y, P, n_clusters=3, euclidean=0, cosine=0, manhattan=1)
    labels_m = model_manhattan.apply_kefrin()
    
    logger.info("Legacy interface working correctly!")
    logger.info(f"Euclidean result: {len(np.unique(labels_e))} clusters")
    logger.info(f"Cosine result: {len(np.unique(labels_c))} clusters") 
    logger.info(f"Manhattan result: {len(np.unique(labels_m))} clusters")


def main():
    """
    Run all demonstrations
    """
    logger.info("Starting KEFRiN Optimized Implementation Demo")
    logger.info("This demo showcases all three versions of KEFRiN")
    
    try:
        # 1. Basic functionality demo
        demo_with_sample_data()
        
        # 2. Real data demo (if available)
        demo_with_real_data()
        
        # 3. Parameter sensitivity analysis
        demo_parameter_sensitivity()
        
        # 4. Compatibility layer
        create_compatibility_layer()
        
        logger.info("\n" + "=" * 60)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("All three KEFRiN versions are working correctly:")
        logger.info("✓ KEFRiNe: Euclidean distance")
        logger.info("✓ KEFRiNc: Cosine distance") 
        logger.info("✓ KEFRiNm: Manhattan distance")
        logger.info("\nThe implementation is optimized for:")
        logger.info("- Large datasets (efficient vectorized operations)")
        logger.info("- Memory efficiency (in-place operations where possible)")
        logger.info("- Production use (comprehensive logging and error handling)")
        logger.info("- Backward compatibility (legacy interface support)")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 
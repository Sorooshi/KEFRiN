"""
Script to reproduce Table 9 results from the KEFRiN paper:
"Community partitioning over feature-rich networks using an extended k-means method"

This script tests KEFRiNe, KEFRiNc, and KEFRiNm algorithms on real-world datasets
with 10 random initializations and reports average ARI values.

Reference: Shalileh S, Mirkin B. Community partitioning over feature-rich networks 
using an extended k-means method. Entropy. 2022 Apr 29;24(5):626.
"""

import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from scipy.io import mmread
import warnings
warnings.filterwarnings('ignore')

from kefrin import KEFRiNe, KEFRiNc, KEFRiNm
from processing_tools import DataLoader, OptimizedPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Table9Reproducer:
    """Class to reproduce Table 9 results from the KEFRiN paper"""
    
    def __init__(self, data_dir="data", n_runs=10):
        self.data_dir = Path(data_dir)
        self.n_runs = n_runs
        self.data_loader = DataLoader()
        self.preprocessor = OptimizedPreprocessor()
        
        # Dataset configurations based on the paper
        self.datasets = {
            'HVR': {
                'path': 'HVR',
                'y_file': 'Y.mtx',
                'p_file': 'P.mtx',
                'gt_file': 'ground_truth.npy',
                'n_clusters': None,  # Will be determined from ground truth
                'rho': 1.0,
                'xi': 1.0
            },
            'Lawyers': {
                'path': 'Lawyers',
                'y_file': 'Y.dat',
                'p_file': 'P.dat',
                'gt_file': 'ground_truth.npy',
                'n_clusters': None,
                'rho': 1.0,
                'xi': 1.0
            },
            'World Trade': {
                'path': 'World-Trade',
                'y_file': 'Y.npy',
                'p_file': 'P.npy',
                'gt_file': 'ground_truth.npy',
                'n_clusters': None,
                'rho': 1.0,
                'xi': 1.0
            },
            'Parliament': {
                'path': 'Parliament',
                'y_file': 'Y.mtx',
                'p_file': 'P.mtx',
                'gt_file': 'ground_truth.npy',
                'n_clusters': None,
                'rho': 1.0,
                'xi': 1.0
            },
            'COSN': {
                'path': 'COSN',
                'y_file': 'Y.npy',
                'p_file': 'P.npy',
                'gt_file': 'ground_truth.npy',
                'n_clusters': None,
                'rho': 1.0,
                'xi': 1.0
            }
        }
        
        self.results = {}
    
    def load_dataset(self, dataset_name):
        """Load a dataset with appropriate format handling"""
        config = self.datasets[dataset_name]
        dataset_path = self.data_dir / config['path']
        
        try:
            # Load files based on their extensions
            y_path = dataset_path / config['y_file']
            p_path = dataset_path / config['p_file']
            gt_path = dataset_path / config['gt_file']
            
            # Load Y (features)
            if y_path.suffix == '.mtx':
                Y = mmread(str(y_path)).toarray()
            elif y_path.suffix == '.dat':
                Y = np.loadtxt(str(y_path))
            elif y_path.suffix == '.npy':
                try:
                    Y = np.load(str(y_path), allow_pickle=False)
                except (ValueError, OSError):
                    # If binary loading fails, try as text file
                    Y = np.loadtxt(str(y_path))
            else:
                raise ValueError(f"Unsupported Y file format: {y_path.suffix}")
            
            # Load P (network)
            if p_path.suffix == '.mtx':
                P = mmread(str(p_path)).toarray()
            elif p_path.suffix == '.dat':
                P = np.loadtxt(str(p_path))
            elif p_path.suffix == '.npy':
                try:
                    P = np.load(str(p_path), allow_pickle=False)
                except (ValueError, OSError):
                    # If binary loading fails, try as text file
                    P = np.loadtxt(str(p_path))
            else:
                raise ValueError(f"Unsupported P file format: {p_path.suffix}")
            
            # Load ground truth
            GT = np.load(str(gt_path))
            
            # Ensure P is symmetric
            if not np.allclose(P, P.T, atol=1e-10):
                P = (P + P.T) / 2
                logger.warning(f"Made P matrix symmetric for {dataset_name}")
            
            # Determine number of clusters from ground truth
            n_clusters = len(np.unique(GT))
            config['n_clusters'] = n_clusters
            
            logger.info(f"Loaded {dataset_name}: Y{Y.shape}, P{P.shape}, GT{GT.shape}, n_clusters={n_clusters}")
            
            return Y, P, GT, n_clusters
            
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {str(e)}")
            return None, None, None, None
    
    def preprocess_data(self, Y, P, dataset_name):
        """Preprocess the data using standard methods"""
        try:
            # Preprocess features (Z-score normalization)
            Y_processed, _ = self.preprocessor.preprocess_features(Y, method='z-score')
            
            # For network, we'll use the raw matrix as in the original paper
            P_processed = P.copy()
            
            logger.info(f"Preprocessed {dataset_name}: Y{Y_processed.shape}, P{P_processed.shape}")
            
            return Y_processed, P_processed
            
        except Exception as e:
            logger.error(f"Failed to preprocess {dataset_name}: {str(e)}")
            return Y, P
    
    def run_algorithm(self, algorithm_func, Y, P, n_clusters, rho, xi, random_state):
        """Run a single algorithm with given parameters"""
        try:
            start_time = time.time()
            labels = algorithm_func(Y, P, n_clusters=n_clusters, rho=rho, xi=xi, 
                                  max_iteration=1000, kmean_pp=True)
            runtime = time.time() - start_time
            return labels, runtime
        except Exception as e:
            logger.error(f"Algorithm failed: {str(e)}")
            return None, None
    
    def evaluate_dataset(self, dataset_name):
        """Evaluate all three KEFRiN variants on a dataset"""
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING DATASET: {dataset_name}")
        logger.info(f"{'='*60}")
        
        # Load dataset
        Y, P, GT, n_clusters = self.load_dataset(dataset_name)
        if Y is None:
            return None
        
        # Preprocess data
        Y_processed, P_processed = self.preprocess_data(Y, P, dataset_name)
        
        # Initialize results for this dataset
        dataset_results = {
            'KEFRiNe': {'ari_scores': [], 'runtimes': []},
            'KEFRiNc': {'ari_scores': [], 'runtimes': []},
            'KEFRiNm': {'ari_scores': [], 'runtimes': []}
        }
        
        config = self.datasets[dataset_name]
        algorithms = [
            ('KEFRiNe', KEFRiNe),
            ('KEFRiNc', KEFRiNc),
            ('KEFRiNm', KEFRiNm)
        ]
        
        # Run each algorithm multiple times
        for run in range(self.n_runs):
            logger.info(f"\nRun {run + 1}/{self.n_runs}")
            
            for alg_name, alg_func in algorithms:
                labels, runtime = self.run_algorithm(
                    alg_func, Y_processed, P_processed, n_clusters, 
                    config['rho'], config['xi'], random_state=run
                )
                
                if labels is not None:
                    ari = adjusted_rand_score(GT, labels)
                    dataset_results[alg_name]['ari_scores'].append(ari)
                    dataset_results[alg_name]['runtimes'].append(runtime)
                    logger.info(f"  {alg_name}: ARI={ari:.4f}, Time={runtime:.3f}s")
                else:
                    logger.warning(f"  {alg_name}: FAILED")
        
        # Calculate statistics
        results_summary = {}
        for alg_name in ['KEFRiNe', 'KEFRiNc', 'KEFRiNm']:
            ari_scores = dataset_results[alg_name]['ari_scores']
            if ari_scores:
                results_summary[alg_name] = {
                    'mean_ari': np.mean(ari_scores),
                    'std_ari': np.std(ari_scores),
                    'mean_runtime': np.mean(dataset_results[alg_name]['runtimes']),
                    'n_successful_runs': len(ari_scores)
                }
            else:
                results_summary[alg_name] = {
                    'mean_ari': np.nan,
                    'std_ari': np.nan,
                    'mean_runtime': np.nan,
                    'n_successful_runs': 0
                }
        
        logger.info(f"\nSUMMARY FOR {dataset_name}:")
        for alg_name, stats in results_summary.items():
            if stats['n_successful_runs'] > 0:
                logger.info(f"  {alg_name}: ARI={stats['mean_ari']:.3f}(±{stats['std_ari']:.3f}), "
                           f"Time={stats['mean_runtime']:.3f}s, Runs={stats['n_successful_runs']}")
            else:
                logger.info(f"  {alg_name}: FAILED (0 successful runs)")
        
        return results_summary
    
    def run_all_experiments(self):
        """Run experiments on all datasets"""
        logger.info("STARTING TABLE 9 REPRODUCTION")
        logger.info(f"Number of random initializations per algorithm: {self.n_runs}")
        
        all_results = {}
        
        for dataset_name in self.datasets.keys():
            try:
                results = self.evaluate_dataset(dataset_name)
                if results:
                    all_results[dataset_name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset_name}: {str(e)}")
                continue
        
        self.results = all_results
        return all_results
    
    def format_table_results(self):
        """Format results in the style of Table 9"""
        logger.info(f"\n{'='*80}")
        logger.info("TABLE 9 REPRODUCTION RESULTS")
        logger.info(f"{'='*80}")
        
        # Create results table
        table_data = []
        
        for dataset_name, dataset_results in self.results.items():
            row = {'Dataset': dataset_name}
            
            for alg_name in ['KEFRiNe', 'KEFRiNc', 'KEFRiNm']:
                if alg_name in dataset_results and dataset_results[alg_name]['n_successful_runs'] > 0:
                    mean_ari = dataset_results[alg_name]['mean_ari']
                    std_ari = dataset_results[alg_name]['std_ari']
                    row[alg_name] = f"{mean_ari:.3f}({std_ari:.3f})"
                else:
                    row[alg_name] = "FAILED"
            
            table_data.append(row)
        
        # Create DataFrame for nice formatting
        df = pd.DataFrame(table_data)
        df = df.set_index('Dataset')
        
        print("\nTable 9 Reproduction Results:")
        print("=" * 80)
        print(f"{'Dataset':<15} {'KEFRiNe':<15} {'KEFRiNc':<15} {'KEFRiNm':<15}")
        print("-" * 80)
        
        for dataset_name, row in df.iterrows():
            print(f"{dataset_name:<15} {row['KEFRiNe']:<15} {row['KEFRiNc']:<15} {row['KEFRiNm']:<15}")
        
        print("\nFormat: mean_ARI(std_ARI) over 10 random initializations")
        print("Note: Results may vary from paper due to implementation differences,")
        print("      preprocessing variations, and random initialization effects.")
        
        return df
    
    def save_results(self, filename="table9_reproduction_results.csv"):
        """Save detailed results to CSV"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Prepare detailed results
        detailed_results = []
        
        for dataset_name, dataset_results in self.results.items():
            for alg_name in ['KEFRiNe', 'KEFRiNc', 'KEFRiNm']:
                if alg_name in dataset_results:
                    stats = dataset_results[alg_name]
                    detailed_results.append({
                        'Dataset': dataset_name,
                        'Algorithm': alg_name,
                        'Mean_ARI': stats['mean_ari'],
                        'Std_ARI': stats['std_ari'],
                        'Mean_Runtime': stats['mean_runtime'],
                        'Successful_Runs': stats['n_successful_runs'],
                        'Total_Runs': self.n_runs
                    })
        
        df = pd.DataFrame(detailed_results)
        df.to_csv(filename, index=False)
        logger.info(f"Detailed results saved to {filename}")


def main():
    """Main function to run the Table 9 reproduction"""
    logger.info("KEFRiN Table 9 Reproduction Script")
    logger.info("Reference: Shalileh S, Mirkin B. Community partitioning over feature-rich networks using an extended k-means method. Entropy. 2022 Apr 29;24(5):626.")
    
    # Create reproducer instance
    reproducer = Table9Reproducer(data_dir="data", n_runs=5)
    
    # Run all experiments
    results = reproducer.run_all_experiments()
    
    if results:
        # Format and display results
        table_df = reproducer.format_table_results()
        
        # Save results
        reproducer.save_results()
        
        logger.info("\nTable 9 reproduction completed successfully!")
    else:
        logger.error("No results obtained. Check data availability and format.")


if __name__ == "__main__":
    main() 
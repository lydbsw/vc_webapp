#!/usr/bin/env python3
"""
Clustering Model for VC Company Data Analysis

This script converts the Jupyter notebook analysis into a standalone Python script
that performs clustering analysis on venture capital company datasets.

Features:
- Data loading and preprocessing
- Feature engineering and selection
- Optimal cluster number determination using Elbow Method and Silhouette Analysis
- K-means clustering with optimal parameters
- Comparison with other clustering algorithms (Agglomerative, Gaussian Mixture)
- Visualization and analysis of clustering results
- Plot saving functionality
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class VCClusteringAnalysis:
    """
    A comprehensive clustering analysis class for venture capital company data.
    """
    
    def __init__(self, dataset_folder="dataset", output_folder="plots"):
        """
        Initialize the clustering analysis.
        
        Args:
            dataset_folder (str): Path to folder containing CSV files
            output_folder (str): Path to folder where plots will be saved
        """
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.csv_dataframes = {}
        self.combined_df = None
        self.features_for_clustering = None
        self.scaler = StandardScaler()
        self.optimal_k = None
        self.optimal_kmeans_model = None
        self.cluster_results = {}
        
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")
    
    def load_and_preprocess_data(self):
        """
        Load CSV files from the dataset folder and preprocess them.
        Handles missing values and outliers.
        """
        print("=== Data Loading and Preprocessing ===")
        
        # Find all CSV files in the dataset folder
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Dataset folder '{self.dataset_folder}' not found!")
            
        csv_files = [f for f in os.listdir(self.dataset_folder) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in '{self.dataset_folder}' folder!")
        
        print(f"Found {len(csv_files)} CSV files in {self.dataset_folder}")
        
        # Load each CSV file
        for csv_file in csv_files:
            file_path = os.path.join(self.dataset_folder, csv_file)
            df_name = os.path.splitext(csv_file)[0]  # Use filename without extension as dict key
            self.csv_dataframes[df_name] = pd.read_csv(file_path)
            print(f"Imported '{csv_file}' as '{df_name}' - Shape: {self.csv_dataframes[df_name].shape}")
        
        # Preprocess each dataframe
        for df_name, df in self.csv_dataframes.items():
            print(f"\nAnalyzing dataframe: {df_name}")
            print("Data Info:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Missing values: {df.isnull().sum().sum()}")
            
            # Handle missing values
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['int64', 'float64']:
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        print(f"Filled missing values in column '{col}' with median: {median_val}")
                    else:
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else None
                        if mode_val is not None:
                            df[col].fillna(mode_val, inplace=True)
                            print(f"Filled missing values in column '{col}' with mode: {mode_val}")
                        else:
                            print(f"Could not fill missing values in column '{col}' (no mode found or all NaNs)")
            
            # Handle outliers using IQR method
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
                if outliers_count > 0:
                    print(f"Found {outliers_count} outliers in column '{col}'. Capping values.")
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            print("-" * 50)
    
    def combine_dataframes(self):
        """
        Merge all dataframes into a single dataframe for analysis.
        """
        print("\n=== Combining Dataframes ===")
        self.combined_df = pd.concat(self.csv_dataframes.values(), ignore_index=True)
        print(f"Combined dataframe shape: {self.combined_df.shape}")
        print(f"Combined dataframe columns: {list(self.combined_df.columns)}")
        print("\nFirst 5 rows of combined data:")
        print(self.combined_df.head())
        
    def feature_engineering(self):
        """
        Perform feature engineering including one-hot encoding and scaling.
        """
        print("\n=== Feature Engineering ===")
        
        # Define numerical and categorical columns for clustering
        numerical_cols = [
            'Last Funding Amount (in USD)',
            'Total Funding Amount (in USD)', 
            'Number of Lead Investors',
            'Number of Investors'
        ]
        
        categorical_cols = [
            'Last Funding Type',
            'Headquarters Location',
            'Primary Industry'
        ]
        
        # Filter to only include columns that exist in the dataframe
        categorical_cols_present = [col for col in categorical_cols if col in self.combined_df.columns]
        numerical_cols_present = [col for col in numerical_cols if col in self.combined_df.columns]
        
        print(f"Numerical columns for clustering: {numerical_cols_present}")
        print(f"Categorical columns for clustering: {categorical_cols_present}")
        
        # Apply one-hot encoding to categorical columns
        combined_df_encoded = pd.get_dummies(self.combined_df, columns=categorical_cols_present, dummy_na=False)
        
        # Get all one-hot encoded column names
        one_hot_encoded_cols = [col for col in combined_df_encoded.columns 
                               if any(col.startswith(cat_col + '_') for cat_col in categorical_cols_present)]
        
        print(f"Created {len(one_hot_encoded_cols)} one-hot encoded features")
        
        # Combine numerical and encoded categorical features
        self.features_for_clustering = combined_df_encoded[numerical_cols_present + one_hot_encoded_cols].copy()
        
        # Convert numerical columns to numeric and handle any conversion issues
        for col in numerical_cols_present:
            if self.features_for_clustering[col].dtype == 'object':
                self.features_for_clustering[col] = pd.to_numeric(self.features_for_clustering[col], errors='coerce')
        
        # Scale numerical features
        self.features_for_clustering.loc[:, numerical_cols_present] = self.scaler.fit_transform(
            self.features_for_clustering[numerical_cols_present]
        )
        
        print(f"Features for clustering shape: {self.features_for_clustering.shape}")
        print("\nFeature engineering completed successfully!")
        
    def find_optimal_clusters(self, k_range=range(2, 11)):
        """
        Find optimal number of clusters using Elbow Method and Silhouette Analysis.
        
        Args:
            k_range (range): Range of k values to test
        """
        print(f"\n=== Finding Optimal Number of Clusters ===")
        print(f"Testing k values from {min(k_range)} to {max(k_range)}")
        
        wcss = []
        silhouette_scores = []
        
        for k in k_range:
            print(f"Testing k={k}...")
            # Instantiate KMeans model
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            
            # Fit the KMeans model and get cluster labels
            cluster_labels = kmeans.fit_predict(self.features_for_clustering)
            
            # Append WCSS (inertia_) to the list
            wcss.append(kmeans.inertia_)
            
            # Calculate Silhouette Score
            if len(set(cluster_labels)) > 1:
                silhouette_scores.append(silhouette_score(self.features_for_clustering, cluster_labels))
            else:
                silhouette_scores.append(0)
        
        # Create visualization
        plt.figure(figsize=(15, 6))
        
        # Plot the Elbow Method (WCSS)
        plt.subplot(1, 2, 1)
        plt.plot(k_range, wcss, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        plt.title('Elbow Method for Optimal k')
        plt.xticks(k_range)
        plt.grid(True, alpha=0.3)
        
        # Plot the Silhouette Analysis
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='orange')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method for Optimal k')
        plt.xticks(k_range)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'optimal_clusters_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k based on highest silhouette score
        self.optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
        print(f"\nBased on Silhouette Analysis, the optimal number of clusters (k) is: {self.optimal_k}")
        print(f"Optimal k silhouette score: {max(silhouette_scores):.4f}")
        
        # Store results for later analysis
        self.cluster_analysis_results = {
            'k_range': list(k_range),
            'wcss': wcss,
            'silhouette_scores': silhouette_scores,
            'optimal_k': self.optimal_k
        }
        
    def apply_optimal_kmeans(self):
        """
        Apply K-means clustering with the optimal number of clusters.
        """
        print(f"\n=== Applying K-Means with Optimal Clusters (k={self.optimal_k}) ===")
        
        # Instantiate KMeans with optimal number of clusters
        self.optimal_kmeans_model = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        
        # Fit the model and get cluster labels
        optimal_kmeans_labels = self.optimal_kmeans_model.fit_predict(self.features_for_clustering)
        
        # Add cluster labels to the combined dataframe
        self.combined_df['optimal_kmeans_cluster_label'] = optimal_kmeans_labels
        
        print(f"K-Means clustering applied successfully!")
        print(f"Cluster distribution: {np.bincount(optimal_kmeans_labels)}")
        
    def visualize_optimal_clusters(self):
        """
        Visualize the optimal clusters using PCA for dimensionality reduction.
        """
        print(f"\n=== Visualizing Optimal Clusters ===")
        
        # Apply PCA to reduce dimensions to 2 for visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.features_for_clustering)
        pca_df = pd.DataFrame(data=principal_components, 
                             columns=['Principal Component 1', 'Principal Component 2'])
        
        # Add cluster labels to PCA dataframe
        pca_df['Cluster'] = self.combined_df['optimal_kmeans_cluster_label']
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_df['Principal Component 1'], 
                            pca_df['Principal Component 2'],
                            c=pca_df['Cluster'], 
                            cmap='viridis', 
                            alpha=0.7,
                            s=50)
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'Optimal K-Means Cluster Visualization using PCA (k={self.optimal_k})')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(self.output_folder, 'optimal_clusters_pca_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Total variance explained by 2 components: {sum(pca.explained_variance_ratio_):.2%}")
        
    def analyze_cluster_characteristics(self):
        """
        Analyze and summarize the characteristics of each cluster.
        """
        print(f"\n=== Analyzing Cluster Characteristics ===")
        
        # Define columns for analysis
        numerical_cols = [
            'Last Funding Amount (in USD)',
            'Total Funding Amount (in USD)',
            'Number of Lead Investors',
            'Number of Investors'
        ]
        
        categorical_cols = [
            'Last Funding Type',
            'Headquarters Location',
            'Primary Industry'
        ]
        
        # Filter to existing columns
        numerical_cols_present = [col for col in numerical_cols if col in self.combined_df.columns]
        categorical_cols_present = [col for col in categorical_cols if col in self.combined_df.columns]
        
        # Create aggregation dictionary
        agg_dict = {}
        for col in numerical_cols_present:
            agg_dict[col] = 'mean'
        for col in categorical_cols_present:
            agg_dict[col] = lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
        
        # Calculate cluster characteristics
        cluster_characteristics = self.combined_df.groupby('optimal_kmeans_cluster_label').agg(agg_dict).reset_index()
        
        print("Cluster Characteristics (Mean for numerical, Mode for categorical):")
        print(cluster_characteristics.to_string(index=False))
        
        # Detailed summary for each cluster
        print(f"\n=== Detailed Cluster Analysis ===")
        for cluster_id in sorted(self.combined_df['optimal_kmeans_cluster_label'].unique()):
            cluster_data = self.combined_df[self.combined_df['optimal_kmeans_cluster_label'] == cluster_id]
            print(f"\nCluster {cluster_id} (n={len(cluster_data)} companies):")
            
            for col in numerical_cols_present:
                mean_val = cluster_data[col].mean()
                print(f"  - Average {col}: {mean_val:,.2f}")
            
            for col in categorical_cols_present:
                mode_val = cluster_data[col].mode()[0] if not cluster_data[col].mode().empty else 'N/A'
                print(f"  - Most Frequent {col}: {mode_val}")
            
            print("-" * 50)
        
        return cluster_characteristics
    
    def compare_clustering_algorithms(self):
        """
        Compare K-Means with other clustering algorithms.
        """
        print(f"\n=== Comparing Clustering Algorithms ===")
        
        algorithms = {
            'K-Means': KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10),
            'Agglomerative': AgglomerativeClustering(n_clusters=self.optimal_k),
            'Gaussian Mixture': GaussianMixture(n_components=self.optimal_k, random_state=42)
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            print(f"Running {name}...")
            
            # Fit and predict
            if hasattr(algorithm, 'fit_predict'):
                labels = algorithm.fit_predict(self.features_for_clustering)
            else:
                algorithm.fit(self.features_for_clustering)
                labels = algorithm.predict(self.features_for_clustering)
            
            # Calculate silhouette score
            if len(set(labels)) > 1:
                silhouette = silhouette_score(self.features_for_clustering, labels)
                results[name] = {
                    'labels': labels,
                    'silhouette_score': silhouette
                }
                print(f"{name} Silhouette Score: {silhouette:.4f}")
            else:
                results[name] = {
                    'labels': labels,
                    'silhouette_score': 0
                }
                print(f"Cannot calculate Silhouette Score for {name}: Only one cluster formed.")
        
        # Find best algorithm
        best_algorithm = max(results.keys(), key=lambda x: results[x]['silhouette_score'])
        print(f"\nBest performing algorithm: {best_algorithm} (Silhouette Score: {results[best_algorithm]['silhouette_score']:.4f})")
        
        # Visualize comparison
        self._visualize_algorithm_comparison(results)
        
        self.cluster_results = results
        return results
    
    def _visualize_algorithm_comparison(self, results):
        """
        Create visualizations comparing different clustering algorithms.
        """
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.features_for_clustering)
        
        # Create subplots for each algorithm
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Plot results for each algorithm
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            scatter = ax.scatter(principal_components[:, 0], 
                               principal_components[:, 1],
                               c=result['labels'], 
                               cmap='viridis', 
                               alpha=0.7,
                               s=30)
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_title(f'{name}\nSilhouette Score: {result["silhouette_score"]:.4f}')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Hide the fourth subplot if only 3 algorithms
        if len(results) == 3:
            axes[3].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'clustering_algorithms_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """
        Save clustering results to CSV files.
        """
        print(f"\n=== Saving Results ===")
        
        # Save combined dataframe with cluster labels
        output_file = os.path.join(self.output_folder, 'clustered_data.csv')
        self.combined_df.to_csv(output_file, index=False)
        print(f"Clustered data saved to: {output_file}")
        
        # Save cluster characteristics
        if hasattr(self, 'cluster_results'):
            characteristics = self.analyze_cluster_characteristics()
            char_file = os.path.join(self.output_folder, 'cluster_characteristics.csv')
            characteristics.to_csv(char_file, index=False)
            print(f"Cluster characteristics saved to: {char_file}")
        
        print(f"All plots saved in: {self.output_folder}")
    
    def run_complete_analysis(self):
        """
        Run the complete clustering analysis pipeline.
        """
        print("Starting Complete VC Clustering Analysis...")
        print("=" * 60)
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Combine dataframes
            self.combine_dataframes()
            
            # Step 3: Feature engineering
            self.feature_engineering()
            
            # Step 4: Find optimal number of clusters
            self.find_optimal_clusters()
            
            # Step 5: Apply optimal K-means
            self.apply_optimal_kmeans()
            
            # Step 6: Visualize optimal clusters
            self.visualize_optimal_clusters()
            
            # Step 7: Analyze cluster characteristics
            self.analyze_cluster_characteristics()
            
            # Step 8: Compare clustering algorithms
            self.compare_clustering_algorithms()
            
            # Step 9: Save results
            self.save_results()
            
            print("\n" + "=" * 60)
            print("Complete VC Clustering Analysis Finished Successfully!")
            print(f"Check the '{self.output_folder}' folder for saved plots and results.")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    """
    Main function to run the clustering analysis.
    """
    # Initialize the clustering analysis
    analyzer = VCClusteringAnalysis(dataset_folder="dataset", output_folder="plots")
    
    # Run complete analysis
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()

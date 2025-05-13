import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MetricsAnalyzer:
    def __init__(self, json_directory):
        """
        Initialize the metrics analyzer with a directory containing JSON metrics files.
        
        Args:
            json_directory (str): Path to directory containing JSON files
        """
        self.json_directory = json_directory
        self.json_files = []
        self.metrics_data = {}
        self.config_names = []
        
    def load_json_files(self, pattern="*.json"):
        """
        Load all JSON files matching the pattern in the specified directory.
        
        Args:
            pattern (str): Glob pattern to match JSON files
        """
        self.json_files = glob.glob(os.path.join(self.json_directory, pattern))
        print(f"Found {len(self.json_files)} JSON files")
        
        for file_path in self.json_files:
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    config_name = data.get("config_name", os.path.basename(file_path))
                    self.metrics_data[config_name] = data
                    self.config_names.append(config_name)
                    print(f"Loaded: {config_name}")
                except json.JSONDecodeError:
                    print(f"Error loading {file_path}: Invalid JSON format")
    
    def extract_ngram_diversity_metrics(self):
        """
        Extract n-gram diversity metrics from all loaded JSON files.
        
        Returns:
            dict: Dictionary containing extracted n-gram metrics
        """
        ngram_metrics = {
            'config_name': [],
            'n_gram': [],
            'unique_ratio_original': [],
            'unique_ratio_generated': [],
            'jensen_shannon_divergence': [],
            'original_entropy_normalized': [],
            'generated_entropy_normalized': []
        }
        
        for config_name, data in self.metrics_data.items():
            for n in ['1-gram', '2-gram', '3-gram']:
                ngram_data = data.get('utility_metrics', {}).get('n_gram_diversity', {}).get(n, {})
                if ngram_data:
                    ngram_metrics['config_name'].append(config_name)
                    ngram_metrics['n_gram'].append(n)
                    ngram_metrics['unique_ratio_original'].append(ngram_data.get('unique_ratio_original', 0))
                    ngram_metrics['unique_ratio_generated'].append(ngram_data.get('unique_ratio_generated', 0))
                    ngram_metrics['jensen_shannon_divergence'].append(ngram_data.get('jensen_shannon_divergence', 0))
                    ngram_metrics['original_entropy_normalized'].append(ngram_data.get('original_entropy_normalized', 0))
                    ngram_metrics['generated_entropy_normalized'].append(ngram_data.get('generated_entropy_normalized', 0))
        
        return pd.DataFrame(ngram_metrics)
    
    def extract_similarity_metrics(self):
        """
        Extract semantic similarity metrics from all loaded JSON files.
        
        Returns:
            dict: Dictionary containing extracted similarity metrics
        """
        similarity_metrics = {
            'config_name': [],
            'mean_similarity': [],
            'median_similarity': [],
            'min_similarity': [],
            'max_similarity': [],
            'std_deviation': [],
            'above_0.80_rate': [],
            'above_0.85_rate': [],
            'above_0.90_rate': [],
            'above_0.95_rate': []
        }
        
        for config_name, data in self.metrics_data.items():
            sim_data = data.get('utility_metrics', {}).get('semantic_similarity', {})
            threshold_data = data.get('privacy_metrics', {}).get('similarity_distribution', {}).get('threshold_counts', {})
            
            if sim_data:
                similarity_metrics['config_name'].append(config_name)
                similarity_metrics['mean_similarity'].append(sim_data.get('mean_similarity', 0))
                similarity_metrics['median_similarity'].append(sim_data.get('median_similarity', 0))
                similarity_metrics['min_similarity'].append(sim_data.get('min_similarity', 0))
                similarity_metrics['max_similarity'].append(sim_data.get('max_similarity', 0))
                similarity_metrics['std_deviation'].append(sim_data.get('std_deviation', 0))
                
                # Extract threshold data
                similarity_metrics['above_0.80_rate'].append(threshold_data.get('above_0.80', {}).get('rate', 0))
                similarity_metrics['above_0.85_rate'].append(threshold_data.get('above_0.85', {}).get('rate', 0))
                similarity_metrics['above_0.90_rate'].append(threshold_data.get('above_0.90', {}).get('rate', 0))
                similarity_metrics['above_0.95_rate'].append(threshold_data.get('above_0.95', {}).get('rate', 0))
        
        return pd.DataFrame(similarity_metrics)
    
    def plot_ngram_diversity(self, save_path=None):
        """
        Generate plots for n-gram diversity metrics.
        
        Args:
            save_path (str, optional): Path to save the plot file. If None, plot is displayed.
        """
        ngram_df = self.extract_ngram_diversity_metrics()
        
        # Prepare figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('N-gram Diversity Metrics Across Configurations', fontsize=16)
        
        # Plot 1: Unique ratio comparison
        sns.barplot(
            data=ngram_df, 
            x='n_gram', 
            y='unique_ratio_generated', 
            hue='config_name',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Unique Ratio (Generated Text)')
        axes[0, 0].set_ylabel('Unique Ratio')
        
        # Plot 2: Jensen-Shannon Divergence
        sns.barplot(
            data=ngram_df, 
            x='n_gram', 
            y='jensen_shannon_divergence', 
            hue='config_name',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Jensen-Shannon Divergence')
        axes[0, 1].set_ylabel('JS Divergence')
        
        # Plot 3: Original vs Generated Entropy
        # Reshape the data for side-by-side comparison
        entropy_data = []
        for _, row in ngram_df.iterrows():
            entropy_data.append({
                'config_name': row['config_name'],
                'n_gram': row['n_gram'],
                'entropy': row['original_entropy_normalized'],
                'type': 'Original'
            })
            entropy_data.append({
                'config_name': row['config_name'],
                'n_gram': row['n_gram'],
                'entropy': row['generated_entropy_normalized'],
                'type': 'Generated'
            })
        
        entropy_df = pd.DataFrame(entropy_data)
        sns.barplot(
            data=entropy_df,
            x='n_gram',
            y='entropy',
            hue='config_name',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Entropy Comparison')
        axes[1, 0].set_ylabel('Normalized Entropy')
        
        # Plot 4: Ratio comparison (Generated/Original)
        ratio_df = ngram_df.copy()
        ratio_df['unique_ratio_comparison'] = ratio_df['unique_ratio_generated'] / ratio_df['unique_ratio_original']
        sns.barplot(
            data=ratio_df,
            x='n_gram',
            y='unique_ratio_comparison',
            hue='config_name',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Unique Ratio Comparison (Generated/Original)')
        axes[1, 1].set_ylabel('Ratio')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            print(f"N-gram diversity plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_similarity_metrics(self, save_path=None):
        """
        Generate plots for semantic similarity metrics.
        
        Args:
            save_path (str, optional): Path to save the plot file. If None, plot is displayed.
        """
        sim_df = self.extract_similarity_metrics()
        
        # Prepare figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Semantic Similarity Metrics Across Configurations', fontsize=16)
        
        # Plot 1: Mean and median similarity
        sim_stats = []
        for _, row in sim_df.iterrows():
            sim_stats.append({
                'config_name': row['config_name'],
                'value': row['mean_similarity'],
                'metric': 'Mean'
            })
            sim_stats.append({
                'config_name': row['config_name'],
                'value': row['median_similarity'],
                'metric': 'Median'
            })
        
        stats_df = pd.DataFrame(sim_stats)
        sns.barplot(
            data=stats_df,
            x='config_name',
            y='value',
            hue='metric',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Mean and Median Similarity')
        axes[0, 0].set_ylabel('Similarity Score')
        axes[0, 0].set_xlabel('Configuration')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Min-Max Range
        for i, config in enumerate(sim_df['config_name']):
            axes[0, 1].bar(
                i, 
                sim_df.loc[i, 'max_similarity'] - sim_df.loc[i, 'min_similarity'],
                bottom=sim_df.loc[i, 'min_similarity'],
                label=config
            )
            # Add error bars for std deviation
            axes[0, 1].errorbar(
                i, 
                sim_df.loc[i, 'mean_similarity'],
                yerr=sim_df.loc[i, 'std_deviation'],
                fmt='ko',
                capsize=5
            )
            
        axes[0, 1].set_title('Similarity Range with Standard Deviation')
        axes[0, 1].set_ylabel('Similarity Score')
        axes[0, 1].set_xticks(range(len(sim_df)))
        axes[0, 1].set_xticklabels(sim_df['config_name'], rotation=45)
        
        # Plot 3: Threshold rates
        threshold_data = []
        thresholds = ['above_0.80_rate', 'above_0.85_rate', 'above_0.90_rate', 'above_0.95_rate']
        threshold_labels = ['> 0.80', '> 0.85', '> 0.90', '> 0.95']
        
        for _, row in sim_df.iterrows():
            for threshold, label in zip(thresholds, threshold_labels):
                threshold_data.append({
                    'config_name': row['config_name'],
                    'threshold': label,
                    'rate': row[threshold]
                })
        
        threshold_df = pd.DataFrame(threshold_data)
        sns.barplot(
            data=threshold_df,
            x='threshold',
            y='rate',
            hue='config_name',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Similarity Threshold Rates')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].set_xlabel('Similarity Threshold')
        
        # Plot 4: Similarity distribution (hypothetical)
        # For actual distribution, we'd need the raw similarity scores
        # Instead, we'll create a simplified view using mean and std
        x = np.linspace(-0.1, 1.1, 1000)
        for i, config in enumerate(sim_df['config_name']):
            mean = sim_df.loc[i, 'mean_similarity']
            std = sim_df.loc[i, 'std_deviation']
            # Create a normal distribution approximation
            y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            axes[1, 1].plot(x, y, label=config)
            
        axes[1, 1].set_title('Approximate Similarity Distribution')
        axes[1, 1].set_xlabel('Similarity Score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(-0.1, 1.1)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            print(f"Similarity metrics plot saved to {save_path}")
        else:
            plt.show()

    def save_metrics_to_csv(self, output_dir):
        """
        Save extracted metrics to CSV files.
        
        Args:
            output_dir (str): Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save n-gram diversity metrics
        ngram_df = self.extract_ngram_diversity_metrics()
        ngram_csv_path = os.path.join(output_dir, 'ngram_diversity_metrics.csv')
        ngram_df.to_csv(ngram_csv_path, index=False)
        print(f"N-gram diversity metrics saved to {ngram_csv_path}")
        
        # Save similarity metrics
        sim_df = self.extract_similarity_metrics()
        sim_csv_path = os.path.join(output_dir, 'similarity_metrics.csv')
        sim_df.to_csv(sim_csv_path, index=False)
        print(f"Similarity metrics saved to {sim_csv_path}")


# Example usage
if __name__ == "__main__":
    # Define path to directory containing JSON files
    json_dir = "results"  # Change this to your directory
    output_dir = "output"  # Change this to your desired output directory
    
    # Initialize and run analyzer
    analyzer = MetricsAnalyzer(json_dir)
    analyzer.load_json_files()
    
    # Generate plots
    analyzer.plot_ngram_diversity(os.path.join(output_dir, "ngram_diversity_plot.png"))
    analyzer.plot_similarity_metrics(os.path.join(output_dir, "similarity_metrics_plot.png"))
  
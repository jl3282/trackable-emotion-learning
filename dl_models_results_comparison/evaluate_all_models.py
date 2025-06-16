import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from pathlib import Path

def extract_config_from_filename(filename):
    """Extract hyperparameters from filename"""
    # Pattern: history_nf32_ks3_dr0.0_lr1e-03_bs128.csv
    pattern = r'history_nf(\d+)_ks(\d+)_dr([\d.]+)_lr([\de.-]+)_bs(\d+)(?:_hs(\d+))?\.csv'
    match = re.search(pattern, filename)
    
    if match:
        config = {
            'num_filters': int(match.group(1)),
            'kernel_size': int(match.group(2)),
            'dropout': float(match.group(3)),
            'lr': float(match.group(4)),
            'batch_size': int(match.group(5)),
            'filename': filename
        }
        # For hybrid models that have hidden_size
        if match.group(6):
            config['hidden_size'] = int(match.group(6))
        
        return config
    return None

def analyze_model_results(results_dir, model_name):
    """Analyze all model results and extract best validation accuracies"""
    history_files = glob.glob(f'{results_dir}/history_*.csv')
    
    results = []
    
    for file_path in history_files:
        filename = Path(file_path).name
        config = extract_config_from_filename(filename)
        
        if config:
            # Read the history file
            try:
                history = pd.read_csv(file_path)
                best_val_acc = history['val_acc'].max()
                epochs_trained = len(history)
                
                config.update({
                    'best_val_acc': best_val_acc,
                    'epochs_trained': epochs_trained,
                    'model_type': model_name
                })
                results.append(config)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return pd.DataFrame(results)

def load_hyperparameter_results(results_file, model_name):
    """Load results from existing hyperparameter_search_results.csv files"""
    try:
        df = pd.read_csv(results_file)
        df['model_type'] = model_name
        return df
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        return pd.DataFrame()

def create_model_comparison_visualizations(all_results):
    """Create comprehensive visualizations comparing all models"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Hyperparameter Trends Across All Model Architectures', fontsize=16, fontweight='bold')
    
    # 1. Model performance comparison
    sns.boxplot(data=all_results, x='model_type', y='best_val_acc', ax=axes[0,0])
    axes[0,0].set_title('Validation Accuracy by Model Type')
    axes[0,0].set_ylabel('Best Validation Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Number of filters across models
    sns.boxplot(data=all_results, x='model_type', y='num_filters', ax=axes[0,1])
    axes[0,1].set_title('Number of Filters by Model Type')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Kernel size trends
    sns.boxplot(data=all_results, x='kernel_size', y='best_val_acc', hue='model_type', ax=axes[0,2])
    axes[0,2].set_title('Validation Accuracy by Kernel Size')
    axes[0,2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Dropout trends
    sns.boxplot(data=all_results, x='dropout', y='best_val_acc', hue='model_type', ax=axes[1,0])
    axes[1,0].set_title('Validation Accuracy by Dropout Rate')
    axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Learning rate trends
    sns.boxplot(data=all_results, x='lr', y='best_val_acc', hue='model_type', ax=axes[1,1])
    axes[1,1].set_title('Validation Accuracy by Learning Rate')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Batch size trends
    sns.boxplot(data=all_results, x='batch_size', y='best_val_acc', hue='model_type', ax=axes[1,2])
    axes[1,2].set_title('Validation Accuracy by Batch Size')
    axes[1,2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 7. Best performance per model type
    model_best = all_results.groupby('model_type')['best_val_acc'].max().sort_values(ascending=False)
    model_best.plot(kind='bar', ax=axes[2,0], color='skyblue')
    axes[2,0].set_title('Best Validation Accuracy by Model Type')
    axes[2,0].set_ylabel('Best Validation Accuracy')
    axes[2,0].tick_params(axis='x', rotation=45)
    
    # 8. Average performance per model type
    model_avg = all_results.groupby('model_type')['best_val_acc'].mean().sort_values(ascending=False)
    model_avg.plot(kind='bar', ax=axes[2,1], color='lightcoral')
    axes[2,1].set_title('Average Validation Accuracy by Model Type')
    axes[2,1].set_ylabel('Average Validation Accuracy')
    axes[2,1].tick_params(axis='x', rotation=45)
    
    # 9. Training efficiency (epochs needed)
    sns.boxplot(data=all_results, x='model_type', y='epochs_trained', ax=axes[2,2])
    axes[2,2].set_title('Training Epochs by Model Type')
    axes[2,2].set_ylabel('Epochs Trained')
    axes[2,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('all_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_analysis(all_results):
    """Create detailed analysis across all models"""
    
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON ANALYSIS")
    print("="*80)
    
    # Overall best models
    top_overall = all_results.nlargest(15, 'best_val_acc')
    
    print("\n🏆 TOP 15 MODELS ACROSS ALL ARCHITECTURES:")
    print("-" * 80)
    for i, (_, row) in enumerate(top_overall.iterrows(), 1):
        model_info = f"{row['model_type']:15s}"
        params = f"nf{row['num_filters']:2d} ks{row['kernel_size']:1d} dr{row['dropout']:.1f} lr{row['lr']:.0e} bs{row['batch_size']:3d}"
        if 'hidden_size' in row and pd.notna(row['hidden_size']):
            params += f" hs{int(row['hidden_size']):3d}"
        
        print(f"{i:2d}. {model_info} | Val Acc: {row['best_val_acc']:.4f} | {params} | Epochs: {row['epochs_trained']:2d}")
    
    # Model-specific analysis
    print(f"\n📊 MODEL TYPE PERFORMANCE SUMMARY:")
    print("-" * 80)
    
    model_stats = all_results.groupby('model_type').agg({
        'best_val_acc': ['count', 'mean', 'std', 'max', 'min'],
        'epochs_trained': 'mean'
    }).round(4)
    
    model_stats.columns = ['Count', 'Mean_Val_Acc', 'Std_Val_Acc', 'Max_Val_Acc', 'Min_Val_Acc', 'Avg_Epochs']
    model_stats = model_stats.sort_values('Mean_Val_Acc', ascending=False)
    
    for model_type, row in model_stats.iterrows():
        print(f"\n{model_type.upper()}:")
        print(f"  • Experiments: {int(row['Count'])} | Avg Accuracy: {row['Mean_Val_Acc']:.4f} ± {row['Std_Val_Acc']:.4f}")
        print(f"  • Best: {row['Max_Val_Acc']:.4f} | Worst: {row['Min_Val_Acc']:.4f} | Avg Epochs: {row['Avg_Epochs']:.1f}")
    
    # Best hyperparameters per model type
    print(f"\n🎯 BEST CONFIGURATION PER MODEL TYPE:")
    print("-" * 80)
    
    for model_type in all_results['model_type'].unique():
        model_data = all_results[all_results['model_type'] == model_type]
        best_config = model_data.loc[model_data['best_val_acc'].idxmax()]
        
        params = f"nf{best_config['num_filters']:2d} ks{best_config['kernel_size']:1d} dr{best_config['dropout']:.1f} lr{best_config['lr']:.0e} bs{best_config['batch_size']:3d}"
        if 'hidden_size' in best_config and pd.notna(best_config['hidden_size']):
            params += f" hs{int(best_config['hidden_size']):3d}"
        
        print(f"{model_type:15s}: {best_config['best_val_acc']:.4f} | {params}")
    
    # Hyperparameter insights across all models
    print(f"\n🔍 HYPERPARAMETER INSIGHTS ACROSS ALL MODELS:")
    print("-" * 80)
    
    for param in ['num_filters', 'kernel_size', 'dropout', 'lr', 'batch_size']:
        if param in all_results.columns:
            param_performance = all_results.groupby(param)['best_val_acc'].agg(['mean', 'count']).round(4)
            param_performance = param_performance.sort_values('mean', ascending=False)
            print(f"\n{param.upper()} (best to worst):")
            for value, row in param_performance.head(5).iterrows():
                print(f"  {value}: {row['mean']:.4f} (n={int(row['count'])})")

def main():
    """Main analysis function"""
    print("Loading and analyzing all model architectures...")
    
    all_results = []
    
    # Define model directories and names
    model_configs = [
        ('tcn_results', 'TCN'),
        ('causal_tcn_results', 'Causal_TCN'), 
        ('deeper_results', 'Deeper_CNN'),
        ('hybrid_results', 'Hybrid_CNN_LSTM'),
        ('improved_tcn_results', 'Improved_TCN'),
        ('improved_spatial_tcn_results', 'Spatial_TCN')
    ]
    
    # Load results from each model type
    for results_dir, model_name in model_configs:
        print(f"Processing {model_name}...")
        
        # Try to load from hyperparameter_search_results.csv first
        results_file = f'{results_dir}/hyperparameter_search_results.csv'
        if Path(results_file).exists():
            df = load_hyperparameter_results(results_file, model_name)
            if not df.empty:
                all_results.append(df)
                print(f"  Loaded {len(df)} results from CSV")
                continue
        
        # If no CSV, try to analyze from history files
        if Path(results_dir).exists():
            df = analyze_model_results(results_dir, model_name)
            if not df.empty:
                all_results.append(df)
                print(f"  Analyzed {len(df)} history files")
            else:
                print(f"  No results found in {results_dir}")
        else:
            print(f"  Directory {results_dir} not found")
    
    if not all_results:
        print("No results found! Make sure you have trained models.")
        return
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    print(f"\n🎯 SUMMARY:")
    print(f"Total models analyzed: {len(combined_results)}")
    print(f"Model architectures: {combined_results['model_type'].unique()}")
    print(f"Overall best accuracy: {combined_results['best_val_acc'].max():.4f}")
    print(f"Overall accuracy range: {combined_results['best_val_acc'].min():.4f} - {combined_results['best_val_acc'].max():.4f}")
    
    # Create visualizations
    create_model_comparison_visualizations(combined_results)
    
    # Create detailed analysis
    create_detailed_analysis(combined_results)
    
    # Save combined results
    combined_results.to_csv('all_models_comparison.csv', index=False)
    print(f"\nResults saved to all_models_comparison.csv")

if __name__ == "__main__":
    main() 
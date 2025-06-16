import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from pathlib import Path
import pickle
import torch
from sklearn.metrics import accuracy_score, classification_report

class ModelComparison:
    """Compare performance of all models"""
    
    def __init__(self):
        self.results = {}
        self.model_types = {
            'deep_learning': ['Spatial TCN', 'CNN', 'Deep CNN', 'TCN', 'Improved TCN', 'Hybrid CNN-LSTM'],
            'traditional_ml': ['Random Forest', 'Logistic Regression']
        }
    
    def load_deep_learning_results(self):
        """Load results from deep learning experiments"""
        
        # Load production Spatial TCN results
        try:
            with open('production_models/final_results.json', 'r') as f:
                spatial_tcn_results = json.load(f)
            self.results['Spatial TCN'] = {
                'test_accuracy': spatial_tcn_results['test_accuracy'],
                'validation_accuracy': spatial_tcn_results['best_val_accuracy'],
                'type': 'deep_learning',
                'parameters': '28.5M'  # From your model
            }
            print("✅ Loaded Spatial TCN results")
        except FileNotFoundError:
            print("❌ Spatial TCN results not found")
        
        # Load hyperparameter search results from different experiments
        experiment_dirs = [
            ('results', 'CNN'),
            ('deeper_results', 'Deep CNN'),
            ('tcn_results', 'TCN'),
            ('improved_tcn_results', 'Improved TCN'),
            ('improved_spatial_tcn_results', 'Improved Spatial TCN'),
            ('causal_tcn_results', 'Causal TCN'),
            ('hybrid_results', 'Hybrid CNN-LSTM')
        ]
        
        for result_dir, model_name in experiment_dirs:
            try:
                result_file = f'{result_dir}/hyperparameter_search_results.csv'
                if Path(result_file).exists():
                    df = pd.read_csv(result_file)
                    best_result = df.loc[df['best_val_acc'].idxmax()]
                    
                    self.results[model_name] = {
                        'validation_accuracy': best_result['best_val_acc'],
                        'best_params': best_result.to_dict(),
                        'type': 'deep_learning',
                        'epochs_trained': best_result.get('epochs_trained', 'N/A')
                    }
                    print(f"✅ Loaded {model_name} results")
                else:
                    print(f"❌ {model_name} results not found at {result_file}")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
    
    def load_traditional_ml_results(self):
        """Load results from traditional ML experiments"""
        try:
            # Find the most recent results file
            result_files = glob.glob('traditional_ml_models/results_*.json')
            if result_files:
                latest_file = max(result_files, key=lambda x: Path(x).stat().st_mtime)
                
                with open(latest_file, 'r') as f:
                    traditional_results = json.load(f)
                
                for model_name, results in traditional_results.items():
                    self.results[model_name] = {
                        'test_accuracy': results['test_accuracy'],
                        'type': 'traditional_ml',
                        'classification_report': results['classification_report']
                    }
                    print(f"✅ Loaded {model_name} results")
            else:
                print("❌ Traditional ML results not found")
                
        except Exception as e:
            print(f"❌ Error loading traditional ML results: {e}")
    
    def create_comparison_table(self):
        """Create a comprehensive comparison table"""
        
        # Prepare data for table
        data = []
        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Type': results['type'].replace('_', ' ').title(),
                'Test Accuracy': results.get('test_accuracy', 'N/A'),
                'Validation Accuracy': results.get('validation_accuracy', 'N/A'),
                'Parameters': results.get('parameters', 'N/A'),
                'Epochs': results.get('epochs_trained', 'N/A')
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by test accuracy (if available), then by validation accuracy
        df['Test_Acc_Numeric'] = pd.to_numeric(df['Test Accuracy'], errors='coerce')
        df['Val_Acc_Numeric'] = pd.to_numeric(df['Validation Accuracy'], errors='coerce')
        
        df = df.sort_values(['Test_Acc_Numeric', 'Val_Acc_Numeric'], 
                           ascending=[False, False], na_position='last')
        
        # Format accuracies as percentages
        for col in ['Test Accuracy', 'Validation Accuracy']:
            df[col] = df[col].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else str(x))
        
        # Drop helper columns
        df = df.drop(['Test_Acc_Numeric', 'Val_Acc_Numeric'], axis=1)
        
        return df
    
    def plot_model_comparison(self, save_path='model_comparison.png'):
        """Create visualization comparing all models"""
        
        # Prepare data for plotting
        models = []
        test_accs = []
        val_accs = []
        model_types = []
        
        for model_name, results in self.results.items():
            models.append(model_name)
            test_accs.append(results.get('test_accuracy', np.nan))
            val_accs.append(results.get('validation_accuracy', np.nan))
            model_types.append(results['type'])
        
        # Create DataFrame
        plot_df = pd.DataFrame({
            'Model': models,
            'Test Accuracy': test_accs,
            'Validation Accuracy': val_accs,
            'Type': model_types
        })
        
        # Remove models without any accuracy data
        plot_df = plot_df.dropna(subset=['Test Accuracy', 'Validation Accuracy'], how='all')
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Color palette
        colors = {'deep_learning': 'skyblue', 'traditional_ml': 'lightcoral'}
        
        # Plot 1: Test Accuracy Comparison
        test_data = plot_df.dropna(subset=['Test Accuracy'])
        if not test_data.empty:
            bars1 = ax1.bar(range(len(test_data)), test_data['Test Accuracy'], 
                           color=[colors[t] for t in test_data['Type']])
            ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Test Accuracy', fontsize=12)
            ax1.set_xlabel('Models', fontsize=12)
            ax1.set_xticks(range(len(test_data)))
            ax1.set_xticklabels(test_data['Model'], rotation=45, ha='right')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 2: Validation Accuracy Comparison
        val_data = plot_df.dropna(subset=['Validation Accuracy'])
        if not val_data.empty:
            bars2 = ax2.bar(range(len(val_data)), val_data['Validation Accuracy'],
                           color=[colors[t] for t in val_data['Type']])
            ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Validation Accuracy', fontsize=12)
            ax2.set_xlabel('Models', fontsize=12)
            ax2.set_xticks(range(len(val_data)))
            ax2.set_xticklabels(val_data['Model'], rotation=45, ha='right')
            ax2.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='skyblue', label='Deep Learning'),
                          Patch(facecolor='lightcoral', label='Traditional ML')]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comparison plot saved: {save_path}")
    
    def analyze_performance_gaps(self):
        """Analyze performance gaps between model types"""
        
        # Separate models by type
        dl_models = {k: v for k, v in self.results.items() if v['type'] == 'deep_learning'}
        ml_models = {k: v for k, v in self.results.items() if v['type'] == 'traditional_ml'}
        
        print("\n🔍 PERFORMANCE ANALYSIS:")
        print("="*50)
        
        # Deep Learning Performance
        if dl_models:
            dl_test_accs = [v.get('test_accuracy') for v in dl_models.values() if v.get('test_accuracy') is not None]
            dl_val_accs = [v.get('validation_accuracy') for v in dl_models.values() if v.get('validation_accuracy') is not None]
            
            if dl_test_accs:
                print(f"Deep Learning Models:")
                print(f"  Test Accuracy - Max: {max(dl_test_accs):.4f}, Min: {min(dl_test_accs):.4f}, Avg: {np.mean(dl_test_accs):.4f}")
            
            if dl_val_accs:
                print(f"  Val Accuracy  - Max: {max(dl_val_accs):.4f}, Min: {min(dl_val_accs):.4f}, Avg: {np.mean(dl_val_accs):.4f}")
        
        # Traditional ML Performance
        if ml_models:
            ml_test_accs = [v.get('test_accuracy') for v in ml_models.values() if v.get('test_accuracy') is not None]
            
            if ml_test_accs:
                print(f"\nTraditional ML Models:")
                print(f"  Test Accuracy - Max: {max(ml_test_accs):.4f}, Min: {min(ml_test_accs):.4f}, Avg: {np.mean(ml_test_accs):.4f}")
        
        # Compare best performers
        all_test_accs = [(k, v.get('test_accuracy')) for k, v in self.results.items() 
                        if v.get('test_accuracy') is not None]
        
        if all_test_accs:
            all_test_accs.sort(key=lambda x: x[1], reverse=True)
            print(f"\n🏆 TOP PERFORMERS (Test Accuracy):")
            for i, (model, acc) in enumerate(all_test_accs[:5], 1):
                model_type = self.results[model]['type'].replace('_', ' ').title()
                print(f"  {i}. {model} ({model_type}): {acc:.4f}")
    
    def generate_recommendations(self):
        """Generate recommendations based on results"""
        
        print("\n💡 RECOMMENDATIONS:")
        print("="*50)
        
        # Find best performers
        all_models_with_test = [(k, v.get('test_accuracy', 0)) for k, v in self.results.items() 
                               if v.get('test_accuracy') is not None]
        
        if all_models_with_test:
            best_model = max(all_models_with_test, key=lambda x: x[1])
            best_name, best_acc = best_model
            best_type = self.results[best_name]['type']
            
            print(f"🥇 Best Overall Model: {best_name} ({best_acc:.4f} test accuracy)")
            
            if best_type == 'traditional_ml':
                print("\n📈 Key Insights:")
                print("• Traditional ML (Random Forest/Logistic Regression) outperformed deep learning")
                print("• This suggests that:")
                print("  - Your dataset may be too small for deep learning to show advantages")
                print("  - Hand-crafted features capture the relevant patterns well")
                print("  - Simpler models are less prone to overfitting on this dataset")
                print("\n🎯 Recommendations:")
                print("• Focus on traditional ML approaches for production")
                print("• Consider ensemble methods combining multiple traditional ML models")
                print("• Explore more sophisticated feature engineering")
                print("• For research: try collecting more data to benefit from deep learning")
                
            else:
                print("\n📈 Key Insights:")
                print("• Deep learning models successfully captured complex patterns")
                print("• The temporal/spatial nature of your data benefits from neural architectures")
                print("\n🎯 Recommendations:")
                print("• Consider ensemble methods combining your best deep learning models")
                print("• Explore model compression for deployment on wearable devices")
                print("• Implement transfer learning for new users/devices")
        
        # Model complexity vs performance
        spatial_tcn_acc = self.results.get('Spatial TCN', {}).get('test_accuracy')
        rf_acc = max([v.get('test_accuracy', 0) for k, v in self.results.items() 
                     if 'Random Forest' in k], default=0)
        
        if spatial_tcn_acc and rf_acc:
            performance_gap = abs(spatial_tcn_acc - rf_acc)
            print(f"\n⚖️  Complexity vs Performance:")
            print(f"• Spatial TCN: {spatial_tcn_acc:.4f} (28.5M parameters)")
            print(f"• Random Forest: {rf_acc:.4f} (~few KB)")
            print(f"• Performance gap: {performance_gap:.4f}")
            
            if performance_gap < 0.05:  # Less than 5% difference
                print("• → Consider Random Forest for production due to simplicity")
            else:
                print("• → Deep learning justified by performance gain")
    
    def save_comparison_report(self, filename='model_comparison_report.txt'):
        """Save detailed comparison report"""
        
        with open(filename, 'w') as f:
            f.write("EMOTION RECOGNITION MODEL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("OVERVIEW:\n")
            f.write(f"Total models compared: {len(self.results)}\n")
            f.write(f"Deep learning models: {len([v for v in self.results.values() if v['type'] == 'deep_learning'])}\n")
            f.write(f"Traditional ML models: {len([v for v in self.results.values() if v['type'] == 'traditional_ml'])}\n\n")
            
            f.write("DETAILED RESULTS:\n")
            for model_name, results in self.results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Type: {results['type']}\n")
                if 'test_accuracy' in results:
                    f.write(f"  Test Accuracy: {results['test_accuracy']:.4f}\n")
                if 'validation_accuracy' in results:
                    f.write(f"  Validation Accuracy: {results['validation_accuracy']:.4f}\n")
                if 'parameters' in results:
                    f.write(f"  Parameters: {results['parameters']}\n")
        
        print(f"📄 Detailed report saved: {filename}")
    
    def run_full_comparison(self):
        """Run complete model comparison analysis"""
        
        print("🔍 COMPREHENSIVE MODEL COMPARISON")
        print("="*50)
        
        # Load all results
        print("\n📂 Loading results...")
        self.load_deep_learning_results()
        self.load_traditional_ml_results()
        
        if not self.results:
            print("❌ No results found to compare!")
            return
        
        print(f"\n✅ Loaded {len(self.results)} models for comparison")
        
        # Create comparison table
        print("\n📊 Creating comparison table...")
        comparison_df = self.create_comparison_table()
        print("\nMODEL COMPARISON TABLE:")
        print(comparison_df.to_string(index=False))
        
        # Save table
        comparison_df.to_csv('model_comparison_table.csv', index=False)
        print("💾 Table saved: model_comparison_table.csv")
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        self.plot_model_comparison()
        
        # Performance analysis
        self.analyze_performance_gaps()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Save report
        self.save_comparison_report()
        
        print("\n✅ Model comparison completed!")
        
        return comparison_df

if __name__ == "__main__":
    # Run comprehensive model comparison
    comparison = ModelComparison()
    results_df = comparison.run_full_comparison() 
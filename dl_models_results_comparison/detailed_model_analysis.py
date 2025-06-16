import pandas as pd
import numpy as np

# Load the comprehensive results
df = pd.read_csv('all_models_comparison.csv')

print("="*80)
print("DETAILED MODEL ANALYSIS: PERFORMANCE vs EFFICIENCY")
print("="*80)

# Calculate efficiency metric (accuracy per epoch)
df['efficiency'] = df['best_val_acc'] / df['epochs_trained']

# Sort by accuracy for top performers
top_models = df.nlargest(20, 'best_val_acc')

print("\n🏆 TOP 20 MODELS BY ACCURACY (with efficiency analysis):")
print("-" * 80)
print(f"{'Rank':<4} {'Model':<15} {'Acc%':<6} {'Epochs':<7} {'Eff.':<6} {'Config'}")
print("-" * 80)

for i, (_, row) in enumerate(top_models.iterrows(), 1):
    config = f"nf{int(row['num_filters'])} ks{int(row['kernel_size'])} dr{row['dropout']} lr{row['lr']:.0e} bs{int(row['batch_size'])}"
    print(f"{i:<4} {row['model_type']:<15} {row['best_val_acc']*100:<6.1f} {row['epochs_trained']:<7.0f} {row['efficiency']*100:<6.2f} {config}")

print("\n" + "="*80)
print("EFFICIENCY ANALYSIS: BEST MODELS BY EPOCHS TRAINED")
print("="*80)

# Group by model type and analyze efficiency
model_efficiency = df.groupby('model_type').agg({
    'best_val_acc': ['mean', 'max', 'std'],
    'epochs_trained': ['mean', 'min', 'max'],
    'efficiency': ['mean', 'max']
}).round(4)

print("\nModel Type Performance Summary:")
print("-" * 60)
for model in df['model_type'].unique():
    model_data = df[df['model_type'] == model]
    avg_acc = model_data['best_val_acc'].mean()
    avg_epochs = model_data['epochs_trained'].mean()
    max_acc = model_data['best_val_acc'].max()
    max_acc_epochs = model_data.loc[model_data['best_val_acc'].idxmax(), 'epochs_trained']
    avg_efficiency = model_data['efficiency'].mean()
    
    print(f"{model:<15}: Avg Acc: {avg_acc:.3f} | Max Acc: {max_acc:.3f} ({max_acc_epochs:.0f} epochs)")
    print(f"                Avg Epochs: {avg_epochs:.1f} | Efficiency: {avg_efficiency:.4f}")
    print()

print("\n" + "="*80)
print("🎯 BEST MODELS BY EFFICIENCY (Accuracy/Epoch > 0.035)")
print("="*80)

efficient_models = df[df['efficiency'] > 0.035].sort_values('best_val_acc', ascending=False)
print(f"{'Model':<15} {'Acc%':<6} {'Epochs':<7} {'Efficiency':<10} {'Config'}")
print("-" * 70)

for _, row in efficient_models.head(15).iterrows():
    config = f"nf{int(row['num_filters'])} ks{int(row['kernel_size'])} dr{row['dropout']}"
    print(f"{row['model_type']:<15} {row['best_val_acc']*100:<6.1f} {row['epochs_trained']:<7.0f} {row['efficiency']:<10.4f} {config}")

print("\n" + "="*80)
print("📊 SECOND BEST MODEL ANALYSIS")
print("="*80)

# Find second best model overall
second_best = df.nlargest(2, 'best_val_acc').iloc[1]
print(f"Overall Second Best: {second_best['model_type']}")
print(f"  Accuracy: {second_best['best_val_acc']:.4f} ({second_best['best_val_acc']*100:.2f}%)")
print(f"  Epochs: {second_best['epochs_trained']}")
print(f"  Efficiency: {second_best['efficiency']:.4f}")
print(f"  Config: nf{int(second_best['num_filters'])} ks{int(second_best['kernel_size'])} dr{second_best['dropout']} lr{second_best['lr']:.0e} bs{int(second_best['batch_size'])}")

# Find best model per type excluding the top performer
print("\nBest Model per Architecture (excluding overall winner):")
print("-" * 60)

for model_type in df['model_type'].unique():
    model_data = df[df['model_type'] == model_type]
    if model_type == df.loc[df['best_val_acc'].idxmax(), 'model_type']:
        # For the winning architecture, get second best
        best_model = model_data.nlargest(2, 'best_val_acc').iloc[1] if len(model_data) > 1 else model_data.iloc[0]
        prefix = "2nd best"
    else:
        best_model = model_data.loc[model_data['best_val_acc'].idxmax()]
        prefix = "Best"
    
    config = f"nf{int(best_model['num_filters'])} ks{int(best_model['kernel_size'])} dr{best_model['dropout']}"
    print(f"{model_type:<15}: {prefix} = {best_model['best_val_acc']*100:.2f}% ({best_model['epochs_trained']:.0f} epochs) {config}")

print("\n" + "="*80)
print("🔍 FAIR COMPARISON: MODELS WITH <30 EPOCHS")
print("="*80)

fair_comparison = df[df['epochs_trained'] < 30].sort_values('best_val_acc', ascending=False)
print(f"{'Rank':<4} {'Model':<15} {'Acc%':<6} {'Epochs':<7} {'Config'}")
print("-" * 55)

for i, (_, row) in enumerate(fair_comparison.head(10).iterrows(), 1):
    config = f"nf{int(row['num_filters'])} ks{int(row['kernel_size'])} dr{row['dropout']}"
    print(f"{i:<4} {row['model_type']:<15} {row['best_val_acc']*100:<6.1f} {row['epochs_trained']:<7.0f} {config}")

print("\n" + "="*80)
print("💡 TRAINING EFFICIENCY INSIGHTS")
print("="*80)

# Compare Spatial TCN vs others on similar epoch counts
spatial_models = df[df['model_type'] == 'Spatial_TCN']
improved_models = df[df['model_type'] == 'Improved_TCN']

print("Spatial TCN Performance:")
for _, row in spatial_models.sort_values('best_val_acc', ascending=False).iterrows():
    print(f"  {row['best_val_acc']*100:.1f}% in {row['epochs_trained']:.0f} epochs (efficiency: {row['efficiency']:.4f})")

print("\nTop Improved TCN Performance (for comparison):")
for _, row in improved_models.nlargest(5, 'best_val_acc').iterrows():
    print(f"  {row['best_val_acc']*100:.1f}% in {row['epochs_trained']:.0f} epochs (efficiency: {row['efficiency']:.4f})") 
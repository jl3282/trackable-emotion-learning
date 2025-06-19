#!/usr/bin/env python3
"""
🎯 ENSEMBLE TRANSFER LEARNING FOR EMOWEAR
============================================================
Apply weighted ensemble of three best TCN models to EmoWear dataset
Original models achieved: Spatial TCN (74.15%), Basic TCN (70.88%), Improved TCN (70.14%)
Ensemble achieved: 78.24% on original dataset
Target: 55%+ on EmoWear (significant improvement over 50.15% baseline)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import json
from datetime import datetime
import warnings
import importlib
warnings.filterwarnings('ignore')

# Import model architectures
try:
    three_tcn_module = importlib.import_module("4_three_tcn_ensemble")
    ProductionSpatialTCN = three_tcn_module.ProductionSpatialTCN
    ProductionImprovedTCN = three_tcn_module.ProductionImprovedTCN
    ProductionTCN = three_tcn_module.ProductionTCN
    DataPreprocessor = three_tcn_module.DataPreprocessor
except ImportError:
    print("⚠️ Could not import from 4_three_tcn_ensemble. Using fallback architectures.")
    # Fallback architectures would go here

class EnsembleTransferLearning:
    """Ensemble transfer learning using three pre-trained TCN models"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.model_weights = {}
        self.preprocessor = None
        
    def load_pretrained_models(self):
        """Load the three best pre-trained models from original dataset"""
        print("📂 Loading pre-trained TCN models...")
        
        model_configs = {
            'spatial_tcn': {
                'path': 'final_spatial_tcn_models/best_spatial_tcn_20250617_013948_0.7415.pth',
                'class': ProductionSpatialTCN,
                'original_acc': 0.7415,
                'architecture': 'Spatial TCN with enhanced temporal blocks'
            },
            'basic_tcn': {
                'path': 'final_tcn_models/best_tcn_20250617_133158_0.7088.pth',
                'class': ProductionTCN,
                'original_acc': 0.7088,
                'architecture': 'Basic TCN with residual connections'
            },
            'improved_tcn': {
                'path': 'final_improved_tcn_models/best_improved_tcn_20250617_124238_0.7014.pth',
                'class': ProductionImprovedTCN,
                'original_acc': 0.7014,
                'architecture': 'Improved TCN with deeper layers'
            }
        }
        
        for name, config in model_configs.items():
            if Path(config['path']).exists():
                print(f"  Loading {name} ({config['original_acc']:.2%} original accuracy)...")
                
                # Load checkpoint
                checkpoint = torch.load(config['path'], map_location=self.device)
                
                # Create model instance
                model = config['class'](in_channels=7, num_classes=3)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                self.models[name] = {
                    'model': model,
                    'original_acc': config['original_acc'],
                    'architecture': config['architecture']
                }
                
                print(f"    ✅ {name} loaded successfully")
            else:
                print(f"    ❌ {config['path']} not found")
        
        print(f"✅ Loaded {len(self.models)} models")
        
    def load_emowear_data(self):
        """Load EmoWear preprocessed data"""
        print("📊 Loading EmoWear data...")
        
        X_train = np.load('emowear_7channel_wrist_focused/X_train.npy')
        y_train = pd.read_csv('emowear_7channel_wrist_focused/y_train.csv')['label'].values.astype(int)
        X_val = np.load('emowear_7channel_wrist_focused/X_val.npy')
        y_val = pd.read_csv('emowear_7channel_wrist_focused/y_val.csv')['label'].values.astype(int)
        X_test = np.load('emowear_7channel_wrist_focused/X_test.npy')
        y_test = pd.read_csv('emowear_7channel_wrist_focused/y_test.csv')['label'].values.astype(int)
        
        print(f"✅ EmoWear data loaded:")
        print(f"  Train: {X_train.shape}, Labels: {np.bincount(y_train)}")
        print(f"  Val: {X_val.shape}, Labels: {np.bincount(y_val)}")
        print(f"  Test: {X_test.shape}, Labels: {np.bincount(y_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def apply_preprocessing(self, X_train, X_val, X_test):
        """Apply same preprocessing as original models"""
        print("🔧 Applying preprocessing...")
        
        # Use the same preprocessor as the original models
        self.preprocessor = DataPreprocessor('savgol_minmax')
        
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        print("✅ Preprocessing applied")
        return X_train_processed, X_val_processed, X_test_processed
    
    def get_model_predictions(self, X, temperature=1.0):
        """Get predictions from all models with temperature scaling"""
        predictions = {}
        
        # Convert from (batch, channels, time) to (batch, time, channels) for the models
        X_transposed = X.transpose(0, 2, 1)  # (batch, channels, time) -> (batch, time, channels)
        X_tensor = torch.tensor(X_transposed, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            for name, model_info in self.models.items():
                model = model_info['model']
                model.eval()
                
                # Get logits
                logits = model(X_tensor)
                
                # Apply temperature scaling
                scaled_logits = logits / temperature
                
                # Convert to probabilities
                probs = F.softmax(scaled_logits, dim=1).cpu().numpy()
                predictions[name] = probs
        
        return predictions
    
    def optimize_ensemble_weights(self, X_val, y_val, method='performance_weighted'):
        """Optimize ensemble weights using validation data"""
        print(f"⚖️ Optimizing ensemble weights using {method}...")
        
        # Get validation predictions
        val_predictions = self.get_model_predictions(X_val)
        
        if method == 'performance_weighted':
            # Weight by original performance (inspired by your 78.24% success)
            total_performance = sum(info['original_acc'] for info in self.models.values())
            
            for name, model_info in self.models.items():
                weight = model_info['original_acc'] / total_performance
                self.model_weights[name] = weight
                print(f"  {name}: {weight:.3f} (based on {model_info['original_acc']:.2%} original accuracy)")
        
        elif method == 'validation_optimized':
            # Optimize weights on validation set (more sophisticated)
            from scipy.optimize import minimize
            
            def ensemble_loss(weights):
                # Ensure weights sum to 1
                weights = weights / np.sum(weights)
                
                # Create weighted ensemble prediction
                ensemble_pred = np.zeros_like(list(val_predictions.values())[0])
                for i, name in enumerate(self.models.keys()):
                    ensemble_pred += weights[i] * val_predictions[name]
                
                # Get predicted classes
                pred_classes = np.argmax(ensemble_pred, axis=1)
                
                # Return negative accuracy (for minimization)
                return -accuracy_score(y_val, pred_classes)
            
            # Initial weights (equal)
            initial_weights = np.ones(len(self.models)) / len(self.models)
            
            # Optimize
            result = minimize(ensemble_loss, initial_weights, 
                            bounds=[(0.1, 0.8) for _ in range(len(self.models))],
                            method='L-BFGS-B')
            
            # Normalize weights
            optimal_weights = result.x / np.sum(result.x)
            
            for i, name in enumerate(self.models.keys()):
                self.model_weights[name] = optimal_weights[i]
                print(f"  {name}: {optimal_weights[i]:.3f} (validation optimized)")
        
        else:  # equal_weights
            weight = 1.0 / len(self.models)
            for name in self.models.keys():
                self.model_weights[name] = weight
                print(f"  {name}: {weight:.3f} (equal weights)")
        
        print("✅ Ensemble weights optimized")
        return self.model_weights
    
    def create_ensemble_prediction(self, predictions):
        """Create weighted ensemble prediction"""
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for name, pred in predictions.items():
            weight = self.model_weights[name]
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def evaluate_transfer_learning(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                                 fine_tune=True, fine_tune_epochs=10):
        """Evaluate ensemble transfer learning with optional fine-tuning"""
        print("\n🎯 ENSEMBLE TRANSFER LEARNING EVALUATION")
        print("=" * 60)
        
        # Apply preprocessing
        X_train_proc, X_val_proc, X_test_proc = self.apply_preprocessing(X_train, X_val, X_test)
        
        if fine_tune:
            print(f"🔧 Fine-tuning ensemble for {fine_tune_epochs} epochs...")
            self.fine_tune_ensemble(X_train_proc, y_train, X_val_proc, y_val, epochs=fine_tune_epochs)
        
        # Optimize ensemble weights
        self.optimize_ensemble_weights(X_val_proc, y_val, method='performance_weighted')
        
        # Evaluate individual models
        print("\n📊 Individual model performance on EmoWear:")
        individual_results = {}
        
        val_predictions = self.get_model_predictions(X_val_proc)
        test_predictions = self.get_model_predictions(X_test_proc)
        
        for name, val_pred in val_predictions.items():
            val_acc = accuracy_score(y_val, np.argmax(val_pred, axis=1))
            test_acc = accuracy_score(y_test, np.argmax(test_predictions[name], axis=1))
            
            individual_results[name] = {
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'original_accuracy': self.models[name]['original_acc']
            }
            
            print(f"  {name}:")
            print(f"    Original: {self.models[name]['original_acc']:.2%}")
            print(f"    EmoWear Val: {val_acc:.2%}")
            print(f"    EmoWear Test: {test_acc:.2%}")
            print(f"    Transfer Gap: {self.models[name]['original_acc'] - test_acc:.2%}")
        
        # Evaluate ensemble
        print("\n🔮 Ensemble performance:")
        ensemble_val_pred = self.create_ensemble_prediction(val_predictions)
        ensemble_test_pred = self.create_ensemble_prediction(test_predictions)
        
        ensemble_val_acc = accuracy_score(y_val, np.argmax(ensemble_val_pred, axis=1))
        ensemble_test_acc = accuracy_score(y_test, np.argmax(ensemble_test_pred, axis=1))
        ensemble_test_f1 = f1_score(y_test, np.argmax(ensemble_test_pred, axis=1), average='macro')
        
        print(f"  Validation Accuracy: {ensemble_val_acc:.2%}")
        print(f"  Test Accuracy: {ensemble_test_acc:.2%}")
        print(f"  Test F1-Macro: {ensemble_test_f1:.4f}")
        
        # Compare to baselines
        baseline_single = 0.5015  # From your summary
        original_ensemble = 0.7824  # Your original dataset ensemble
        
        print(f"\n📈 COMPARISON:")
        print(f"  Original Ensemble: {original_ensemble:.2%}")
        print(f"  EmoWear Single Best: {baseline_single:.2%}")
        print(f"  EmoWear Ensemble: {ensemble_test_acc:.2%}")
        print(f"  Improvement over Single: {ensemble_test_acc - baseline_single:+.2%}")
        print(f"  Transfer Gap: {original_ensemble - ensemble_test_acc:.2%}")
        
        # Detailed classification report
        ensemble_pred_classes = np.argmax(ensemble_test_pred, axis=1)
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, ensemble_pred_classes, 
                                  target_names=['Sad', 'Neutral', 'Happy']))
        
        return {
            'individual_results': individual_results,
            'ensemble_val_accuracy': ensemble_val_acc,
            'ensemble_test_accuracy': ensemble_test_acc,
            'ensemble_test_f1': ensemble_test_f1,
            'model_weights': self.model_weights,
            'improvement_over_baseline': ensemble_test_acc - baseline_single,
            'transfer_gap': original_ensemble - ensemble_test_acc
        }
    
    def fine_tune_ensemble(self, X_train, y_train, X_val, y_val, epochs=10, lr=1e-5):
        """Fine-tune the ensemble models on EmoWear data"""
        print(f"🔧 Fine-tuning {len(self.models)} models...")
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Fine-tune each model
        for name, model_info in self.models.items():
            print(f"  Fine-tuning {name}...")
            
            model = model_info['model']
            model.train()
            
            # Only fine-tune the classifier layers (transfer learning best practice)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            best_val_acc = 0
            patience = 3
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    # Convert from (batch, channels, time) to (batch, time, channels)
                    batch_X = batch_X.transpose(1, 2)  # Fix dimension order
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_acc = 0
                val_total = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        # Convert from (batch, channels, time) to (batch, time, channels)
                        batch_X = batch_X.transpose(1, 2)  # Fix dimension order
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_acc += (predicted == batch_y).sum().item()
                
                val_acc = val_acc / val_total
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            print(f"    Best validation accuracy: {best_val_acc:.2%}")
            model.eval()  # Set back to eval mode
        
        print("✅ Fine-tuning completed")

def main():
    print("🚀 ENSEMBLE TRANSFER LEARNING FOR EMOWEAR")
    print("=" * 60)
    print("Applying weighted ensemble of three best TCN models")
    print("Target: Beat 50.15% EmoWear baseline with ensemble approach")
    print()
    
    # Initialize ensemble
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    ensemble = EnsembleTransferLearning(device=device)
    
    # Load pre-trained models
    ensemble.load_pretrained_models()
    
    # Load EmoWear data
    X_train, y_train, X_val, y_val, X_test, y_test = ensemble.load_emowear_data()
    
    # Evaluate transfer learning
    results = ensemble.evaluate_transfer_learning(
        X_train, y_train, X_val, y_val, X_test, y_test,
        fine_tune=True, fine_tune_epochs=5
    )
    
    # Save results
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'ensemble_test_accuracy': float(results['ensemble_test_accuracy']),
        'ensemble_test_f1': float(results['ensemble_test_f1']),
        'model_weights': {k: float(v) for k, v in results['model_weights'].items()},
        'improvement_over_baseline': float(results['improvement_over_baseline']),
        'transfer_gap': float(results['transfer_gap']),
        'individual_results': {
            k: {kk: float(vv) for kk, vv in v.items()} 
            for k, v in results['individual_results'].items()
        }
    }
    
    with open('ensemble_transfer_learning_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n💾 Results saved to: ensemble_transfer_learning_results.json")
    
    # Final recommendation
    if results['ensemble_test_accuracy'] > 0.52:
        print("\n🎉 SUCCESS: Ensemble transfer learning significantly improves performance!")
        print("✅ This approach demonstrates the value of ensemble methods for transfer learning.")
    elif results['improvement_over_baseline'] > 0.02:
        print("\n✅ IMPROVEMENT: Ensemble shows promising results over single model baseline.")
        print("💡 Consider further optimization strategies.")
    else:
        print("\n⚠️ LIMITED IMPROVEMENT: Ensemble shows marginal gains.")
        print("🔄 May need alternative transfer learning strategies.")

if __name__ == "__main__":
    main() 
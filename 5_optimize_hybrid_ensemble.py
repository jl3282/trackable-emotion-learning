import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import glob
from pathlib import Path
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

from three_tcn_ensemble import ProductionSpatialTCN, ProductionImprovedTCN, ProductionTCN, DataPreprocessor

class OptimizedHybridEnsemble:
    """Optimized hybrid ensemble with better feature engineering"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.feature_models = {}
        
    def extract_enhanced_features(self, X):
        """Enhanced feature extraction with more sophisticated features"""
        features = []
        
        for i in range(X.shape[0]):
            sample_features = []
            
            # Per-channel features
            for channel in range(X.shape[2]):
                signal_data = X[i, :, channel]
                
                # Basic statistical features
                sample_features.extend([
                    np.mean(signal_data),
                    np.std(signal_data),
                    np.var(signal_data),
                    np.min(signal_data),
                    np.max(signal_data),
                    np.median(signal_data),
                    np.percentile(signal_data, 25),
                    np.percentile(signal_data, 75),
                    signal_data.max() - signal_data.min(),
                    np.mean(np.abs(np.diff(signal_data))),
                    np.std(np.diff(signal_data)),
                ])
                
                # Rolling statistics (temporal features)
                window_size = min(5, len(signal_data)//2)
                if window_size > 1:
                    rolling_mean = np.convolve(signal_data, np.ones(window_size)/window_size, mode='valid')
                    rolling_std = np.array([np.std(signal_data[j:j+window_size]) for j in range(len(signal_data)-window_size+1)])
                    
                    sample_features.extend([
                        np.mean(rolling_mean),
                        np.std(rolling_mean),
                        np.mean(rolling_std),
                        np.std(rolling_std),
                    ])
                else:
                    sample_features.extend([0, 0, 0, 0])
                
                # Frequency domain features
                try:
                    fft = np.fft.fft(signal_data)
                    freqs = np.fft.fftfreq(len(signal_data), 1/50)
                    psd = np.abs(fft) ** 2
                    
                    # More detailed frequency bands
                    very_low = np.sum(psd[(freqs >= 0.1) & (freqs <= 1)])
                    low_freq = np.sum(psd[(freqs >= 1) & (freqs <= 4)])
                    mid_freq = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
                    high_freq = np.sum(psd[(freqs >= 8) & (freqs <= 15)])
                    very_high = np.sum(psd[(freqs >= 15) & (freqs <= 25)])
                    total_power = np.sum(psd[freqs >= 0.1])
                    
                    if total_power > 0:
                        sample_features.extend([
                            very_low / total_power,
                            low_freq / total_power,
                            mid_freq / total_power,
                            high_freq / total_power,
                            very_high / total_power,
                            np.argmax(psd[freqs >= 0]) if len(psd[freqs >= 0]) > 0 else 0,  # Dominant frequency
                        ])
                    else:
                        sample_features.extend([0, 0, 0, 0, 0, 0])
                        
                except:
                    sample_features.extend([0, 0, 0, 0, 0, 0])
            
            # Cross-channel features
            try:
                # Correlation between channels
                for ch1 in range(X.shape[2]):
                    for ch2 in range(ch1+1, X.shape[2]):
                        corr = np.corrcoef(X[i, :, ch1], X[i, :, ch2])[0, 1]
                        sample_features.append(corr if not np.isnan(corr) else 0)
                
                # Global statistics across all channels
                all_channels = X[i, :, :].flatten()
                sample_features.extend([
                    np.mean(all_channels),
                    np.std(all_channels),
                    np.var(all_channels),
                ])
                
            except:
                # Add zeros if cross-channel features fail
                n_cross_channel = (X.shape[2] * (X.shape[2] - 1)) // 2 + 3
                sample_features.extend([0] * n_cross_channel)
            
            features.append(sample_features)
        
        return np.array(features)
    
    def load_models(self):
        """Load TCN models"""
        model_paths = {
            'spatial': 'final_spatial_tcn_models/best_spatial_tcn_20250617_013948_0.7415.pth',
            'improved': 'final_improved_tcn_models/best_improved_tcn_20250617_124238_0.7014.pth',
            'basic': 'final_tcn_models/best_tcn_20250617_133158_0.7088.pth'
        }
        
        for model_name, path in model_paths.items():
            if Path(path).exists():
                checkpoint = torch.load(path, map_location=self.device)
                
                if model_name == 'spatial':
                    model = ProductionSpatialTCN()
                elif model_name == 'improved':
                    model = ProductionImprovedTCN()
                else:
                    model = ProductionTCN()
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                self.models[model_name] = model
    
    def optimize_weights(self, tcn_probs, feature_probs, y_val):
        """Optimize ensemble weights using validation set"""
        best_acc = 0
        best_weights = (0.7, 0.3)
        
        # Test different weight combinations
        weight_combinations = [
            (0.9, 0.1), (0.85, 0.15), (0.8, 0.2), (0.75, 0.25),
            (0.7, 0.3), (0.65, 0.35), (0.6, 0.4), (0.55, 0.45)
        ]
        
        for tcn_weight, feature_weight in weight_combinations:
            ensemble_probs = tcn_weight * tcn_probs + feature_weight * feature_probs
            preds = np.argmax(ensemble_probs, axis=1)
            acc = accuracy_score(y_val, preds)
            
            if acc > best_acc:
                best_acc = acc
                best_weights = (tcn_weight, feature_weight)
        
        return best_weights, best_acc
    
    def run_optimization(self):
        """Run complete optimization"""
        print("🚀 OPTIMIZED HYBRID ENSEMBLE")
        print("="*50)
        
        # Load and preprocess data
        print("\n📊 Loading data...")
        x_files = sorted(glob.glob('deep_learning_data/*_x.npy'))
        y_files = sorted(glob.glob('deep_learning_data/*_y.csv'))
        
        all_X, all_y = [], []
        for xf, yf in zip(x_files, y_files):
            X = np.load(xf)
            y = pd.read_csv(yf).values.squeeze()
            n = min(len(X), len(y))
            all_X.append(X[:n])
            all_y.append(y[:n])
        
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        y_all = np.where(y_all==-1, 0, np.where(y_all==0, 1, 2))
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
        )
        
        # Preprocess
        preprocessor = DataPreprocessor('savgol_minmax')
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # Extract enhanced features
        print("🔧 Extracting enhanced features...")
        X_train_features = self.extract_enhanced_features(X_train_processed)
        X_val_features = self.extract_enhanced_features(X_val_processed)
        X_test_features = self.extract_enhanced_features(X_test_processed)
        print(f"Extracted {X_train_features.shape[1]} enhanced features")
        
        # Load models
        self.load_models()
        
        # Train enhanced feature models
        print("🤖 Training enhanced feature models...")
        
        # Enhanced Random Forest
        rf = RandomForestClassifier(
            n_estimators=300,  # More trees
            max_depth=20,      # Deeper trees
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_features, y_train)
        
        # Enhanced Logistic Regression with feature selection
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.pipeline import Pipeline
        
        lr_pipeline = Pipeline([
            ('feature_selection', SelectKBest(f_classif, k=min(100, X_train_features.shape[1]))),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=0.5,  # More regularization
                max_iter=2000,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        lr_pipeline.fit(X_train_features, y_train)
        
        # Get validation predictions for weight optimization
        print("⚖️  Optimizing ensemble weights...")
        
        # TCN predictions on validation set
        tcn_val_preds = {}
        for model_name, model in self.models.items():
            X_tensor = torch.tensor(X_val_processed, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                pred = F.softmax(model(X_tensor), dim=1).cpu().numpy()
            tcn_val_preds[model_name] = pred
        
        # TCN ensemble on validation
        spatial_weight, improved_weight, basic_weight = 0.341, 0.328, 0.331
        tcn_val_ensemble = (spatial_weight * tcn_val_preds['spatial'] + 
                           improved_weight * tcn_val_preds['improved'] +
                           basic_weight * tcn_val_preds['basic'])
        
        # Feature predictions on validation
        rf_val_probs = rf.predict_proba(X_val_features)
        lr_val_probs = lr_pipeline.predict_proba(X_val_features)
        feature_val_ensemble = 0.6 * rf_val_probs + 0.4 * lr_val_probs
        
        # Optimize weights
        best_weights, best_val_acc = self.optimize_weights(
            tcn_val_ensemble, feature_val_ensemble, y_val
        )
        
        print(f"   Best weights: TCN={best_weights[0]:.2f}, Features={best_weights[1]:.2f}")
        print(f"   Validation accuracy: {best_val_acc:.4f}")
        
        # Test on final test set
        print("\n🎯 Final evaluation on test set...")
        
        # TCN predictions on test
        tcn_test_preds = {}
        for model_name, model in self.models.items():
            X_tensor = torch.tensor(X_test_processed, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                pred = F.softmax(model(X_tensor), dim=1).cpu().numpy()
            tcn_test_preds[model_name] = pred
        
        tcn_test_ensemble = (spatial_weight * tcn_test_preds['spatial'] + 
                            improved_weight * tcn_test_preds['improved'] +
                            basic_weight * tcn_test_preds['basic'])
        
        # Feature predictions on test
        rf_test_probs = rf.predict_proba(X_test_features)
        lr_test_probs = lr_pipeline.predict_proba(X_test_features)
        feature_test_ensemble = 0.6 * rf_test_probs + 0.4 * lr_test_probs
        
        # Final optimized ensemble
        optimized_ensemble = (best_weights[0] * tcn_test_ensemble + 
                             best_weights[1] * feature_test_ensemble)
        optimized_preds = np.argmax(optimized_ensemble, axis=1)
        optimized_acc = accuracy_score(y_test, optimized_preds)
        
        # Baseline comparison
        baseline_preds = np.argmax(tcn_test_ensemble, axis=1)
        baseline_acc = accuracy_score(y_test, baseline_preds)
        
        print(f"\n📈 FINAL RESULTS:")
        print(f"   Baseline TCN Ensemble: {baseline_acc:.4f}")
        print(f"   Optimized Hybrid: {optimized_acc:.4f}")
        print(f"   Improvement: {optimized_acc - baseline_acc:+.4f}")
        
        if optimized_acc > baseline_acc:
            print(f"   🎉 Successfully improved performance!")
        
        # Classification report
        from sklearn.metrics import classification_report
        print(f"\n📋 Optimized Ensemble Classification Report:")
        report = classification_report(y_test, optimized_preds, 
                                     target_names=['Sad', 'Neutral', 'Happy'])
        print(report)
        
        return {
            'baseline_accuracy': baseline_acc,
            'optimized_accuracy': optimized_acc,
            'improvement': optimized_acc - baseline_acc,
            'best_weights': best_weights
        }

if __name__ == "__main__":
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Run optimization
    optimizer = OptimizedHybridEnsemble(device)
    results = optimizer.run_optimization() 
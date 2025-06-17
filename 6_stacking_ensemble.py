import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import glob
from pathlib import Path
import json
import warnings
import importlib
warnings.filterwarnings('ignore')

# Import models from the file starting with a number
three_tcn_module = importlib.import_module("4_three_tcn_ensemble")
ProductionSpatialTCN = three_tcn_module.ProductionSpatialTCN
ProductionImprovedTCN = three_tcn_module.ProductionImprovedTCN
ProductionTCN = three_tcn_module.ProductionTCN
DataPreprocessor = three_tcn_module.DataPreprocessor

class StackingEnsemble:
    """Stacking ensemble using a meta-learner to combine base model predictions."""

    def __init__(self, device='cpu'):
        self.device = device
        self.tcn_models = {}
        self.feature_models = {}
        self.meta_learner = None

    def extract_enhanced_features(self, X):
        """Extracts the same 171 enhanced features as the previous best model."""
        features = []
        for i in range(X.shape[0]):
            sample_features = []
            for channel in range(X.shape[2]):
                signal_data = X[i, :, channel]
                sample_features.extend([
                    np.mean(signal_data), np.std(signal_data), np.var(signal_data),
                    np.min(signal_data), np.max(signal_data), np.median(signal_data),
                    np.percentile(signal_data, 25), np.percentile(signal_data, 75),
                    signal_data.max() - signal_data.min(),
                    np.mean(np.abs(np.diff(signal_data))),
                    np.std(np.diff(signal_data)),
                ])
                window_size = min(5, len(signal_data)//2)
                if window_size > 1:
                    rolling_mean = np.convolve(signal_data, np.ones(window_size)/window_size, mode='valid')
                    rolling_std = np.array([np.std(signal_data[j:j+window_size]) for j in range(len(signal_data)-window_size+1)])
                    sample_features.extend([
                        np.mean(rolling_mean), np.std(rolling_mean),
                        np.mean(rolling_std), np.std(rolling_std),
                    ])
                else: sample_features.extend([0, 0, 0, 0])
                try:
                    fft = np.fft.fft(signal_data)
                    freqs = np.fft.fftfreq(len(signal_data), 1/50)
                    psd = np.abs(fft) ** 2
                    bands = [(0.1, 1), (1, 4), (4, 8), (8, 15), (15, 25)]
                    total_power = np.sum(psd[freqs >= 0.1])
                    if total_power > 0:
                        for start, end in bands:
                            power = np.sum(psd[(freqs >= start) & (freqs <= end)])
                            sample_features.append(power / total_power)
                        sample_features.append(np.argmax(psd[freqs >= 0]) if len(psd[freqs >= 0]) > 0 else 0)
                    else: sample_features.extend([0] * (len(bands) + 1))
                except: sample_features.extend([0] * (len(bands) + 1))
            try:
                for ch1 in range(X.shape[2]):
                    for ch2 in range(ch1+1, X.shape[2]):
                        corr = np.corrcoef(X[i, :, ch1], X[i, :, ch2])[0, 1]
                        sample_features.append(corr if not np.isnan(corr) else 0)
                all_channels = X[i, :, :].flatten()
                sample_features.extend([np.mean(all_channels), np.std(all_channels), np.var(all_channels)])
            except:
                n_cross_channel = (X.shape[2] * (X.shape[2] - 1)) // 2 + 3
                sample_features.extend([0] * n_cross_channel)
            features.append(sample_features)
        return np.array(features)

    def load_tcn_models(self):
        """Loads the three pre-trained TCN models."""
        print("📂 Loading TCN models...")
        model_paths = {
            'spatial': 'final_spatial_tcn_models/best_spatial_tcn_20250617_013948_0.7415.pth',
            'improved': 'final_improved_tcn_models/best_improved_tcn_20250617_124238_0.7014.pth',
            'basic': 'final_tcn_models/best_tcn_20250617_133158_0.7088.pth'
        }
        model_classes = {'spatial': ProductionSpatialTCN, 'improved': ProductionImprovedTCN, 'basic': ProductionTCN}
        for name, path in model_paths.items():
            if Path(path).exists():
                checkpoint = torch.load(path, map_location=self.device)
                model = model_classes[name]()
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                self.tcn_models[name] = model
                print(f"   ✅ {name.capitalize()} TCN loaded")

    def get_base_model_predictions(self, X_processed, X_features):
        """Gets predictions from all base models to use as input for the meta-learner."""
        all_preds = []
        # TCN predictions
        with torch.no_grad():
            for model in self.tcn_models.values():
                X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)
                probs = F.softmax(model(X_tensor), dim=1).cpu().numpy()
                all_preds.append(probs)
        # Feature model predictions
        for model in self.feature_models.values():
            probs = model.predict_proba(X_features)
            all_preds.append(probs)
        return np.concatenate(all_preds, axis=1)

    def run(self):
        """Executes the full stacking pipeline."""
        print("🚀 Stacking Ensemble Pipeline")
        print("="*60)

        # 1. Load and split data
        print("\n📊 Loading and splitting data...")
        x_files = sorted(glob.glob('deep_learning_data/*_x.npy'))
        y_files = sorted(glob.glob('deep_learning_data/*_y.csv'))
        all_X, all_y = [], []
        for xf, yf in zip(x_files, y_files):
            X = np.load(xf)
            y_raw = pd.read_csv(yf).values.squeeze()
            # Map labels here
            y = np.where(y_raw==-1, 0, np.where(y_raw==0, 1, 2))
            n = min(len(X), len(y))
            all_X.append(X[:n])
            all_y.append(y[:n])
        
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.15, stratify=y_all, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # 2. Preprocess data and extract features
        preprocessor = DataPreprocessor('savgol_minmax')
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        print("\n🔧 Extracting features...")
        X_train_features = self.extract_enhanced_features(X_train_processed)
        X_val_features = self.extract_enhanced_features(X_val_processed)
        X_test_features = self.extract_enhanced_features(X_test_processed)

        # 3. Load TCN models and train feature-based models
        self.load_tcn_models()
        print("\n🤖 Training feature-based models...")
        rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3, random_state=42, n_jobs=-1)
        rf.fit(X_train_features, y_train)
        self.feature_models['rf'] = rf
        lr_pipeline = Pipeline([
            ('fs', SelectKBest(f_classif, k=min(100, X_train_features.shape[1]))),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.5, max_iter=2000, random_state=42, class_weight='balanced'))
        ])
        lr_pipeline.fit(X_train_features, y_train)
        self.feature_models['lr'] = lr_pipeline
        print("   ✅ Base models trained")

        # 4. Train meta-learner on validation set predictions
        print("\n🧠 Training meta-learner...")
        meta_features_val = self.get_base_model_predictions(X_val_processed, X_val_features)
        
        # Using a simple but powerful Logistic Regression as the meta-learner
        self.meta_learner = LogisticRegression(C=1.0, random_state=42, solver='lbfgs', max_iter=1000)
        self.meta_learner.fit(meta_features_val, y_val)
        print("   ✅ Meta-learner trained")

        # 5. Evaluate on test set
        print("\n🎯 Evaluating on test set...")
        meta_features_test = self.get_base_model_predictions(X_test_processed, X_test_features)
        stacking_preds = self.meta_learner.predict(meta_features_test)
        stacking_acc = accuracy_score(y_test, stacking_preds)

        # 6. Compare to previous best model
        baseline_acc = 0.7703 # From 5_optimize_hybrid_ensemble.py
        
        print("\n📈 FINAL RESULTS:")
        print(f"   Previous Best (Hybrid): {baseline_acc:.4f}")
        print(f"   Stacking Ensemble:      {stacking_acc:.4f}")
        print(f"   Improvement:            {stacking_acc - baseline_acc:+.4f}")

        if stacking_acc > baseline_acc:
            print("\n🎉 Stacking ensemble improves performance!")
            print(classification_report(y_test, stacking_preds, target_names=['Sad', 'Neutral', 'Happy']))
            with open('stacking_ensemble_results.json', 'w') as f:
                json.dump({'stacking_accuracy': stacking_acc, 'baseline_accuracy': baseline_acc}, f)
        else:
            print("\n📊 Stacking ensemble does not improve performance. The file will be removed.")
        
        return stacking_acc, baseline_acc

if __name__ == "__main__":
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥️  Using device: {device}")

    stacker = StackingEnsemble(device)
    final_accuracy, baseline_accuracy = stacker.run()
    
    # If no improvement, delete the file as per user request
    if final_accuracy <= baseline_accuracy:
        import os
        os.remove('6_stacking_ensemble.py')
        print("\nFile '6_stacking_ensemble.py' has been deleted.") 
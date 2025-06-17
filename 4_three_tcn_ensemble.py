import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import glob
from pathlib import Path
from scipy import signal
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# MODEL ARCHITECTURES
# ================================================================

# Spatial TCN Architecture
class SpatialTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout2d(dropout)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.drop1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out)
        
        out += residual
        out = F.relu(out)
        return out

class ProductionSpatialTCN(nn.Module):
    def __init__(self, in_channels=7, num_classes=3, num_filters=64, 
                 kernel_size=3, dropout=0.1):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size, 
                              padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        # Spatial temporal blocks with increasing dilation
        self.tcn1 = SpatialTemporalBlock(num_filters, num_filters*2, 
                                        kernel_size, dilation=1, dropout=dropout)
        self.tcn2 = SpatialTemporalBlock(num_filters*2, num_filters*4, 
                                        kernel_size, dilation=2, dropout=dropout)
        self.tcn3 = SpatialTemporalBlock(num_filters*4, num_filters*8, 
                                        kernel_size, dilation=4, dropout=dropout)
        self.tcn4 = SpatialTemporalBlock(num_filters*8, num_filters*16, 
                                        kernel_size, dilation=8, dropout=dropout)
        self.tcn5 = SpatialTemporalBlock(num_filters*16, num_filters*32, 
                                        kernel_size, dilation=16, dropout=dropout)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Enhanced classifier with residual connections
        self.fc1 = nn.Linear(num_filters*32, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop_fc2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        self.drop_fc3 = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Input: (batch, time, channels) → (batch, channels, time)
        x = x.permute(0, 2, 1)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Temporal blocks
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        x = self.tcn4(x)
        x = self.tcn5(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)
        
        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.relu(x)
        x = self.drop_fc3(x)
        
        return self.classifier(x)

# Improved TCN Architecture
class ImprovedTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Symmetric padding for better temporal modeling
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection with 1x1 convolution for channel matching
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class ProductionImprovedTCN(nn.Module):
    def __init__(self, in_channels=7, num_classes=3, num_filters=64, 
                 kernel_size=3, dropout=0.1):
        super().__init__()
        
        # Initial convolution with more filters
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        # More temporal blocks with different dilation rates
        self.tcn1 = ImprovedTemporalBlock(num_filters, num_filters*2, kernel_size, dilation=1)
        self.tcn2 = ImprovedTemporalBlock(num_filters*2, num_filters*4, kernel_size, dilation=2)
        self.tcn3 = ImprovedTemporalBlock(num_filters*4, num_filters*8, kernel_size, dilation=4)
        self.tcn4 = ImprovedTemporalBlock(num_filters*8, num_filters*16, kernel_size, dilation=8)
        self.tcn5 = ImprovedTemporalBlock(num_filters*16, num_filters*32, kernel_size, dilation=16)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Deeper fully connected layers
        self.fc1 = nn.Linear(num_filters*32, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(dropout)
        self.fc4 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: (batch, time, channels) → (batch, channels, time)
        x = x.permute(0, 2, 1)
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Temporal blocks
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        x = self.tcn4(x)
        x = self.tcn5(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.drop(x)
        
        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = F.relu(x)
        x = self.drop(x)
        
        return self.fc4(x)

# Basic TCN Architecture
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class ProductionTCN(nn.Module):
    def __init__(self, in_channels=7, num_classes=3, num_filters=32, 
                 kernel_size=3, dropout=0.3):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        # Temporal blocks with increasing dilation
        self.tcn1 = TemporalBlock(num_filters, num_filters*2, kernel_size, dilation=1)
        self.tcn2 = TemporalBlock(num_filters*2, num_filters*4, kernel_size, dilation=2)
        self.tcn3 = TemporalBlock(num_filters*4, num_filters*8, kernel_size, dilation=4)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters*8, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, time, channels) → (batch, channels, time)
        x = x.permute(0, 2, 1)
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Temporal blocks
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.tcn3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.drop(x)
        
        return self.fc3(x)

# ================================================================
# DATA PREPROCESSING
# ================================================================

class DataPreprocessor:
    def __init__(self, augmentation_strategy='savgol_minmax'):
        self.strategy = augmentation_strategy
        self.scaler = None
        self.fitted = False
        
    def savitzky_golay_filter(self, X, window_length=5, polyorder=2):
        """Apply Savitzky-Golay smoothing"""
        X_filtered = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                if len(X[i, :, j]) >= window_length:
                    X_filtered[i, :, j] = signal.savgol_filter(
                        X[i, :, j], window_length, polyorder)
        return X_filtered
    
    def fit_transform(self, X_train):
        """Fit preprocessor on training data and transform"""
        X_processed = X_train.copy()
        
        # Apply denoising if needed
        if 'savgol' in self.strategy:
            print("Applying Savitzky-Golay denoising...")
            X_processed = self.savitzky_golay_filter(X_processed)
        
        # Apply normalization
        if 'minmax' in self.strategy:
            print("Applying Min-Max normalization...")
            self.scaler = MinMaxScaler()
        else:
            print("Applying Z-score normalization...")
            self.scaler = StandardScaler()
        
        # Fit and transform
        train_flat = X_processed.reshape(-1, X_processed.shape[-1])
        self.scaler.fit(train_flat)
        scaled_flat = self.scaler.transform(train_flat)
        X_processed = scaled_flat.reshape(X_processed.shape)
        
        self.fitted = True
        return X_processed
    
    def transform(self, X):
        """Transform new data using fitted preprocessor"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first!")
        
        X_processed = X.copy()
        
        # Apply same denoising
        if 'savgol' in self.strategy:
            X_processed = self.savitzky_golay_filter(X_processed)
        
        # Apply same normalization
        flat = X_processed.reshape(-1, X_processed.shape[-1])
        scaled_flat = self.scaler.transform(flat)
        return scaled_flat.reshape(X_processed.shape)

# ================================================================
# THREE-MODEL ENSEMBLE
# ================================================================

class ThreeTCNEnsemble:
    """Ensemble of Spatial TCN + Improved TCN + Basic TCN"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.preprocessor = None
        self.ensemble_weights = None
        
    def load_models(self, spatial_tcn_path, improved_tcn_path, basic_tcn_path):
        """Load all three trained models"""
        print("Loading Spatial TCN model...")
        
        # Load Spatial TCN
        spatial_model = ProductionSpatialTCN(
            in_channels=7, num_classes=3, num_filters=64, 
            kernel_size=3, dropout=0.1
        ).to(self.device)
        
        spatial_checkpoint = torch.load(spatial_tcn_path, map_location=self.device)
        if 'model_state_dict' in spatial_checkpoint:
            spatial_model.load_state_dict(spatial_checkpoint['model_state_dict'])
        else:
            spatial_model.load_state_dict(spatial_checkpoint)
        spatial_model.eval()
        self.models['spatial_tcn'] = spatial_model
        
        print("Loading Improved TCN model...")
        
        # Load Improved TCN
        improved_model = ProductionImprovedTCN(
            in_channels=7, num_classes=3, num_filters=64, 
            kernel_size=3, dropout=0.1
        ).to(self.device)
        
        improved_checkpoint = torch.load(improved_tcn_path, map_location=self.device)
        if 'model_state_dict' in improved_checkpoint:
            improved_model.load_state_dict(improved_checkpoint['model_state_dict'])
        else:
            improved_model.load_state_dict(improved_checkpoint)
        improved_model.eval()
        self.models['improved_tcn'] = improved_model
        
        print("Loading Basic TCN model...")
        
        # Load Basic TCN
        basic_model = ProductionTCN(
            in_channels=7, num_classes=3, num_filters=32, 
            kernel_size=3, dropout=0.3
        ).to(self.device)
        
        basic_checkpoint = torch.load(basic_tcn_path, map_location=self.device)
        if 'model_state_dict' in basic_checkpoint:
            basic_model.load_state_dict(basic_checkpoint['model_state_dict'])
        else:
            basic_model.load_state_dict(basic_checkpoint)
        basic_model.eval()
        self.models['basic_tcn'] = basic_model
        
        print("✅ All three models loaded successfully")
        
    def load_data(self):
        """Load and preprocess data"""
        print("📊 Loading data...")
        
        # Load data files
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
        
        # Remap labels
        y_all = np.where(y_all==-1, 0, np.where(y_all==0, 1, 2))
        
        print(f"Data: {len(X_all)} samples, {X_all.shape[1]} timesteps, {X_all.shape[2]} channels")
        return X_all, y_all
        
    def evaluate_individual_models(self, X_test, y_test):
        """Evaluate each model individually"""
        print("\n🔍 Evaluating individual models...")
        
        # Create test dataset
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        individual_results = {}
        
        for model_name, model in self.models.items():
            model.eval()
            all_preds = []
            all_targets = []
            all_probs = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    probs = F.softmax(output, dim=1)
                    pred = output.argmax(dim=1)
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            accuracy = accuracy_score(all_targets, all_preds)
            individual_results[model_name] = {
                'accuracy': accuracy,
                'predictions': all_preds,
                'probabilities': all_probs
            }
            
            print(f"   {model_name}: {accuracy:.4f}")
        
        return individual_results
    
    def create_ensemble_predictions(self, X_test, y_test, weight_strategy='performance'):
        """Create ensemble predictions"""
        print(f"\n🎯 Creating ensemble predictions with {weight_strategy} weighting...")
        
        # Get individual model results
        individual_results = self.evaluate_individual_models(X_test, y_test)
        
        # Set ensemble weights based on individual performance
        if weight_strategy == 'performance':
            spatial_acc = individual_results['spatial_tcn']['accuracy']
            improved_acc = individual_results['improved_tcn']['accuracy']
            basic_acc = individual_results['basic_tcn']['accuracy']
            
            # Weight based on relative performance
            total_acc = spatial_acc + improved_acc + basic_acc
            spatial_weight = spatial_acc / total_acc
            improved_weight = improved_acc / total_acc
            basic_weight = basic_acc / total_acc
            
        elif weight_strategy == 'equal':
            spatial_weight = 1/3
            improved_weight = 1/3
            basic_weight = 1/3
        
        elif weight_strategy == 'spatial_heavy':
            spatial_weight = 0.5
            improved_weight = 0.3
            basic_weight = 0.2
            
        elif weight_strategy == 'top_two':
            # Weight only the top 2 performing models
            accuracies = {
                'spatial_tcn': individual_results['spatial_tcn']['accuracy'],
                'improved_tcn': individual_results['improved_tcn']['accuracy'],
                'basic_tcn': individual_results['basic_tcn']['accuracy']
            }
            sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
            
            # Top 2 models get weight, bottom gets 0.1
            spatial_weight = 0.55 if sorted_models[0][0] == 'spatial_tcn' else (0.35 if sorted_models[1][0] == 'spatial_tcn' else 0.1)
            improved_weight = 0.55 if sorted_models[0][0] == 'improved_tcn' else (0.35 if sorted_models[1][0] == 'improved_tcn' else 0.1)
            basic_weight = 0.55 if sorted_models[0][0] == 'basic_tcn' else (0.35 if sorted_models[1][0] == 'basic_tcn' else 0.1)
            
        else:  # custom weights
            spatial_weight = 0.4
            improved_weight = 0.4
            basic_weight = 0.2
        
        self.ensemble_weights = {
            'spatial_tcn': spatial_weight,
            'improved_tcn': improved_weight,
            'basic_tcn': basic_weight
        }
        
        print(f"   Ensemble weights: Spatial={spatial_weight:.3f}, Improved={improved_weight:.3f}, Basic={basic_weight:.3f}")
        
        # Create ensemble predictions
        spatial_probs = np.array(individual_results['spatial_tcn']['probabilities'])
        improved_probs = np.array(individual_results['improved_tcn']['probabilities'])
        basic_probs = np.array(individual_results['basic_tcn']['probabilities'])
        
        # Weighted average of probabilities
        ensemble_probs = (spatial_weight * spatial_probs + 
                         improved_weight * improved_probs +
                         basic_weight * basic_probs)
        
        # Get final predictions
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        return {
            'predictions': ensemble_preds,
            'probabilities': ensemble_probs,
            'individual_results': individual_results
        }
    
    def evaluate_ensemble(self, X_test, y_test, weight_strategy='performance'):
        """Complete ensemble evaluation"""
        print("🏭 THREE TCN ENSEMBLE EVALUATION")
        print("="*60)
        
        # Create ensemble predictions
        ensemble_results = self.create_ensemble_predictions(X_test, y_test, weight_strategy)
        
        # Calculate ensemble metrics
        ensemble_accuracy = accuracy_score(y_test, ensemble_results['predictions'])
        
        # Classification report
        class_names = ['Sad', 'Neutral', 'Happy']
        report = classification_report(
            y_test, ensemble_results['predictions'], 
            target_names=class_names, output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_results['predictions'])
        
        # Print results
        print(f"\n🎯 ENSEMBLE RESULTS:")
        print(f"   Ensemble Accuracy: {ensemble_accuracy:.4f}")
        
        print(f"\n📊 Individual vs Ensemble Performance:")
        for model_name, results in ensemble_results['individual_results'].items():
            print(f"   {model_name}: {results['accuracy']:.4f}")
        print(f"   Ensemble: {ensemble_accuracy:.4f}")
        
        best_individual = max(
            ensemble_results['individual_results']['spatial_tcn']['accuracy'],
            ensemble_results['individual_results']['improved_tcn']['accuracy'],
            ensemble_results['individual_results']['basic_tcn']['accuracy']
        )
        improvement = ensemble_accuracy - best_individual
        print(f"   Improvement: {improvement:+.4f}")
        
        print(f"\n📈 Classification Report:")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name in class_names:
                print(f"   {class_name}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        print(f"\n   Overall metrics:")
        print(f"   Macro Avg: Precision={report['macro avg']['precision']:.3f}, "
              f"Recall={report['macro avg']['recall']:.3f}, F1={report['macro avg']['f1-score']:.3f}")
        
        # Confusion matrix
        print(f"\n🔢 Confusion Matrix:")
        print("     Pred: Sad  Neutral  Happy")
        for i, class_name in enumerate(class_names):
            row_str = f"True {class_name:7}: "
            for j in range(len(class_names)):
                row_str += f"{cm[i,j]:6d}  "
            print(row_str)
        
        # Save results
        results_dict = {
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_weights': self.ensemble_weights,
            'individual_accuracies': {
                name: res['accuracy'] for name, res in ensemble_results['individual_results'].items()
            },
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'improvement': improvement,
            'timestamp': datetime.now().isoformat()
        }
        
        return results_dict

def main():
    """Main ensemble evaluation"""
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥️  Device: {device}")
    
    # Create ensemble
    ensemble = ThreeTCNEnsemble(device)
    
    # Load models
    spatial_tcn_path = 'final_spatial_tcn_models/best_spatial_tcn_20250617_013948_0.7415.pth'
    improved_tcn_path = 'final_improved_tcn_models/best_improved_tcn_20250617_124238_0.7014.pth'
    basic_tcn_path = 'final_tcn_models/best_tcn_20250617_133158_0.7088.pth'
    
    ensemble.load_models(spatial_tcn_path, improved_tcn_path, basic_tcn_path)
    
    # Load and preprocess data
    X_all, y_all = ensemble.load_data()
    
    # Same data split as training
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )
    
    # Preprocess test data
    preprocessor = DataPreprocessor('savgol_minmax')
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"   Test set: {len(X_test)} samples")
    
    # Evaluate ensemble with different weighting strategies
    strategies = ['performance', 'equal', 'spatial_heavy', 'top_two']
    
    best_strategy = None
    best_accuracy = 0
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Testing {strategy} weighting strategy:")
        print(f"{'='*50}")
        
        results = ensemble.evaluate_ensemble(X_test_processed, y_test, strategy)
        all_results[strategy] = results
        
        if results['ensemble_accuracy'] > best_accuracy:
            best_accuracy = results['ensemble_accuracy']
            best_strategy = strategy
    
    print(f"\n🏆 BEST ENSEMBLE STRATEGY: {best_strategy}")
    print(f"   Best Accuracy: {best_accuracy:.4f}")
    
    # Save best results
    with open('three_tcn_ensemble_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Three-model ensemble evaluation completed!")
    print(f"   Results saved: three_tcn_ensemble_results.json")

if __name__ == "__main__":
    main() 
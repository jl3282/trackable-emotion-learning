import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# IMPROVED TCN ARCHITECTURE
# ================================================================

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
        self.tcn5 = ImprovedTemporalBlock(num_filters*16, num_filters*32, kernel_size, dilation=16)  # Added fifth block
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Deeper fully connected layers
        self.fc1 = nn.Linear(num_filters*32, 256)  # Updated input size for fc1
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(dropout)
        self.fc4 = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier/Kaiming initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
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
        x = self.tcn5(x)  # Added fifth block
        
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

# ================================================================
# DATA PREPROCESSING PIPELINE
# ================================================================

class DataPreprocessor:
    def __init__(self, augmentation_strategy='savgol_minmax'):
        """
        augmentation_strategy options:
        - 'baseline': Z-score normalization only
        - 'savgol_zscore': Savitzky-Golay + Z-score (best from experiment)
        - 'minmax': Min-Max scaling (2nd best from experiment)
        - 'savgol_minmax': Savitzky-Golay + Min-Max (combined best)
        """
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
    
    def save(self, filepath):
        """Save preprocessor state"""
        state = {
            'strategy': self.strategy,
            'scaler': self.scaler,
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, filepath):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(state['strategy'])
        preprocessor.scaler = state['scaler']
        preprocessor.fitted = state['fitted']
        return preprocessor

# ================================================================
# TRAINING PIPELINE
# ================================================================

class ImprovedTCNTrainer:
    def __init__(self, model, device, save_dir='final_improved_tcn_models'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_model_path = None
        
    def setup_training(self, lr=2e-3, weight_decay=1e-4, 
                      scheduler_type='cosine', scheduler_params=None):
        """Setup optimizer, scheduler, and loss function"""
        
        # Advanced optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=scheduler_params.get('T_0', 10),
                T_mult=scheduler_params.get('T_mult', 2),
                eta_min=scheduler_params.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5),
                verbose=True
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 20),
                gamma=scheduler_params.get('gamma', 0.5)
            )
        else:
            self.scheduler = None
            
        # Loss function with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"✅ Training setup complete:")
        print(f"   Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
        print(f"   Scheduler: {scheduler_type}")
        print(f"   Loss: CrossEntropyLoss with label smoothing")
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        return total_loss / len(train_loader), correct / total
    
    def validate_epoch(self, val_loader):
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def train(self, train_loader, val_loader, epochs=120, early_stopping_patience=20,
              save_best=True, verbose=True):
        """Full training loop with early stopping"""
        
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print(f"   Early stopping patience: {early_stopping_patience}")
        print(f"   Device: {self.device}")
        
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Check for best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                epochs_without_improvement = 0
                
                if save_best:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.best_model_path = self.save_dir / f"best_improved_tcn_{timestamp}_{val_acc:.4f}.pth"
                    self.save_model(self.best_model_path)
                    
            else:
                epochs_without_improvement += 1
            
            # Verbose output
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Best Val: {self.best_val_acc:.4f}")
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                print(f"   Best validation accuracy: {self.best_val_acc:.4f}")
                break
        
        print(f"\n✅ Training completed!")
        print(f"   Final validation accuracy: {val_acc:.4f}")
        print(f"   Best validation accuracy: {self.best_val_acc:.4f}")
        if self.best_model_path:
            print(f"   Best model saved: {self.best_model_path}")
        
        return self.history
    
    def save_model(self, filepath, include_optimizer=False):
        """Save model with metadata"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'in_channels': 7,
                'num_classes': 3,
                'num_filters': 64,
                'kernel_size': 3,
                'dropout': 0.1
            },
            'training_history': self.history,
            'best_val_acc': self.best_val_acc,
            'timestamp': datetime.now().isoformat()
        }
        
        if include_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                save_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(save_dict, filepath)
        print(f"💾 Model saved: {filepath}")
    
    def load_model(self, filepath, load_optimizer=False):
        """Load model with metadata"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('training_history', self.history)
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"📂 Model loaded: {filepath}")
        print(f"   Best validation accuracy: {self.best_val_acc:.4f}")
        
        return checkpoint

# ================================================================
# EVALUATION AND ANALYSIS
# ================================================================

class ModelEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def evaluate(self, test_loader, class_names=None):
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Classification report
        if class_names is None:
            class_names = ['Sad', 'Neutral', 'Happy']
        
        report = classification_report(all_targets, all_preds, 
                                     target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = ['Sad', 'Neutral', 'Happy']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ================================================================
# COMPLETE PIPELINE
# ================================================================

def create_improved_tcn_pipeline(retrain_from_scratch=True, 
                                existing_model_path=None,
                                augmentation_strategy='savgol_minmax'):
    """
    Complete production pipeline for Improved TCN
    
    Args:
        retrain_from_scratch: Whether to train from scratch or use transfer learning
        existing_model_path: Path to existing model for transfer learning
        augmentation_strategy: Data augmentation strategy to use
    """
    
    print("🏭 IMPROVED TCN PRODUCTION PIPELINE")
    print("="*50)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥️  Device: {device}")
    
    # Load and preprocess data
    print(f"\n📊 Loading and preprocessing data...")
    print(f"   Augmentation strategy: {augmentation_strategy}")
    
    # Your data loading code here (same as before)
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
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples") 
    print(f"   Test: {len(X_test)} samples")
    
    # Preprocess data
    preprocessor = DataPreprocessor(augmentation_strategy)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    preprocessor.save('final_improved_tcn_models/preprocessor.pkl')
    print("💾 Preprocessor saved")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_processed, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_processed, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_processed, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\n🏗️  Creating Improved TCN model...")
    model = ProductionImprovedTCN(
        in_channels=7,
        num_classes=3,
        num_filters=64,
        kernel_size=3,
        dropout=0.1
    )
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup trainer
    trainer = ImprovedTCNTrainer(model, device)
    
    # Transfer learning or train from scratch
    if not retrain_from_scratch and existing_model_path:
        print(f"\n🔄 Loading existing model for transfer learning...")
        checkpoint = trainer.load_model(existing_model_path)
        
        print("   Using transfer learning approach")
        epochs = 50  # Fewer epochs for fine-tuning
        lr = 1e-4   # Lower learning rate for fine-tuning
    else:
        print(f"\n🆕 Training from scratch...")
        epochs = 120  # As requested
        lr = 1e-3   # Reduced from 2e-3 to 1e-3 for more stable training
    
    # Setup training with advanced features
    scheduler_params = {
        'T_0': 30,        # Increased from 15 to 30 for longer cycles
        'T_mult': 2,      # Period multiplier
        'eta_min': 1e-6   # Minimum learning rate
    }
    
    trainer.setup_training(
        lr=lr,
        weight_decay=1e-4,           # L2 regularization
        scheduler_type='cosine',      # Cosine annealing with warm restarts
        scheduler_params=scheduler_params
    )
    
    # Train model
    print(f"\n🎯 Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=20,  # As requested
        save_best=True,
        verbose=True
    )
    
    # Evaluate on test set
    print(f"\n📈 Evaluating on test set...")
    evaluator = ModelEvaluator(model, device)
    
    # Load best model for evaluation
    if trainer.best_model_path:
        trainer.load_model(trainer.best_model_path)
    
    results = evaluator.evaluate(test_loader)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Test Accuracy: {results['accuracy']:.4f}")
    print(f"   Best Val Accuracy: {trainer.best_val_acc:.4f}")
    
    # Classification report
    print(f"\n📊 Classification Report:")
    for class_name, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"   {class_name}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'],
        save_path='final_improved_tcn_models/confusion_matrix.png'
    )
    
    # Save final results
    final_results = {
        'test_accuracy': results['accuracy'],
        'best_val_accuracy': trainer.best_val_acc,
        'classification_report': results['classification_report'],
        'model_path': str(trainer.best_model_path) if trainer.best_model_path else None,
        'augmentation_strategy': augmentation_strategy,
        'training_history': history
    }
    
    with open('final_improved_tcn_models/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n✅ Pipeline completed!")
    print(f"   Model saved: {trainer.best_model_path}")
    print(f"   Results saved: final_improved_tcn_models/final_results.json")
    print(f"   Preprocessor saved: final_improved_tcn_models/preprocessor.pkl")
    
    return {
        'model': model,
        'trainer': trainer,
        'preprocessor': preprocessor,
        'results': results,
        'history': history
    }

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline_results = create_improved_tcn_pipeline(
        retrain_from_scratch=True,
        augmentation_strategy='savgol_minmax'
    ) 
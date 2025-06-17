import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from scipy import signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

class DataAugmentationExperiment:
    """
    Systematic framework for testing data augmentation strategies
    """
    
    def __init__(self, base_data_path='deep_learning_data'):
        self.base_data_path = base_data_path
        self.X_raw = None
        self.y_raw = None
        self.subject_ids = None
        self.device = self._get_device()
        
    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_raw_data(self):
        """Load raw data from all sessions"""
        x_files = sorted(glob.glob(f'{self.base_data_path}/*_x.npy'))
        y_files = sorted(glob.glob(f'{self.base_data_path}/*_y.csv'))
        
        all_X, all_y, subject_ids = [], [], []
        
        for xf, yf in zip(x_files, y_files):
            # Extract subject ID from filename
            subject_id = xf.split('/')[-1].split('_')[0] + '_' + xf.split('/')[-1].split('_')[1]
            
            X = np.load(xf)
            y = pd.read_csv(yf).values.squeeze()
            n = min(len(X), len(y))
            
            all_X.append(X[:n])
            all_y.append(y[:n])
            subject_ids.extend([subject_id] * n)
            
        self.X_raw = np.concatenate(all_X, axis=0)
        self.y_raw = np.concatenate(all_y, axis=0)
        self.subject_ids = np.array(subject_ids)
        
        print(f"Loaded {len(self.X_raw)} samples from {len(set(subject_ids))} subjects")
        print(f"Data shape: {self.X_raw.shape}")
        return self.X_raw, self.y_raw, self.subject_ids

# ================================================================
# NORMALIZATION STRATEGIES
# ================================================================

class NormalizationStrategies:
    """Different normalization approaches"""
    
    @staticmethod
    def z_score_global(X_train, X_val, X_test):
        """Standard Z-score normalization (your current method)"""
        scaler = StandardScaler()
        train_flat = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(train_flat)
        
        def normalize(X):
            flat = X.reshape(-1, X.shape[-1])
            scaled = scaler.transform(flat)
            return scaled.reshape(X.shape)
        
        return normalize(X_train), normalize(X_val), normalize(X_test), scaler
    
    @staticmethod
    def min_max_global(X_train, X_val, X_test):
        """Min-Max normalization (0-1 scaling)"""
        scaler = MinMaxScaler()
        train_flat = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(train_flat)
        
        def normalize(X):
            flat = X.reshape(-1, X.shape[-1])
            scaled = scaler.transform(flat)
            return scaled.reshape(X.shape)
        
        return normalize(X_train), normalize(X_val), normalize(X_test), scaler
    
    @staticmethod
    def robust_global(X_train, X_val, X_test):
        """Robust normalization (median and IQR)"""
        scaler = RobustScaler()
        train_flat = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(train_flat)
        
        def normalize(X):
            flat = X.reshape(-1, X.shape[-1])
            scaled = scaler.transform(flat)
            return scaled.reshape(X.shape)
        
        return normalize(X_train), normalize(X_val), normalize(X_test), scaler
    
    @staticmethod
    def z_score_per_subject(X_train, X_val, X_test, train_subjects, val_subjects, test_subjects):
        """Z-score normalization per subject"""
        X_train_norm = X_train.copy()
        X_val_norm = X_val.copy()
        X_test_norm = X_test.copy()
        
        # Get subject statistics from training data
        subject_stats = {}
        unique_subjects = np.unique(train_subjects)
        
        for subject in unique_subjects:
            subject_mask = train_subjects == subject
            subject_data = X_train[subject_mask]
            
            # Calculate mean and std per channel
            subject_flat = subject_data.reshape(-1, subject_data.shape[-1])
            mean = np.mean(subject_flat, axis=0)
            std = np.std(subject_flat, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            
            subject_stats[subject] = {'mean': mean, 'std': std}
        
        # Normalize each dataset
        def normalize_by_subject(X, subjects, stats):
            X_norm = X.copy()
            for subject in np.unique(subjects):
                mask = subjects == subject
                if subject in stats:
                    # Use subject-specific stats
                    mean, std = stats[subject]['mean'], stats[subject]['std']
                else:
                    # Fallback to global stats for unseen subjects
                    global_flat = X_train.reshape(-1, X_train.shape[-1])
                    mean = np.mean(global_flat, axis=0)
                    std = np.std(global_flat, axis=0)
                    std[std == 0] = 1
                
                # Apply normalization
                for i in range(X_norm.shape[-1]):
                    X_norm[mask, :, i] = (X_norm[mask, :, i] - mean[i]) / std[i]
            
            return X_norm
        
        X_train_norm = normalize_by_subject(X_train, train_subjects, subject_stats)
        X_val_norm = normalize_by_subject(X_val, val_subjects, subject_stats)
        X_test_norm = normalize_by_subject(X_test, test_subjects, subject_stats)
        
        return X_train_norm, X_val_norm, X_test_norm, subject_stats

# ================================================================
# DENOISING STRATEGIES
# ================================================================

class DenoisingStrategies:
    """Different denoising approaches"""
    
    @staticmethod
    def savitzky_golay_filter(X, window_length=5, polyorder=2):
        """Savitzky-Golay smoothing filter"""
        X_denoised = X.copy()
        for i in range(X.shape[0]):  # For each sample
            for j in range(X.shape[2]):  # For each channel
                X_denoised[i, :, j] = signal.savgol_filter(X[i, :, j], window_length, polyorder)
        return X_denoised
    
    @staticmethod
    def median_filter_denoising(X, kernel_size=3):
        """Median filter for removing outliers"""
        X_denoised = X.copy()
        for i in range(X.shape[0]):  # For each sample
            for j in range(X.shape[2]):  # For each channel
                X_denoised[i, :, j] = median_filter(X[i, :, j], size=kernel_size)
        return X_denoised
    
    @staticmethod
    def butterworth_lowpass(X, cutoff=5, fs=24, order=4):
        """Butterworth low-pass filter"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
        X_denoised = X.copy()
        for i in range(X.shape[0]):  # For each sample
            for j in range(X.shape[2]):  # For each channel
                X_denoised[i, :, j] = signal.filtfilt(b, a, X[i, :, j])
        return X_denoised
    
    @staticmethod
    def moving_average(X, window_size=3):
        """Simple moving average filter"""
        X_denoised = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                X_denoised[i, :, j] = np.convolve(X[i, :, j], 
                                                 np.ones(window_size)/window_size, 
                                                 mode='same')
        return X_denoised

# ================================================================
# AUGMENTATION PIPELINE
# ================================================================

class AugmentationPipeline:
    """Complete augmentation pipeline with multiple strategies"""
    
    AUGMENTATION_CONFIGS = {
        'baseline': {
            'normalization': 'z_score_global',
            'denoising': None,
            'description': 'Current baseline (Z-score global)'
        },
        'min_max_global': {
            'normalization': 'min_max_global',
            'denoising': None,
            'description': 'Min-Max scaling instead of Z-score'
        },
        'robust_global': {
            'normalization': 'robust_global',
            'denoising': None,
            'description': 'Robust scaling (median/IQR)'
        },
        'per_subject_zscore': {
            'normalization': 'z_score_per_subject',
            'denoising': None,
            'description': 'Z-score normalization per subject'
        },
        'savgol_denoising': {
            'normalization': 'z_score_global',
            'denoising': 'savitzky_golay',
            'description': 'Z-score + Savitzky-Golay denoising'
        },
        'lowpass_denoising': {
            'normalization': 'z_score_global',
            'denoising': 'butterworth_lowpass',
            'description': 'Z-score + Butterworth low-pass filter'
        },
        'median_denoising': {
            'normalization': 'z_score_global',
            'denoising': 'median_filter',
            'description': 'Z-score + Median filter denoising'
        },
        'combined_per_subject_savgol': {
            'normalization': 'z_score_per_subject',
            'denoising': 'savitzky_golay',
            'description': 'Per-subject Z-score + Savitzky-Golay'
        },
        'combined_minmax_lowpass': {
            'normalization': 'min_max_global',
            'denoising': 'butterworth_lowpass',
            'description': 'Min-Max + Butterworth filter'
        }
    }
    
    def __init__(self, experiment):
        self.experiment = experiment
        self.results = []
    
    def apply_augmentation(self, config_name, X_train, X_val, X_test, 
                          train_subjects, val_subjects, test_subjects):
        """Apply a specific augmentation configuration"""
        
        config = self.AUGMENTATION_CONFIGS[config_name]
        X_train_aug, X_val_aug, X_test_aug = X_train.copy(), X_val.copy(), X_test.copy()
        
        # Apply denoising first (if specified)
        if config['denoising']:
            print(f"  Applying {config['denoising']} denoising...")
            if config['denoising'] == 'savitzky_golay':
                X_train_aug = DenoisingStrategies.savitzky_golay_filter(X_train_aug)
                X_val_aug = DenoisingStrategies.savitzky_golay_filter(X_val_aug)
                X_test_aug = DenoisingStrategies.savitzky_golay_filter(X_test_aug)
            elif config['denoising'] == 'butterworth_lowpass':
                X_train_aug = DenoisingStrategies.butterworth_lowpass(X_train_aug)
                X_val_aug = DenoisingStrategies.butterworth_lowpass(X_val_aug)
                X_test_aug = DenoisingStrategies.butterworth_lowpass(X_test_aug)
            elif config['denoising'] == 'median_filter':
                X_train_aug = DenoisingStrategies.median_filter_denoising(X_train_aug)
                X_val_aug = DenoisingStrategies.median_filter_denoising(X_val_aug)
                X_test_aug = DenoisingStrategies.median_filter_denoising(X_test_aug)
        
        # Apply normalization
        print(f"  Applying {config['normalization']} normalization...")
        if config['normalization'] == 'z_score_global':
            X_train_aug, X_val_aug, X_test_aug, scaler = NormalizationStrategies.z_score_global(
                X_train_aug, X_val_aug, X_test_aug)
        elif config['normalization'] == 'min_max_global':
            X_train_aug, X_val_aug, X_test_aug, scaler = NormalizationStrategies.min_max_global(
                X_train_aug, X_val_aug, X_test_aug)
        elif config['normalization'] == 'robust_global':
            X_train_aug, X_val_aug, X_test_aug, scaler = NormalizationStrategies.robust_global(
                X_train_aug, X_val_aug, X_test_aug)
        elif config['normalization'] == 'z_score_per_subject':
            X_train_aug, X_val_aug, X_test_aug, scaler = NormalizationStrategies.z_score_per_subject(
                X_train_aug, X_val_aug, X_test_aug, train_subjects, val_subjects, test_subjects)
        
        return X_train_aug, X_val_aug, X_test_aug, scaler

# ================================================================
# QUICK MODEL FOR TESTING
# ================================================================

class QuickTCN(nn.Module):
    """Simplified TCN for rapid testing of augmentation strategies"""
    
    def __init__(self, in_channels=7, num_classes=3, num_filters=32, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, 3, padding=1, dilation=2)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, 3, padding=2, dilation=4)
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters*4, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

# ================================================================
# EXPERIMENT RUNNER
# ================================================================

def run_augmentation_experiment():
    """
    Main function to run systematic data augmentation experiments
    """
    
    print("="*80)
    print("DATA AUGMENTATION SYSTEMATIC EXPERIMENT")
    print("="*80)
    
    # Initialize experiment
    exp = DataAugmentationExperiment()
    pipeline = AugmentationPipeline(exp)
    
    # Load raw data
    print("\n1. Loading raw data...")
    X_raw, y_raw, subject_ids = exp.load_raw_data()
    
    # Split data while preserving subject information
    print("\n2. Splitting data...")
    def remap_labels(y): 
        return np.where(y==-1, 0, np.where(y==0, 1, 2))
    
    y_mapped = remap_labels(y_raw)
    
    # Split by subject to avoid data leakage
    unique_subjects = np.unique(subject_ids)
    np.random.seed(42)
    np.random.shuffle(unique_subjects)
    
    n_train = int(0.7 * len(unique_subjects))
    n_val = int(0.15 * len(unique_subjects))
    
    train_subjects_list = unique_subjects[:n_train]
    val_subjects_list = unique_subjects[n_train:n_train+n_val]
    test_subjects_list = unique_subjects[n_train+n_val:]
    
    # Create masks
    train_mask = np.isin(subject_ids, train_subjects_list)
    val_mask = np.isin(subject_ids, val_subjects_list)
    test_mask = np.isin(subject_ids, test_subjects_list)
    
    X_train_base = X_raw[train_mask]
    y_train = y_mapped[train_mask]
    train_subjects = subject_ids[train_mask]
    
    X_val_base = X_raw[val_mask]
    y_val = y_mapped[val_mask]
    val_subjects = subject_ids[val_mask]
    
    X_test_base = X_raw[test_mask]
    y_test = y_mapped[test_mask]
    test_subjects = subject_ids[test_mask]
    
    print(f"Train: {len(X_train_base)} samples from {len(train_subjects_list)} subjects")
    print(f"Val: {len(X_val_base)} samples from {len(val_subjects_list)} subjects")
    print(f"Test: {len(X_test_base)} samples from {len(test_subjects_list)} subjects")
    
    # Test each augmentation strategy
    results = []
    
    print(f"\n3. Testing {len(pipeline.AUGMENTATION_CONFIGS)} augmentation strategies...")
    
    for config_name, config in pipeline.AUGMENTATION_CONFIGS.items():
        print(f"\n--- Testing: {config_name} ---")
        print(f"Description: {config['description']}")
        
        try:
            # Apply augmentation
            X_train_aug, X_val_aug, X_test_aug, scaler = pipeline.apply_augmentation(
                config_name, X_train_base, X_val_base, X_test_base,
                train_subjects, val_subjects, test_subjects
            )
            
            # Quick training test (10 epochs only for speed)
            print("  Training quick model...")
            
            # Convert to tensors
            X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val_aug, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            
            # Train model
            model = QuickTCN().to(exp.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            best_val_acc = 0
            for epoch in range(10):  # Quick test - only 5 epochs
                # Train
                model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(exp.device), y_batch.to(exp.device)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validate
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(exp.device), y_batch.to(exp.device)
                        outputs = model(X_batch)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()
                
                val_acc = val_correct / val_total
                best_val_acc = max(best_val_acc, val_acc)
                
                print(f"    Epoch {epoch+1}/5: Val Acc = {val_acc:.4f}")
            
            results.append({
                'config_name': config_name,
                'description': config['description'],
                'normalization': config['normalization'],
                'denoising': config['denoising'],
                'best_val_acc': best_val_acc,
                'improvement_vs_baseline': 0  # Will calculate later
            })
            
            print(f"  ✅ Best validation accuracy: {best_val_acc:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                'config_name': config_name,
                'description': config['description'],
                'normalization': config['normalization'],
                'denoising': config['denoising'],
                'best_val_acc': 0,
                'improvement_vs_baseline': 0,
                'error': str(e)
            })
    
    # Calculate improvements vs baseline
    baseline_acc = next(r['best_val_acc'] for r in results if r['config_name'] == 'baseline')
    for result in results:
        if 'error' not in result:
            result['improvement_vs_baseline'] = result['best_val_acc'] - baseline_acc
    
    # Save and display results
    results_df = pd.DataFrame(results)
    results_df.to_csv('augmentation_experiment_results.csv', index=False)
    
    print("\n" + "="*80)
    print("AUGMENTATION EXPERIMENT RESULTS")
    print("="*80)
    
    # Sort by performance
    results_df_clean = results_df[~results_df['best_val_acc'].isna()].copy()
    results_df_clean = results_df_clean.sort_values('best_val_acc', ascending=False)
    
    print(f"\n🏆 RANKING:")
    for i, (_, row) in enumerate(results_df_clean.iterrows(), 1):
        improvement = row['improvement_vs_baseline']
        improvement_str = f"({improvement:+.4f})" if improvement != 0 else "(baseline)"
        print(f"{i:2d}. {row['config_name']:<25} {row['best_val_acc']:.4f} {improvement_str}")
        print(f"    {row['description']}")
    
    print(f"\n📊 BEST STRATEGIES:")
    top_3 = results_df_clean.head(3)
    for _, row in top_3.iterrows():
        print(f"• {row['config_name']}: {row['best_val_acc']:.4f} - {row['description']}")
    
    print(f"\nResults saved to: augmentation_experiment_results.csv")
    return results_df

if __name__ == "__main__":
    results = run_augmentation_experiment() 
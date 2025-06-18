#!/usr/bin/env python3
"""
EmoWear Dataset Preprocessing Pipeline
=====================================

This script preprocesses the EmoWear dataset using the same pipeline as the previous work:
- Savitzky-Golay smoothing for denoising
- MinMax normalization for scaling
- Windowed sampling for temporal features
- Timestamp alignment with emotion labels

Creates X (windowed input tensors) and y (emotion labels) ready for TCN training.
"""

import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EmoWearPreprocessor:
    def __init__(self, data_dir="EmoWear_data", window_size=128, overlap=0.75, target_freq=32, 
                 label_strategy="quadrant", signal_focus="accelerometer", validation_strategy="loso"):
        """
        Initialize EmoWear preprocessor
        
        Args:
            data_dir: Directory containing EmoWear data
            window_size: Size of temporal windows (samples)
            overlap: Overlap between windows (0.75 = 75% overlap)
            target_freq: Target sampling frequency for resampling
            label_strategy: "valence" or "quadrant" for emotion labeling
            signal_focus: "accelerometer", "all_physiological", or "core_signals"
            validation_strategy: "random" or "loso" (Leave-One-Subject-Out)
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.overlap = overlap
        self.target_freq = target_freq
        self.step_size = int(window_size * (1 - overlap))
        self.label_strategy = label_strategy
        self.signal_focus = signal_focus
        self.validation_strategy = validation_strategy
        
        # Load analysis results if available
        try:
            with open('emowear_analysis_results.json', 'r') as f:
                self.analysis_results = json.load(f)
            self.core_signals = self.analysis_results['core_signals']
        except:
            self.core_signals = ['e4-acc', 'bh3-acc']  # Fallback to accelerometer only
        
        # Set target signals based on focus
        if signal_focus == "accelerometer":
            self.target_signals = ['e4-acc', 'bh3-acc']  # Focus on accelerometer for transfer learning
        elif signal_focus == "core_signals":
            self.target_signals = self.core_signals
        else:  # all_physiological
            self.target_signals = ['e4-acc', 'e4-bvp', 'e4-eda', 'e4-hr', 'e4-ibi', 'e4-skt',
                                 'bh3-acc', 'bh3-ecg', 'bh3-hr', 'bh3-rr', 'bh3-rsp', 
                                 'bh3-bb', 'bh3-br', 'bh3-hr_confidence']
        
        self.participants = self._get_participants()
        self.processing_log = []
        
        print(f"EmoWear Preprocessor initialized:")
        print(f"  • Window size: {window_size} samples")
        print(f"  • Overlap: {overlap*100:.0f}%")
        print(f"  • Step size: {self.step_size} samples")
        print(f"  • Target frequency: {target_freq} Hz")
        print(f"  • Label strategy: {label_strategy}")
        print(f"  • Signal focus: {signal_focus}")
        print(f"  • Target signals: {len(self.target_signals)} signals")
        print(f"  • Validation strategy: {validation_strategy}")
        
    def _get_participants(self):
        """Get list of participant directories"""
        dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        return sorted(dirs)
    
    def load_participant_data(self, participant):
        """Load all signal data and labels for a participant"""
        participant_dir = os.path.join(self.data_dir, participant)
        
        # Load emotion labels
        surveys_path = os.path.join(participant_dir, "surveys.csv")
        markers_path = os.path.join(participant_dir, "markers-phase2.csv")
        
        if not (os.path.exists(surveys_path) and os.path.exists(markers_path)):
            return None, None, None
            
        surveys = pd.read_csv(surveys_path)
        markers = pd.read_csv(markers_path)
        
        # Create emotion labels based on strategy
        if self.label_strategy == "valence":
            # Valence-based labels: 0=Sad, 1=Neutral, 2=Happy
            surveys['emotion_label'] = pd.cut(surveys['valence'], 
                                            bins=[0, 3.5, 6.5, 10], 
                                            labels=[0, 1, 2]).astype(int)
        else:  # quadrant-based
            # Quadrant-based labels: 0=LVLA, 1=LVHA, 2=HVLA, 3=HVHA
            def classify_emotion_2d(row):
                v, a = row['valence'], row['arousal']
                if v > 5.5 and a > 5.5:
                    return 3  # HVHA (High Valence, High Arousal)
                elif v > 5.5 and a <= 5.5:
                    return 2  # HVLA (High Valence, Low Arousal)
                elif v <= 4.5 and a > 5.5:
                    return 1  # LVHA (Low Valence, High Arousal)
                else:
                    return 0  # LVLA (Low Valence, Low Arousal)
            
            surveys['emotion_label'] = surveys.apply(classify_emotion_2d, axis=1)
        
        # Load signal data
        signals_data = {}
        missing_signals = []
        
        for signal in self.target_signals:
            if signal.startswith('e4-'):
                signal_name = signal.replace('e4-', '')
                filepath = os.path.join(participant_dir, f'signals-e4-{signal_name}.csv')
            else:  # bh3- signals
                signal_name = signal.replace('bh3-', '')
                filepath = os.path.join(participant_dir, f'signals-bh3-{signal_name}.csv')
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if len(df) > 100:  # Only include signals with sufficient data
                        signals_data[signal] = df
                    else:
                        missing_signals.append(f"{signal} (insufficient data: {len(df)} samples)")
                except Exception as e:
                    missing_signals.append(f"{signal} (load error: {str(e)})")
                    continue
            else:
                missing_signals.append(f"{signal} (file not found)")
        
        # Log missing signals for this participant
        if missing_signals:
            self.processing_log.append({
                'participant': participant,
                'issue': 'missing_signals',
                'details': missing_signals
            })
        
        return signals_data, surveys, markers
    
    def preprocess_signal(self, signal_data, signal_name):
        """
        Preprocess individual signal using Savitzky-Golay + MinMax normalization
        
        Args:
            signal_data: DataFrame with 'timestamp' and 'value' columns
            signal_name: Name of the signal for debugging
            
        Returns:
            preprocessed_data: Dictionary with 'timestamp', 'value', 'normalized_value'
        """
        if len(signal_data) < 50:
            return None
            
        # Sort by timestamp
        signal_data = signal_data.sort_values('timestamp').reset_index(drop=True)
        
        # Handle multi-column signals (e.g., ACC with x,y,z)
        if 'value' in signal_data.columns:
            values = signal_data['value'].values
        else:
            # For signals like ACC that have multiple columns, compute magnitude
            value_cols = [col for col in signal_data.columns if col != 'timestamp']
            if len(value_cols) == 3:  # ACC data with x,y,z
                values = np.sqrt(np.sum(signal_data[value_cols].values**2, axis=1))
            elif len(value_cols) == 1:
                values = signal_data[value_cols[0]].values
            else:
                # Take first non-timestamp column
                values = signal_data[value_cols[0]].values
        
        # Remove invalid values
        valid_mask = np.isfinite(values)
        if np.sum(valid_mask) < len(values) * 0.8:  # Skip if >20% invalid
            return None
            
        timestamps = signal_data['timestamp'].values[valid_mask]
        values = values[valid_mask]
        
        # Interpolate to regular time grid
        if len(timestamps) < 10:
            return None
            
        # Calculate sampling frequency
        time_diffs = np.diff(timestamps)
        median_dt = np.median(time_diffs[time_diffs > 0])
        original_freq = 1.0 / median_dt if median_dt > 0 else 32
        
        # Resample to target frequency if needed
        if abs(original_freq - self.target_freq) > 1:
            target_timestamps = np.arange(timestamps[0], timestamps[-1], 1.0/self.target_freq)
            values = np.interp(target_timestamps, timestamps, values)
            timestamps = target_timestamps
        
        # Apply Savitzky-Golay smoothing (same as previous pipeline)
        if len(values) >= 15:  # Minimum length for savgol
            try:
                window_length = min(15, len(values) if len(values) % 2 == 1 else len(values) - 1)
                if window_length >= 5:
                    values_smooth = savgol_filter(values, window_length, 3)
                else:
                    values_smooth = values
            except:
                values_smooth = values
        else:
            values_smooth = values
        
        # MinMax normalization (same as previous pipeline)
        scaler = MinMaxScaler()
        values_normalized = scaler.fit_transform(values_smooth.reshape(-1, 1)).flatten()
        
        return {
            'timestamp': timestamps,
            'value': values_smooth,
            'normalized_value': values_normalized,
            'scaler': scaler
        }
    
    def align_signals_with_labels(self, signals_data, surveys, markers):
        """
        Align preprocessed signals with emotion labels using timestamps
        """
        aligned_data = []
        
        # Process each emotion experiment
        for idx, row in surveys.iterrows():
            exp_id = row['exp']
            emotion_label = row['emotion_label']
            
            # Find corresponding marker timing
            marker_row = markers[markers['exp'] == exp_id]
            if len(marker_row) == 0:
                continue
                
            marker_row = marker_row.iloc[0]
            
            # Get experiment time window (from video start to post-experiment)
            start_time = marker_row['vidB']  # Video begins
            end_time = marker_row['postB']   # Post-experiment begins
            
            if pd.isna(start_time) or pd.isna(end_time):
                continue
                
            duration = end_time - start_time
            if duration < 10:  # Skip very short experiments
                continue
            
            # Extract signal segments for this time window
            experiment_signals = {}
            
            for signal_name, signal_data in signals_data.items():
                if signal_data is None:
                    continue
                    
                # Find data within time window
                mask = (signal_data['timestamp'] >= start_time) & (signal_data['timestamp'] <= end_time)
                segment_length = np.sum(mask)
                
                if segment_length >= self.window_size:  # Ensure sufficient data
                    segment = signal_data['normalized_value'][mask]
                    experiment_signals[signal_name] = segment
            
            min_required_signals = 1 if len(signals_data) == 2 else 2  # Adjust for accelerometer focus
            if len(experiment_signals) >= min_required_signals:
                aligned_data.append({
                    'participant': row.get('participant', 'unknown'),
                    'experiment': exp_id,
                    'emotion_label': emotion_label,
                    'signals': experiment_signals,
                    'duration': duration
                })
        
        return aligned_data
    
    def create_windowed_samples(self, aligned_data):
        """
        Create windowed samples from aligned signal data
        
        Returns:
            X: Array of shape (n_samples, n_features, window_size)
            y: Array of emotion labels
            sample_info: List of dictionaries with sample metadata
        """
        all_windows = []
        all_labels = []
        sample_info = []
        
        # First pass: determine consistent set of signals across all experiments
        all_signal_sets = [set(data_point['signals'].keys()) for data_point in aligned_data]
        common_signals = set.intersection(*all_signal_sets) if all_signal_sets else set()
        
        if len(common_signals) < 4:
            # Fall back to most common signals
            signal_counts = {}
            for signal_set in all_signal_sets:
                for signal in signal_set:
                    signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            # Select top signals that appear in at least 80% of experiments
            min_appearances = len(aligned_data) * 0.8
            common_signals = {signal for signal, count in signal_counts.items() 
                            if count >= min_appearances}
        
        common_signals = sorted(list(common_signals))  # Consistent ordering
        print(f"Using {len(common_signals)} common signals: {common_signals}")
        
        for data_point in aligned_data:
            signals = data_point['signals']
            emotion_label = data_point['emotion_label']
            
            # Check if this experiment has all common signals
            if not all(signal in signals for signal in common_signals):
                continue
            
            # Find common length across common signals
            min_length = min(len(signals[signal]) for signal in common_signals)
            
            if min_length < self.window_size:
                continue
            
            # Truncate all signals to common length
            truncated_signals = {}
            for signal_name in common_signals:
                truncated_signals[signal_name] = signals[signal_name][:min_length]
            
            # Create overlapping windows
            n_windows = (min_length - self.window_size) // self.step_size + 1
            
            for i in range(n_windows):
                start_idx = i * self.step_size
                end_idx = start_idx + self.window_size
                
                # Extract window from each signal in consistent order
                window_data = []
                for signal_name in common_signals:
                    signal_window = truncated_signals[signal_name][start_idx:end_idx]
                    window_data.append(signal_window)
                
                all_windows.append(window_data)
                all_labels.append(emotion_label)
                sample_info.append({
                    'participant': data_point['participant'],
                    'experiment': data_point['experiment'],
                    'window_idx': i,
                    'emotion_label': emotion_label,
                    'signal_names': common_signals,
                    'global_window_idx': len(all_windows) - 1
                })
        
        # Convert to numpy arrays
        X = np.array(all_windows)  # Shape: (n_samples, n_features, window_size)
        y = np.array(all_labels)
        
        return X, y, sample_info
    
    def process_all_participants(self):
        """
        Process all participants and create final dataset
        """
        print(f"\nProcessing {len(self.participants)} participants...")
        
        all_aligned_data = []
        processing_stats = {
            'participants_processed': 0,
            'participants_skipped': 0,
            'total_experiments': 0,
            'valid_experiments': 0,
            'signals_per_participant': []
        }
        
        for participant in tqdm(self.participants, desc="Processing participants"):
            # Load participant data
            signals_data, surveys, markers = self.load_participant_data(participant)
            
            if signals_data is None:
                processing_stats['participants_skipped'] += 1
                continue
            
            # Preprocess signals with imputation for missing signals
            preprocessed_signals = {}
            for signal_name in self.target_signals:
                if signal_name in signals_data:
                    processed = self.preprocess_signal(signals_data[signal_name], signal_name)
                    if processed is not None:
                        preprocessed_signals[signal_name] = processed
            
            # Check if we have minimum required signals
            min_required = 1 if self.signal_focus == "accelerometer" else 2
            if len(preprocessed_signals) < min_required:
                processing_stats['participants_skipped'] += 1
                self.processing_log.append({
                    'participant': participant,
                    'issue': 'insufficient_signals',
                    'details': f'Only {len(preprocessed_signals)} signals available, need at least {min_required}. Available: {list(preprocessed_signals.keys())}'
                })
                continue
            
            # Align with labels
            aligned_data = self.align_signals_with_labels(preprocessed_signals, surveys, markers)
            
            if len(aligned_data) > 0:
                processing_stats['participants_processed'] += 1
                processing_stats['total_experiments'] += len(surveys)
                processing_stats['valid_experiments'] += len(aligned_data)
                processing_stats['signals_per_participant'].append(len(preprocessed_signals))
                
                # Add participant info to aligned data
                for data_point in aligned_data:
                    data_point['participant'] = participant
                
                all_aligned_data.extend(aligned_data)
            else:
                processing_stats['participants_skipped'] += 1
        
        print(f"\nProcessing completed:")
        print(f"  • Participants processed: {processing_stats['participants_processed']}")
        print(f"  • Participants skipped: {processing_stats['participants_skipped']}")
        print(f"  • Valid experiments: {processing_stats['valid_experiments']}/{processing_stats['total_experiments']}")
        if processing_stats['signals_per_participant']:
            print(f"  • Average signals per participant: {np.mean(processing_stats['signals_per_participant']):.1f}")
        
        return all_aligned_data, processing_stats
    
    def create_final_dataset(self):
        """
        Create final preprocessed dataset ready for training
        """
        print("=" * 60)
        print("EMOWEAR PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Process all participants
        aligned_data, stats = self.process_all_participants()
        
        if len(aligned_data) == 0:
            print("ERROR: No valid data found!")
            return None
        
        # Create windowed samples
        print(f"\nCreating windowed samples from {len(aligned_data)} valid experiments...")
        X, y, sample_info = self.create_windowed_samples(aligned_data)
        
        print(f"Dataset created:")
        print(f"  • Total windows: {len(X)}")
        print(f"  • Input shape: {X.shape}")
        print(f"  • Label distribution: {np.unique(y, return_counts=True)}")
        
        # Data splitting based on validation strategy
        if self.validation_strategy == "loso":
            # Leave-One-Subject-Out: Use participants for splits
            participants = [info['participant'] for info in sample_info]
            unique_participants = sorted(list(set(participants)))
            
            # Create participant mapping
            participant_indices = {}
            for i, participant in enumerate(participants):
                if participant not in participant_indices:
                    participant_indices[participant] = []
                participant_indices[participant].append(i)
            
            # For LOSO, we'll save data for each fold
            print(f"\nLOSO Validation Strategy:")
            print(f"  • Total participants: {len(unique_participants)}")
            print(f"  • Each participant will be used as test set once")
            print(f"  • Remaining participants split into train/val")
            
            # Save full dataset for LOSO processing
            X_train, X_val, X_test = X, X, X  # Placeholder, actual splits done during training
            y_train, y_val, y_test = y, y, y  # Placeholder
            
        else:
            # Random splits (same as previous pipeline: 70/15/15)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
            )
            
            print(f"\nRandom Data splits:")
            print(f"  • Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
            print(f"  • Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
            print(f"  • Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Save preprocessed data
        save_dir = f"emowear_preprocessed_data_{self.signal_focus}_{self.label_strategy}_{self.validation_strategy}"
        os.makedirs(save_dir, exist_ok=True)
        
        if self.validation_strategy == "loso":
            # Save full dataset and participant info for LOSO processing
            np.save(f"{save_dir}/X_full.npy", X)
            pd.DataFrame({'label': y}).to_csv(f"{save_dir}/y_full.csv", index=False)
            
            # Save sample info with window indices and participant mapping
            sample_info_df = pd.DataFrame(sample_info)
            sample_info_df.to_csv(f"{save_dir}/sample_info.csv", index=False)
            
            # Save participant indices for LOSO splits
            with open(f"{save_dir}/participant_indices.json", 'w') as f:
                json.dump(participant_indices, f, indent=2)
                
        else:
            # Save regular train/val/test splits
            np.save(f"{save_dir}/X_train.npy", X_train)
            np.save(f"{save_dir}/X_val.npy", X_val)
            np.save(f"{save_dir}/X_test.npy", X_test)
            
            pd.DataFrame({'label': y_train}).to_csv(f"{save_dir}/y_train.csv", index=False)
            pd.DataFrame({'label': y_val}).to_csv(f"{save_dir}/y_val.csv", index=False)
            pd.DataFrame({'label': y_test}).to_csv(f"{save_dir}/y_test.csv", index=False)
        
        # Save metadata
        if self.label_strategy == "valence":
            label_mapping = {0: 'Sad', 1: 'Neutral', 2: 'Happy'}
        else:  # quadrant
            label_mapping = {0: 'LVLA', 1: 'LVHA', 2: 'HVLA', 3: 'HVHA'}
            
        metadata = {
            'dataset': 'EmoWear',
            'preprocessing': {
                'window_size': self.window_size,
                'overlap': self.overlap,
                'target_freq': self.target_freq,
                'normalization': 'MinMax',
                'smoothing': 'Savitzky-Golay',
                'label_strategy': self.label_strategy,
                'signal_focus': self.signal_focus,
                'validation_strategy': self.validation_strategy
            },
            'data_shape': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'window_size': X.shape[2]
            },
            'label_mapping': label_mapping,
            'target_signals': self.target_signals,
            'processing_stats': stats,
            'processing_log': self.processing_log
        }
        
        with open(f"{save_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Preprocessed data saved to '{save_dir}/'")
        print(f"   • Label strategy: {self.label_strategy} ({len(label_mapping)} classes)")
        print(f"   • Signal focus: {self.signal_focus} ({len(self.target_signals)} signals)")
        print(f"   • Validation: {self.validation_strategy}")
        print(f"   • Input shape: (batch, {X.shape[1]}, {X.shape[2]})")
        if self.processing_log:
            print(f"   • Processing issues logged: {len(self.processing_log)} entries")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, metadata

def main():
    """Main preprocessing pipeline"""
    # Create preprocessor focusing on accelerometer for transfer learning
    preprocessor = EmoWearPreprocessor(
        window_size=128,                    # Same as previous
        overlap=0.75,                       # 75% overlap for good temporal coverage
        target_freq=32,                     # 32 Hz target frequency
        label_strategy="quadrant",          # Use quadrant-based labels (4 classes)
        signal_focus="accelerometer",       # Focus on accelerometer for transfer learning
        validation_strategy="loso"          # Use Leave-One-Subject-Out validation
    )
    
    # Create preprocessed dataset
    results = preprocessor.create_final_dataset()
    
    if results is not None:
        X_train, X_val, X_test, y_train, y_val, y_test, metadata = results
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"✅ EmoWear dataset preprocessed with quadrant-based labels")
        print(f"✅ Accelerometer focus for transfer learning with existing TCN models")
        print(f"✅ LOSO validation strategy for robust evaluation")
        print(f"✅ Window indices and participant info saved for analysis")
        
        return results
    else:
        print("❌ Preprocessing failed!")
        return None

if __name__ == "__main__":
    main() 
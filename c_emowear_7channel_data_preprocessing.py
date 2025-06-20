#!/usr/bin/env python3
"""
EmoWear 7-Channel Wrist-Focused Mapping
======================================

User-specified mapping focused on wrist (E4) sensors:
- 5 E4 wrist sensors: e4-acc, e4-bvp, e4-eda, e4-hr, e4-skt
- 2 BH3 chest sensors: bh3-acc, bh3-hr

Total: 7 channels, all REAL sensor data, no simulation.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class EmoWearWristFocusedMapper:
    """Map EmoWear to 7 channels focused on wrist sensors"""
    
    def __init__(self):
        self.input_dir = "emowear_preprocessed_data"
        self.output_dir = "emowear_7channel_wrist_focused"
        
        # User-specified channel mapping
        self.target_channels = [
            'e4-acc',    # Channel 0: Wrist accelerometer
            'e4-bvp',    # Channel 1: Wrist blood volume pulse
            'e4-eda',    # Channel 2: Wrist electrodermal activity
            'e4-hr',     # Channel 3: Wrist heart rate
            'e4-skt',    # Channel 4: Wrist skin temperature
            'bh3-acc',   # Channel 5: Chest accelerometer
            'bh3-hr'     # Channel 6: Chest heart rate
        ]
        
        # Load metadata
        self.load_metadata()
        
        print("🎯 EmoWear → 7-Channel WRIST-FOCUSED Mapper")
        print("="*60)
        print("User-specified channel mapping (NO SIMULATION):")
        for i, signal in enumerate(self.target_channels):
            location = "WRIST" if signal.startswith('e4-') else "CHEST"
            print(f"  Channel {i}: {signal:8s} → {location}")
        print("="*60)
        
    def load_metadata(self):
        """Load existing metadata"""
        with open(f"{self.input_dir}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✓ Current data: {self.metadata['data_shape']}")
        print(f"✓ Available signals: {self.metadata['core_signals']}")
        
        # Create signal index mapping
        self.signal_indices = {signal: idx for idx, signal in enumerate(self.metadata['core_signals'])}
        print(f"✓ Signal indices: {self.signal_indices}")
        
        # Verify all target channels are available
        missing_signals = []
        for signal in self.target_channels:
            if signal not in self.signal_indices:
                missing_signals.append(signal)
        
        if missing_signals:
            print(f"❌ ERROR: Missing signals: {missing_signals}")
            print(f"Available signals: {list(self.signal_indices.keys())}")
            raise ValueError(f"Cannot find required signals: {missing_signals}")
        else:
            print(f"✅ All 7 target signals are available!")
    
    def convert_to_7channel_wrist(self, data_file, output_file):
        """Convert data file using wrist-focused mapping"""
        print(f"\n🔄 Converting {data_file} → {output_file}")
        
        # Load original data
        X = np.load(f"{self.input_dir}/{data_file}")
        print(f"  Original shape: {X.shape}")
        
        # Extract the 7 specified channels in order
        channels_7 = []
        
        for i, signal_name in enumerate(self.target_channels):
            source_idx = self.signal_indices[signal_name]
            signal_data = X[:, source_idx, :]  # Shape: (samples, time)
            
            # Apply normalization to ensure consistency
            scaler = MinMaxScaler()
            signal_normalized = scaler.fit_transform(signal_data.reshape(-1, 1)).reshape(signal_data.shape)
            
            channels_7.append(signal_normalized)
            
            location = "WRIST" if signal_name.startswith('e4-') else "CHEST"
            print(f"    Channel {i}: {signal_name} ({location}, idx {source_idx}) → "
                  f"μ={signal_normalized.mean():.3f}, σ={signal_normalized.std():.3f}")
        
        # Stack into 7-channel format: (samples, 7, time)
        X_7channel = np.stack(channels_7, axis=1)
        
        print(f"  ✅ Created 7-channel data: {X_7channel.shape}")
        
        # Save converted data
        np.save(f"{self.output_dir}/{output_file}", X_7channel)
        
        return X_7channel
    
    def convert_all_data(self):
        """Convert all data using wrist-focused mapping"""
        print(f"\n🚀 Converting all data with WRIST-FOCUSED mapping...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Convert data files
        data_files = ['X_train.npy', 'X_val.npy', 'X_test.npy']
        converted_shapes = {}
        
        for data_file in data_files:
            if os.path.exists(f"{self.input_dir}/{data_file}"):
                X_converted = self.convert_to_7channel_wrist(data_file, data_file)
                converted_shapes[data_file] = X_converted.shape
            else:
                print(f"  ⚠️  {data_file} not found")
        
        # Copy label files
        label_files = ['y_train.csv', 'y_val.csv', 'y_test.csv']
        for label_file in label_files:
            if os.path.exists(f"{self.input_dir}/{label_file}"):
                df = pd.read_csv(f"{self.input_dir}/{label_file}")
                df.to_csv(f"{self.output_dir}/{label_file}", index=False)
                print(f"  📋 Copied {label_file}: {len(df)} labels")
        
        # Create detailed metadata
        channel_details = {}
        for i, signal in enumerate(self.target_channels):
            location = "wrist" if signal.startswith('e4-') else "chest"
            sensor_type = self.classify_signal_type(signal)
            
            channel_details[str(i)] = {
                'signal_name': signal,
                'location': location,
                'sensor_type': sensor_type,
                'source_index': self.signal_indices[signal],
                'description': self.get_signal_description(signal)
            }
        
        new_metadata = {
            'dataset': 'EmoWear_7Channel_Wrist_Focused',
            'source': 'Converted from 12-channel EmoWear using WRIST-FOCUSED mapping',
            'focus': 'Wrist sensors (E4) with minimal chest sensors (BH3)',
            'preprocessing': self.metadata['preprocessing'],
            'data_shape': {
                'n_samples': self.metadata['data_shape']['n_samples'],
                'n_channels': 7,
                'window_size': self.metadata['data_shape']['window_size']
            },
            'label_mapping': self.metadata['label_mapping'],
            'channel_mapping': channel_details,
            'target_channels': self.target_channels,
            'conversion_method': 'direct_wrist_focused_mapping',
            'converted_shapes': converted_shapes,
            'sensor_distribution': {
                'wrist_sensors': [s for s in self.target_channels if s.startswith('e4-')],
                'chest_sensors': [s for s in self.target_channels if s.startswith('bh3-')]
            }
        }
        
        with open(f"{self.output_dir}/metadata.json", 'w') as f:
            json.dump(new_metadata, f, indent=2)
        
        print(f"\n💾 Wrist-focused conversion completed!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  ✅ 5 wrist sensors + 2 chest sensors = 7 channels")
        print(f"  ✅ NO simulated data - all real sensors")
        
        return converted_shapes
    
    def classify_signal_type(self, signal_name):
        """Classify signal by type"""
        if 'acc' in signal_name:
            return 'motion'
        elif 'hr' in signal_name or 'bvp' in signal_name:
            return 'cardiac'
        elif 'eda' in signal_name:
            return 'neural'
        elif 'skt' in signal_name:
            return 'thermal'
        else:
            return 'other'
    
    def get_signal_description(self, signal_name):
        """Get human-readable description"""
        descriptions = {
            'e4-acc': 'Wrist accelerometer (motion/activity)',
            'e4-bvp': 'Wrist blood volume pulse (cardiovascular)',
            'e4-eda': 'Wrist electrodermal activity (arousal/stress)',
            'e4-hr': 'Wrist heart rate (cardiac activity)',
            'e4-skt': 'Wrist skin temperature (thermal regulation)',
            'bh3-acc': 'Chest accelerometer (breathing/movement)',
            'bh3-hr': 'Chest heart rate (cardiac variability)'
        }
        return descriptions.get(signal_name, f'Unknown signal: {signal_name}')
    
    def analyze_conversion(self):
        """Analyze the wrist-focused conversion"""
        print(f"\n🔍 Analyzing wrist-focused conversion...")
        
        # Load converted test data
        X_test = np.load(f"{self.output_dir}/X_test.npy")
        print(f"  Converted shape: {X_test.shape}")
        
        # Analyze by sensor location
        wrist_channels = [i for i, s in enumerate(self.target_channels) if s.startswith('e4-')]
        chest_channels = [i for i, s in enumerate(self.target_channels) if s.startswith('bh3-')]
        
        print(f"\n📊 Sensor distribution:")
        print(f"  Wrist sensors (E4): {len(wrist_channels)} channels → {wrist_channels}")
        print(f"  Chest sensors (BH3): {len(chest_channels)} channels → {chest_channels}")
        
        # Analyze each channel
        print(f"\n📈 Channel details:")
        for i, signal in enumerate(self.target_channels):
            location = "WRIST" if signal.startswith('e4-') else "CHEST"
            sensor_type = self.classify_signal_type(signal)
            
            ch_data = X_test[:, i, :]
            ch_mean = ch_data.mean()
            ch_std = ch_data.std()
            
            print(f"    Channel {i}: {signal:8s} ({location:5s}) → "
                  f"Type: {sensor_type:7s}, μ={ch_mean:.3f}, σ={ch_std:.3f}")
        
        # Signal type counts
        signal_types = [self.classify_signal_type(s) for s in self.target_channels]
        type_counts = {t: signal_types.count(t) for t in set(signal_types)}
        
        print(f"\n🎯 Signal type distribution:")
        for signal_type, count in type_counts.items():
            print(f"  {signal_type}: {count} channels")
        
        print(f"\n✅ Conversion summary:")
        print(f"  • Focus: Wrist-centric with minimal chest data")
        print(f"  • Wrist coverage: 5/7 channels (71%)")
        print(f"  • All real sensor data (no simulation)")
        print(f"  • Ready for wrist-focused transfer learning")
        
        return type_counts

def main():
    """Convert EmoWear to 7-channel wrist-focused format"""
    
    mapper = EmoWearWristFocusedMapper()
    
    # Convert data
    shapes = mapper.convert_all_data()
    
    # Analyze conversion
    analysis = mapper.analyze_conversion()
    
    print(f"\n{'='*60}")
    print(f"🎉 WRIST-FOCUSED CONVERSION COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ 7 channels: 5 wrist + 2 chest sensors")
    print(f"✅ All real sensor data (NO simulation)")
    print(f"✅ Wrist-centric for transfer learning")
    
    # Show the final mapping
    print(f"\n📋 FINAL 7-CHANNEL WRIST-FOCUSED MAPPING:")
    for i, signal in enumerate(mapper.target_channels):
        location = "WRIST" if signal.startswith('e4-') else "CHEST"
        description = mapper.get_signal_description(signal)
        print(f"  Channel {i}: {signal:8s} ({location}) → {description}")
    
    return mapper, shapes

if __name__ == "__main__":
    mapper, shapes = main() 
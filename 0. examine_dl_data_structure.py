import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def examine_x_data():
    """Examine the structure of X data to understand the 7 channels"""
    
    # Load a sample X file
    X_sample = np.load('deep_learning_data/mo_ew2_accdata_21_10_139-1336_x.npy')
    print('='*60)
    print('DEEP LEARNING DATA STRUCTURE ANALYSIS')
    print('='*60)
    print(f'X shape: {X_sample.shape}')
    print(f'X data type: {X_sample.dtype}')
    print(f'Number of windows: {X_sample.shape[0]}')
    print(f'Time steps per window: {X_sample.shape[1]}')
    print(f'Number of channels: {X_sample.shape[2]}')
    
    print('\n' + '='*60)
    print('CHANNEL STATISTICS')
    print('='*60)
    
    # Show statistics for each channel
    print('Channel statistics (mean ± std):')
    for i in range(X_sample.shape[-1]):
        channel_data = X_sample[:, :, i]
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        print(f'Channel {i}: {mean_val:8.4f} ± {std_val:8.4f} (range: {min_val:8.4f} to {max_val:8.4f})')
    
    print('\n' + '='*60)
    print('SAMPLE DATA FROM FIRST WINDOW')
    print('='*60)
    print('Shape of first window:', X_sample[0].shape)
    print('First 5 time steps of first window:')
    print(X_sample[0][:5])
    
    # Analyze patterns to identify channels
    print('\n' + '='*60)
    print('CHANNEL PATTERN ANALYSIS')
    print('='*60)
    
    # Look at typical accelerometer ranges and heart rate ranges
    for i in range(X_sample.shape[-1]):
        channel_data = X_sample[:, :, i].flatten()
        
        # Check if values are in typical accelerometer range (-20 to 20 g)
        acc_like = np.sum((channel_data >= -20) & (channel_data <= 20)) / len(channel_data)
        
        # Check if values are in typical heart rate range (40 to 200 bpm)
        hr_like = np.sum((channel_data >= 40) & (channel_data <= 200)) / len(channel_data)
        
        # Check if values are mostly positive (heart rate characteristic)
        positive_ratio = np.sum(channel_data > 0) / len(channel_data)
        
        print(f'Channel {i}:')
        print(f'  - Accelerometer-like (±20g): {acc_like:.2%}')
        print(f'  - Heart rate-like (40-200): {hr_like:.2%}')
        print(f'  - Positive values: {positive_ratio:.2%}')
        print(f'  - Mean absolute value: {np.mean(np.abs(channel_data)):.4f}')
        
        # Guess the channel type
        if hr_like > 0.8 and positive_ratio > 0.95:
            print(f'  -> LIKELY HEART RATE')
        elif acc_like > 0.9:
            print(f'  -> LIKELY ACCELEROMETER')
        else:
            print(f'  -> UNKNOWN SENSOR')
        print()

def examine_multiple_files():
    """Examine multiple files to confirm pattern"""
    print('\n' + '='*60)
    print('EXAMINING MULTIPLE FILES FOR CONSISTENCY')
    print('='*60)
    
    import glob
    x_files = glob.glob('deep_learning_data/*_x.npy')[:5]  # Check first 5 files
    
    for file_path in x_files:
        print(f'\nFile: {file_path.split("/")[-1]}')
        X = np.load(file_path)
        print(f'Shape: {X.shape}')
        
        # Check channel 6 (index 6) specifically for heart rate characteristics
        if X.shape[-1] > 6:
            channel_6 = X[:, :, 6].flatten()
            hr_range = np.sum((channel_6 >= 40) & (channel_6 <= 200)) / len(channel_6)
            positive = np.sum(channel_6 > 0) / len(channel_6)
            mean_val = np.mean(channel_6)
            
            print(f'Channel 6: mean={mean_val:.2f}, HR-like={hr_range:.2%}, positive={positive:.2%}')

def create_visualization():
    """Create visualization of all channels"""
    print('\n' + '='*60)
    print('CREATING CHANNEL VISUALIZATION')
    print('='*60)
    
    # Load sample data
    X_sample = np.load('deep_learning_data/mo_ew2_accdata_21_10_139-1336_x.npy')
    
    # Plot first window of each channel
    fig, axes = plt.subplots(7, 1, figsize=(12, 14))
    fig.suptitle('All 7 Channels from First Window', fontsize=16)
    
    for i in range(7):
        axes[i].plot(X_sample[0, :, i])
        axes[i].set_title(f'Channel {i} (mean: {np.mean(X_sample[:, :, i]):.2f})')
        axes[i].grid(True, alpha=0.3)
        
        # Add suspected channel type
        channel_data = X_sample[:, :, i].flatten()
        hr_like = np.sum((channel_data >= 40) & (channel_data <= 200)) / len(channel_data)
        positive_ratio = np.sum(channel_data > 0) / len(channel_data)
        
        if hr_like > 0.8 and positive_ratio > 0.95:
            axes[i].set_ylabel('Heart Rate?', color='red')
        else:
            axes[i].set_ylabel('Accelerometer?', color='blue')
    
    axes[-1].set_xlabel('Time Steps')
    plt.tight_layout()
    plt.savefig('channel_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('Visualization saved as channel_analysis.png')

if __name__ == "__main__":
    examine_x_data()
    examine_multiple_files()
    create_visualization() 
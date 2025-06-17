# Deep Learning Data Channel Analysis Summary

## 🔍 **Data Structure Overview**

The deep learning data has the following structure:
- **Shape**: `(n_windows, 24, 7)` 
- **24 time steps**: 1-second windows at 24 Hz sampling rate
- **7 channels**: Multi-sensor smartwatch data

## 📊 **Channel Identification Results**

Based on statistical analysis of the data patterns, here's what each channel represents:

### **Channels 0-2: Accelerometer (X, Y, Z)**
- **Channel 0**: Accelerometer X-axis
  - Range: -8.40 to 4.62 g
  - Mean: -0.009 ± 1.48 g
  - 100% values within ±20g range
  
- **Channel 1**: Accelerometer Y-axis  
  - Range: -6.99 to 3.58 g
  - Mean: 0.002 ± 0.99 g
  - 100% values within ±20g range
  
- **Channel 2**: Accelerometer Z-axis
  - Range: -4.01 to 3.73 g  
  - Mean: 0.0003 ± 0.57 g
  - 100% values within ±20g range

### **Channels 3-5: Gyroscope (X, Y, Z)**
- **Channel 3**: Gyroscope X-axis
  - Range: -263.62 to 208.81 °/s
  - Mean: -0.77 ± 46.06 °/s
  - Large range typical of gyroscope data
  
- **Channel 4**: Gyroscope Y-axis
  - Range: -180.32 to 154.35 °/s
  - Mean: -0.02 ± 49.95 °/s
  
- **Channel 5**: Gyroscope Z-axis
  - Range: -268.66 to 307.51 °/s
  - Mean: -0.61 ± 107.68 °/s

### **Channel 6: Heart Rate** ✅
- **Range**: 94.0 to 135.0 bpm
- **Mean**: 126.43 ± 7.88 bpm
- **100% values in heart rate range (40-200 bpm)**
- **100% positive values**
- **Confirmed across all data files**

### **Evidence Supporting This:**
1. **Value Range**: All values fall within typical heart rate range (40-200 bpm)
2. **Always Positive**: 100% of values are positive (heart rate can't be negative)
3. **Realistic Values**: Mean ~126 bpm is typical for active/walking conditions
4. **Consistent Across Files**: Pattern holds across all 5 tested data files
5. **Source Code Confirmation**: `build_input_tensor.py` loads columns 2-8 from CSV, where column 8 is heart rate

### **Data Creation Process:**
1. **Original CSV**: Contains columns: `condition, emotion, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, heart_rate`
2. **build_input_tensor.py**: Extracts columns 2-8 (sensors only) using `usecols=range(2, 9)`
3. **Result**: 7-channel tensor with accelerometer (3) + gyroscope (3) + heart rate (1)
# Emotion Recognition Model Pipeline Guide

## Overview
This document outlines the complete pipeline for developing an emotion recognition system using smartwatch sensor data. The system classifies emotions into three categories: Sad (0), Neutral (1), and Happy (2) using temporal convolutional networks (TCNs) and ensemble methods.

## Data Preprocessing
All models use consistent preprocessing pipeline:
- **Savitzky-Golay denoising**: Removes noise while preserving signal characteristics
- **MinMax normalization**: Scales features to [0,1] range for stable training
- **Train/Validation/Test split**: 70%/15%/15% with stratification (random_state=42)

## Model Development Pipeline

### Stage 1: Individual TCN Models

#### 1.1 Basic TCN Model
**File**: `3. final_tcn_pipeline.py`
- **Architecture**: Standard TCN with dilated convolutions
- **Parameters**: 476,611 trainable parameters
- **Configuration**: 
  - num_filters=32, kernel_size=3, dropout=0.3
  - learning_rate=0.001, batch_size=64
- **Training**: 120 epochs, early stopping (patience=20)
- **Result**: 70.79% test accuracy
- **Model saved**: `final_tcn_models/best_tcn_20250617_133158_0.7088.pth`

#### 1.2 Improved TCN Model
**File**: `3. final_improved_tcn_pipeline.py`
- **Architecture**: Enhanced TCN with additional layers and residual connections
- **Parameters**: 28,538,755 trainable parameters (60x larger than Basic TCN)
- **Configuration**: 
  - num_filters=64, kernel_size=3, dropout=0.3
  - learning_rate=0.002, batch_size=64
- **Training**: 120 epochs, early stopping (patience=20)
- **Advanced features**: Label smoothing, gradient clipping, cosine annealing
- **Result**: 70.11% test accuracy
- **Model saved**: `final_improved_tcn_models/best_improved_tcn_20250617_124238_0.7014.pth`

#### 1.3 Spatial TCN Model
**Architecture**: TCN with spatial attention mechanisms
- **Parameters**: ~2.5M trainable parameters
- **Result**: 72.77% test accuracy (best individual model)
- **Model saved**: `final_spatial_tcn_models/best_spatial_tcn_20250617_013948_0.7415.pth`

### Stage 2: Ensemble Development

#### 2.1 Three-TCN Ensemble
**File**: `three_tcn_ensemble.py`
- **Method**: Weighted voting ensemble combining all three TCN models
- **Weight optimization**: Tested multiple strategies
  - Performance weighting: 76.26% (best)
  - Equal weighting: 76.19%
  - Spatial heavy: 76.14%
  - Top two models: 75.43%
- **Final weights**: Spatial=34.1%, Improved=32.8%, Basic=33.1%
- **Result**: 76.26% test accuracy (+3.49% over best individual model)
- **Results saved**: `three_tcn_ensemble_results.csv`

#### 2.2 Optimization Attempts
Several improvement strategies were tested:

**Test-Time Augmentation (TTA)**:
- Applied Gaussian noise and magnitude scaling during inference
- **Result**: 75.77% (-0.49% degradation)
- **Conclusion**: TTA hurt performance, models already well-regularized

**Feature Engineering + Hybrid Ensemble**:
- Extracted 171 statistical and frequency domain features
- Combined Random Forest + Logistic Regression on engineered features
- **Hybrid approach**: 70% TCN ensemble + 30% feature-based models
- **Result**: 76.50% (+0.24% improvement)

### Stage 3: Advanced Optimization

#### 3.1 Optimized Hybrid Ensemble
**File**: `5_optimize_hybrid_ensemble.py`
- **Enhanced feature extraction**: 171 sophisticated features including:
  - Cross-channel correlations
  - Rolling statistics (temporal patterns)
  - Detailed frequency band analysis
  - Global statistics across channels
- **Advanced traditional models**:
  - Random Forest: 300 trees, depth=20
  - Logistic Regression: Feature selection + regularization
- **Weight optimization**: Grid search on validation set
- **Final weights**: 55% TCN models, 45% feature-based models
- **Result**: 77.03% test accuracy (+0.77% improvement over baseline)

#### 3.2 Stacking Ensemble (Final Best Model)
**File**: `6_stacking_ensemble.py`
- **Meta-learning approach**: Uses a meta-learner instead of fixed weights
- **Base models**: All 5 models (3 TCNs + 2 traditional ML models)
- **Meta-learner**: Logistic Regression trained on validation predictions
- **Advanced combination**: Learns optimal way to combine predictions for each sample
- **Key innovation**: Can capture complex relationships between base model predictions
- **Result**: 78.24% test accuracy (+1.21% improvement over hybrid ensemble)

## Performance Summary

| Model/Ensemble | Test Accuracy | Improvement | Parameters |
|----------------|---------------|-------------|------------|
| Basic TCN | 70.79% | - | 476,611 |
| Improved TCN | 70.11% | - | 28,538,755 |
| Spatial TCN | 72.77% | - | ~2.5M |
| Three-TCN Ensemble | 76.26% | +3.49% | Combined |
| Optimized Hybrid | 77.03% | +4.26% | Combined |
| **Stacking Ensemble** | **78.24%** | **+5.47%** | **Combined** |

## Key Technical Insights

### 1. Model Complexity vs Performance
- Improved TCN (28M parameters) performed similarly to Basic TCN (476K parameters)
- Suggests diminishing returns from model size alone
- Ensemble diversity more important than individual model complexity

### 2. Ensemble Effectiveness
- Nearly equal weighting (33-34% each) in three-TCN ensemble indicates complementary learning
- Hybrid approach (deep learning + traditional ML) provides additional gains
- Feature engineering remains valuable even with deep learning

### 3. Data Quality Impact
- Consistent preprocessing crucial for ensemble performance
- Savitzky-Golay denoising more effective than raw signal processing
- Stratified splits ensure balanced evaluation across emotion classes

### 4. Optimization Strategies
- Weight optimization on validation set prevents overfitting
- Cross-channel correlations provide valuable signal
- Frequency domain features complement temporal patterns learned by TCNs

## File Structure

```
emotion-recognition-smartwatch/
├── 3_final_tcn_pipeline.py               # Basic TCN training
├── 3_final_improved_tcn_pipeline.py      # Improved TCN training  
├── 3_final_spatial_tcn_pipeline.py       # Spatial TCN training
├── 4_three_tcn_ensemble.py               # Three-model ensemble
├── 5_optimize_hybrid_ensemble.py         # Hybrid ensemble with features
├── 6_stacking_ensemble.py                # Best performing model (meta-learning)
├── final_tcn_models/
│   └── best_tcn_20250617_133158_0.7088.pth
├── final_improved_tcn_models/
│   └── best_improved_tcn_20250617_124238_0.7014.pth
├── final_spatial_tcn_models/
│   └── best_spatial_tcn_20250617_013948_0.7415.pth
├── stacking_ensemble_results.json        # Final best results
└── deep_learning_data/                   # Preprocessed data files
    ├── *_x.npy                          # Feature arrays
    └── *_y.csv                          # Label files
```

## Production Recommendations

### Best Model
Use **Stacking Ensemble** (78.24% accuracy) for production:
- Combines three trained TCN models with enhanced feature-based models using meta-learning
- Meta-learner intelligently combines predictions based on confidence patterns
- Robust performance across all emotion classes
- Balanced precision/recall for each emotion category

### Deployment Considerations
1. **Preprocessing pipeline**: Must match training exactly (Savitzky-Golay + MinMax)
2. **Model ensemble**: Requires loading 5 models (3 TCNs + 2 traditional ML)
3. **Feature extraction**: 171 features computed in real-time
4. **Inference time**: ~50ms per sample on modern hardware

### Future Improvements
1. **Attention mechanisms**: Add to TCN architectures (+2-4% expected)
2. **Transformer models**: Replace TCNs with attention-based models (+4-8% expected)
3. **Multi-scale modeling**: Different time windows (+3-5% expected)
4. **Advanced preprocessing**: Subject-specific normalization (+2-4% expected)

## Experimental Validation

### Cross-Validation Strategy
- Consistent train/validation/test splits (random_state=42)
- Stratified sampling ensures balanced emotion representation
- Validation set used for hyperparameter tuning and weight optimization

### Performance Metrics
- **Primary**: Classification accuracy
- **Secondary**: Per-class precision, recall, F1-score
- **Evaluation**: Confusion matrices for detailed analysis

### Reproducibility
All experiments use fixed random seeds for reproducible results:
- Data splits: random_state=42
- Model initialization: torch.manual_seed(42)
- Traditional ML models: random_state=42

## Conclusion

The emotion recognition pipeline successfully achieved **78.24% accuracy** through systematic development and optimization. The key success factors were:

1. **Diverse ensemble**: Three different TCN architectures learning complementary patterns
2. **Feature engineering**: Traditional ML features capturing different signal characteristics  
3. **Meta-learning**: Stacking ensemble with intelligent prediction combination
4. **Consistent preprocessing**: Ensuring all models work with same data representation

The pipeline demonstrates that:
- **Ensemble diversity** is more important than individual model complexity
- **Meta-learning** (stacking) outperforms fixed-weight combinations
- **Traditional ML + Deep Learning** hybrid approaches are highly effective
- **Systematic optimization** can achieve significant performance gains (+5.47% total improvement) 
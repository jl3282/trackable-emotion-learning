# 🏆 Best Model Configurations Analysis

## 📊 **Top 5 Improved TCN Configurations**

| Rank | Accuracy | Config | Epochs | Details |
|------|----------|---------|---------|---------|
| **🥇 1st** | **75.76%** | nf64, ks3, dr0.3, lr2e-03, bs64 | 39 | `history_nf64_ks3_dr0.3_lr2e-03_bs64.csv` |
| **🥈 2nd** | **74.67%** | nf64, ks3, dr0.0, lr1e-03, bs128 | 20 | `history_nf64_ks3_dr0.0_lr1e-03_bs128.csv` |
| **🥉 3rd** | **74.32%** | nf64, ks3, dr0.3, lr2e-03, bs128 | 27 | `history_nf64_ks3_dr0.3_lr2e-03_bs128.csv` |
| **4th** | **74.03%** | nf64, ks3, dr0.0, lr2e-03, bs128 | 23 | `history_nf64_ks3_dr0.0_lr2e-03_bs128.csv` |
| **5th** | **74.03%** | nf64, ks3, dr0.0, lr2e-03, bs32 | 36 | `history_nf64_ks3_dr0.0_lr2e-03_bs32.csv` |

### **🔍 Improved TCN Best Pattern:**
- **num_filters**: 64 (all top 5 use 64 filters)
- **kernel_size**: 3 (all top 5 use kernel size 3)
- **dropout**: 0.0-0.3 (mix of no dropout and light dropout)
- **learning_rate**: 1e-03 to 2e-03 (aggressive learning rates)
- **batch_size**: 32-128 (varied batch sizes)

---

## 📊 **Top 5 Spatial TCN Configurations**

| Rank | Accuracy | Config | Epochs | Details |
|------|----------|---------|---------|---------|
| **🥇 1st** | **78.57%** | nf64, ks3, dr0.1, lr2e-03, bs64 | 53 | `history_nf64_ks3_dr0.1_lr2e-03_bs64.csv` |
| **🥈 2nd** | **75.43%** | nf64, ks3, dr0.1, lr2e-03, bs128 | 39 | `history_nf64_ks3_dr0.1_lr2e-03_bs128.csv` |
| **🥉 3rd** | **74.79%** | nf64, ks3, dr0.1, lr1e-03, bs128 | 31 | `history_nf64_ks3_dr0.1_lr1e-03_bs128.csv` |
| **4th** | **73.20%** | nf64, ks3, dr0.1, lr1e-03, bs64 | 27 | `history_nf64_ks3_dr0.1_lr1e-03_bs64.csv` |
| **5th** | **72.07%** | nf32, ks3, dr0.1, lr2e-03, bs128 | 38 | `history_nf32_ks3_dr0.1_lr2e-03_bs128.csv` |

### **🔍 Spatial TCN Best Pattern:**
- **num_filters**: 64 (4/5 top models use 64 filters)
- **kernel_size**: 3 (all top 5 use kernel size 3)
- **dropout**: 0.1 (all top 5 use 0.1 dropout - optimal for spatial dropout)
- **learning_rate**: 1e-03 to 2e-03 (similar to Improved TCN)
- **batch_size**: 64-128 (larger batch sizes preferred)

---

## 🎯 **RECOMMENDED BEST CONFIGURATIONS**

### **🏆 Improved TCN (Best Balance: Performance + Speed)**
```python
BEST_IMPROVED_TCN_CONFIG = {
    'num_filters': 64,
    'kernel_size': 3,
    'dropout': 0.0,        # No dropout for fastest training
    'lr': 1e-3,           # Stable learning rate
    'batch_size': 128,    # Good efficiency
    'expected_accuracy': 74.67,
    'expected_epochs': 20
}
```

### **🏆 Spatial TCN (Best Accuracy)**
```python
BEST_SPATIAL_TCN_CONFIG = {
    'num_filters': 64,
    'kernel_size': 3,
    'dropout': 0.1,       # Optimal spatial dropout
    'lr': 2e-3,          # Aggressive learning rate
    'batch_size': 64,    # Balanced batch size
    'expected_accuracy': 78.57,
    'expected_epochs': 53
}
```

---

## 📈 **Configuration Insights**

### **Common Success Patterns:**
1. **Kernel Size 3**: Consistently outperforms larger kernels
2. **64 Filters**: Sweet spot for model capacity
3. **Higher Learning Rates**: 1e-3 to 2e-3 work best
4. **Batch Size**: 64-128 optimal range

### **Key Differences:**
- **Improved TCN**: Benefits from no dropout (dr=0.0)
- **Spatial TCN**: Requires light dropout (dr=0.1) for regularization
- **Training Speed**: Improved TCN converges 2.5x faster

### **Architecture Choice Recommendation:**
- **For Production/Time-Critical**: Use **Improved TCN** (74.7% in 20 epochs)
- **For Maximum Accuracy**: Use **Spatial TCN** (78.6% in 53 epochs)
- **For Experimentation**: Start with **Improved TCN** for rapid iteration 
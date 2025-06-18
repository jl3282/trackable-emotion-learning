#!/usr/bin/env python3
"""
EmoWear Dataset Analysis and Preprocessing Pipeline
=================================================

This script analyzes the EmoWear dataset to determine optimal labeling strategy
and creates preprocessed data for emotion recognition models.

Dataset Structure:
- 47 participants (01-9TZK to 49-9WEW)
- Multiple modalities: E4 wrist sensors (ACC, BVP, EDA, HR, IBI, SKT) and Body-hub sensors (ECG, ACC, HR, RR, RSP)
- Labels: surveys.csv contains valence/arousal/dominance ratings, markers-phase2.csv contains timing info
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter, resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class EmoWearAnalyzer:
    def __init__(self, data_dir="EmoWear_data"):
        self.data_dir = data_dir
        self.participants = self._get_participants()
        self.all_surveys = []
        self.all_markers = []
        self.signal_types = ['e4-acc', 'e4-bvp', 'e4-eda', 'e4-hr', 'e4-ibi', 'e4-skt']
        
    def _get_participants(self):
        """Get list of participant directories"""
        dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        return sorted(dirs)
    
    def analyze_labels(self):
        """Analyze label distributions across all participants"""
        print("=" * 60)
        print("EMOWEAR DATASET LABEL ANALYSIS")
        print("=" * 60)
        
        # Collect all survey data
        for participant in self.participants:
            survey_path = os.path.join(self.data_dir, participant, "surveys.csv")
            markers_path = os.path.join(self.data_dir, participant, "markers-phase2.csv")
            
            if os.path.exists(survey_path):
                df = pd.read_csv(survey_path)
                df['participant'] = participant
                self.all_surveys.append(df)
                
            if os.path.exists(markers_path):
                df = pd.read_csv(markers_path)
                df['participant'] = participant
                self.all_markers.append(df)
        
        # Combine all surveys
        combined_surveys = pd.concat(self.all_surveys, ignore_index=True)
        
        print(f"Total participants: {len(self.participants)}")
        print(f"Total emotion samples: {len(combined_surveys)}")
        print(f"Average samples per participant: {len(combined_surveys) / len(self.participants):.1f}")
        
        # Analyze continuous labels (valence, arousal, dominance)
        print("\n" + "=" * 40)
        print("CONTINUOUS LABELS ANALYSIS")
        print("=" * 40)
        
        for label in ['valence', 'arousal', 'dominance']:
            values = combined_surveys[label]
            print(f"\n{label.upper()}:")
            print(f"  Range: {values.min():.1f} - {values.max():.1f}")
            print(f"  Mean: {values.mean():.2f} ± {values.std():.2f}")
            print(f"  Distribution: Q1={values.quantile(0.25):.1f}, Median={values.median():.1f}, Q3={values.quantile(0.75):.1f}")
        
        # Create discrete labels based on valence and arousal
        print("\n" + "=" * 40)
        print("DISCRETE LABEL CREATION")
        print("=" * 40)
        
        # Strategy 1: Valence-based (similar to previous work)
        combined_surveys['emotion_valence'] = pd.cut(combined_surveys['valence'], 
                                                   bins=[0, 3.5, 6.5, 10], 
                                                   labels=[0, 1, 2])  # Sad, Neutral, Happy
        
        # Strategy 2: Russell's Circumplex Model (Valence + Arousal)
        def classify_emotion_2d(row):
            v, a = row['valence'], row['arousal']
            if v > 5.5 and a > 5.5:  # High valence, high arousal
                return 2  # Happy/Excited
            elif v > 5.5 and a <= 5.5:  # High valence, low arousal
                return 1  # Content/Calm
            elif v <= 4.5 and a > 5.5:  # Low valence, high arousal
                return 0  # Angry/Stressed
            else:  # Low valence, low arousal
                return 0  # Sad/Depressed
        
        combined_surveys['emotion_2d'] = combined_surveys.apply(classify_emotion_2d, axis=1)
        
        # Strategy 3: Arousal-based
        combined_surveys['emotion_arousal'] = pd.cut(combined_surveys['arousal'], 
                                                   bins=[0, 3.5, 6.5, 10], 
                                                   labels=[0, 1, 2])  # Low, Medium, High
        
        # Print distributions
        strategies = [('Valence-based', 'emotion_valence'), 
                     ('2D (Val+Arousal)', 'emotion_2d'),
                     ('Arousal-based', 'emotion_arousal')]
        
        for name, col in strategies:
            print(f"\n{name} Distribution:")
            counts = combined_surveys[col].value_counts().sort_index()
            for i, count in enumerate(counts):
                pct = count / len(combined_surveys) * 100
                print(f"  Class {i}: {count:4d} samples ({pct:5.1f}%)")
        
        # Visualize distributions
        self._plot_label_distributions(combined_surveys)
        
        return combined_surveys
    
    def _plot_label_distributions(self, df):
        """Create visualization of label distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Continuous labels
        for i, label in enumerate(['valence', 'arousal', 'dominance']):
            axes[0, i].hist(df[label], bins=20, alpha=0.7, edgecolor='black')
            axes[0, i].set_title(f'{label.capitalize()} Distribution')
            axes[0, i].set_xlabel(label.capitalize())
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].axvline(df[label].mean(), color='red', linestyle='--', label=f'Mean: {df[label].mean():.2f}')
            axes[0, i].legend()
        
        # Discrete labels
        strategies = [('Valence-based', 'emotion_valence'), 
                     ('2D (Val+Arousal)', 'emotion_2d'),
                     ('Arousal-based', 'emotion_arousal')]
        
        for i, (name, col) in enumerate(strategies):
            counts = df[col].value_counts().sort_index()
            axes[1, i].bar(range(len(counts)), counts.values, alpha=0.7, edgecolor='black')
            axes[1, i].set_title(f'{name}\nDistribution')
            axes[1, i].set_xlabel('Emotion Class')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_xticks(range(len(counts)))
            
            # Add percentage labels
            total = counts.sum()
            for j, count in enumerate(counts.values):
                pct = count / total * 100
                axes[1, i].text(j, count + total*0.01, f'{pct:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig('emowear_label_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Valence-Arousal scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['valence'], df['arousal'], c=df['emotion_2d'], 
                            cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Emotion Class (2D)')
        plt.xlabel('Valence')
        plt.ylabel('Arousal') 
        plt.title('Valence-Arousal Space with 2D Emotion Classification')
        plt.grid(True, alpha=0.3)
        
        # Add quadrant lines
        plt.axhline(5.5, color='red', linestyle='--', alpha=0.5)
        plt.axvline(4.5, color='red', linestyle='--', alpha=0.5)
        plt.axvline(5.5, color='red', linestyle='--', alpha=0.5)
        
        plt.savefig('emowear_valence_arousal_space.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def recommend_labeling_strategy(self, surveys_df):
        """Recommend optimal labeling strategy based on distribution analysis"""
        print("\n" + "=" * 50)
        print("LABELING STRATEGY RECOMMENDATION")
        print("=" * 50)
        
        strategies = [('valence-based', 'emotion_valence'), 
                     ('2d-valence-arousal', 'emotion_2d'),
                     ('arousal-based', 'emotion_arousal')]
        
        recommendations = []
        
        for name, col in strategies:
            counts = surveys_df[col].value_counts().sort_index()
            min_class = counts.min()
            max_class = counts.max()
            balance_ratio = min_class / max_class
            
            recommendations.append({
                'strategy': name,
                'balance_ratio': balance_ratio,
                'min_samples': min_class,
                'total_samples': len(surveys_df),
                'distribution': counts.to_dict()
            })
            
            print(f"\n{name.replace('-', ' ').title()}:")
            print(f"  Balance ratio: {balance_ratio:.3f} (higher is better)")
            print(f"  Min class size: {min_class} samples")
            print(f"  Class distribution: {dict(counts)}")
        
        # Rank strategies
        best_strategy = max(recommendations, key=lambda x: x['balance_ratio'])
        
        print(f"\n{'='*20} RECOMMENDATION {'='*20}")
        print(f"Best strategy: {best_strategy['strategy'].replace('-', ' ').title()}")
        print(f"Reason: Best balance ratio ({best_strategy['balance_ratio']:.3f})")
        print(f"This ensures sufficient samples for all emotion classes.")
        
        return best_strategy['strategy'], surveys_df['emotion_valence' if 'valence' in best_strategy['strategy'] else 
                                                   'emotion_2d' if '2d' in best_strategy['strategy'] else 'emotion_arousal']
    
    def analyze_signal_availability(self):
        """Analyze which signals are available across participants"""
        print("\n" + "=" * 50)
        print("SIGNAL AVAILABILITY ANALYSIS")
        print("=" * 50)
        
        signal_availability = defaultdict(int)
        participant_signals = {}
        
        for participant in self.participants:
            participant_dir = os.path.join(self.data_dir, participant)
            available_signals = []
            
            # Check E4 signals
            e4_signals = ['acc', 'bvp', 'eda', 'hr', 'ibi', 'skt']
            for signal in e4_signals:
                filepath = os.path.join(participant_dir, f'signals-e4-{signal}.csv')
                if os.path.exists(filepath):
                    signal_availability[f'e4-{signal}'] += 1
                    available_signals.append(f'e4-{signal}')
            
            # Check body-hub signals
            bh_signals = ['acc', 'ecg', 'hr', 'rr', 'rsp', 'bb', 'br', 'hr_confidence']
            for signal in bh_signals:
                filepath = os.path.join(participant_dir, f'signals-bh3-{signal}.csv')
                if os.path.exists(filepath):
                    signal_availability[f'bh3-{signal}'] += 1
                    available_signals.append(f'bh3-{signal}')
            
            participant_signals[participant] = available_signals
        
        print("Signal availability across participants:")
        print(f"Total participants: {len(self.participants)}")
        
        for signal, count in sorted(signal_availability.items()):
            pct = count / len(self.participants) * 100
            print(f"  {signal:15s}: {count:2d}/{len(self.participants)} ({pct:5.1f}%)")
        
        # Recommend core signals (available in >80% of participants)
        core_signals = [signal for signal, count in signal_availability.items() 
                       if count / len(self.participants) >= 0.8]
        
        print(f"\nRecommended core signals (≥80% availability): {len(core_signals)} signals")
        for signal in core_signals:
            print(f"  - {signal}")
        
        return core_signals, participant_signals

def main():
    """Main analysis pipeline"""
    print("Initializing EmoWear Dataset Analysis...")
    
    analyzer = EmoWearAnalyzer()
    
    # 1. Analyze labels and distributions
    surveys_df = analyzer.analyze_labels()
    
    # 2. Get labeling recommendation  
    recommended_strategy, recommended_labels = analyzer.recommend_labeling_strategy(surveys_df)
    
    # 3. Analyze signal availability
    core_signals, participant_signals = analyzer.analyze_signal_availability()
    
    # 4. Summary and next steps
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   • {len(analyzer.participants)} participants")
    print(f"   • {len(surveys_df)} emotion samples")
    print(f"   • {len(core_signals)} core signals with ≥80% availability")
    
    print(f"\n🎯 Recommended Labeling Strategy:")
    print(f"   • Strategy: {recommended_strategy.replace('-', ' ').title()}")
    print(f"   • Classes: 0=Sad/Negative, 1=Neutral, 2=Happy/Positive")
    print(f"   • Total samples: {len(recommended_labels)}")
    
    print(f"\n📡 Core Signals for Modeling:")
    for i, signal in enumerate(core_signals[:8], 1):  # Show top 8
        print(f"   {i}. {signal}")
    if len(core_signals) > 8:
        print(f"   ... and {len(core_signals) - 8} more")
    
    # Save analysis results
    results = {
        'recommended_strategy': recommended_strategy,
        'core_signals': core_signals,
        'total_participants': len(analyzer.participants),
        'total_samples': len(surveys_df),
        'label_distribution': recommended_labels.value_counts().to_dict()
    }
    
    import json
    with open('emowear_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Analysis results saved to 'emowear_analysis_results.json'")
    
    return analyzer, surveys_df, recommended_strategy, core_signals

if __name__ == "__main__":
    analyzer, surveys_df, strategy, signals = main()
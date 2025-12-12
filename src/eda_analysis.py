"""
Exploratory Data Analysis for Depression Prediction Dataset

This script performs comprehensive EDA to understand:
1. Dataset structure and dimensions
2. Missing value patterns (distinguish reasonable vs anomalous)
3. Target variable distribution (class imbalance)
4. Feature distributions and statistics
5. Correlations and relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DepressionEDA:
    def __init__(self, train_path, test_path):
        """
        Initialize EDA with train and test datasets
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.output_dir = Path('eda_outputs')
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load train and test datasets"""
        print("Loading datasets...")
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        print("\n" + "="*80 + "\n")
        
    def basic_info(self):
        """Display basic dataset information"""
        print("TRAIN DATASET INFO:")
        print("-" * 80)
        print(self.train_df.info())
        print("\n" + "="*80 + "\n")
        
        print("TRAIN DATASET DESCRIPTION:")
        print("-" * 80)
        print(self.train_df.describe())
        print("\n" + "="*80 + "\n")
        
        print("TRAIN DATASET HEAD:")
        print("-" * 80)
        print(self.train_df.head(10))
        print("\n" + "="*80 + "\n")
        
    def analyze_target_distribution(self):
        """Analyze target variable distribution and class imbalance"""
        print("TARGET VARIABLE ANALYSIS:")
        print("-" * 80)
        
        depression_counts = self.train_df['Depression'].value_counts()
        depression_pct = self.train_df['Depression'].value_counts(normalize=True) * 100
        
        print("Depression Distribution (counts):")
        print(depression_counts)
        print("\nDepression Distribution (percentage):")
        print(depression_pct)
        
        imbalance_ratio = depression_counts.max() / depression_counts.min()
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 1.5:
            print("WARNING: Significant class imbalance detected!")
            print("Recommendation: Use SMOTE, class weights, or stratified sampling")
        
        print("\n" + "="*80 + "\n")
        
        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        depression_counts.plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
        axes[0].set_title('Depression Class Distribution (Counts)')
        axes[0].set_xlabel('Depression')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels(['No Depression (0)', 'Depression (1)'], rotation=0)
        
        depression_pct.plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'])
        axes[1].set_title('Depression Class Distribution (Percentage)')
        axes[1].set_xlabel('Depression')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_xticklabels(['No Depression (0)', 'Depression (1)'], rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_missing_values(self):
        """Analyze missing values patterns - distinguish reasonable vs anomalous"""
        print("MISSING VALUES ANALYSIS:")
        print("-" * 80)
        
        # Calculate missing values
        missing_train = self.train_df.isnull().sum()
        missing_train_pct = (missing_train / len(self.train_df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_train.index,
            'Missing_Count': missing_train.values,
            'Missing_Percentage': missing_train_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        print("Missing Values Summary:")
        print(missing_df.to_string(index=False))
        print("\n")
        
        # Analyze reasonable missing patterns
        print("REASONABLE MISSING VALUE PATTERNS:")
        print("-" * 80)
        
        # Pattern 1: Students should have no Profession, Work Pressure, Job Satisfaction
        students = self.train_df[self.train_df['Working Professional or Student'] == 'Student']
        print(f"\nTotal Students: {len(students)}")
        print(f"Students with missing Profession: {students['Profession'].isnull().sum()}")
        print(f"Students with missing Work Pressure: {students['Work Pressure'].isnull().sum()}")
        print(f"Students with missing Job Satisfaction: {students['Job Satisfaction'].isnull().sum()}")
        
        # Pattern 2: Working Professionals should have no CGPA, Academic Pressure, Study Satisfaction
        professionals = self.train_df[self.train_df['Working Professional or Student'] == 'Working Professional']
        print(f"\nTotal Working Professionals: {len(professionals)}")
        print(f"Professionals with missing CGPA: {professionals['CGPA'].isnull().sum()}")
        print(f"Professionals with missing Academic Pressure: {professionals['Academic Pressure'].isnull().sum()}")
        print(f"Professionals with missing Study Satisfaction: {professionals['Study Satisfaction'].isnull().sum()}")
        
        # Pattern 3: Check for anomalous missing values
        print("\n\nANOMALOUS MISSING VALUE PATTERNS:")
        print("-" * 80)
        
        # Students who should have CGPA but don't
        students_no_cgpa = students[students['CGPA'].isnull()]
        print(f"Students with missing CGPA: {len(students_no_cgpa)} (potential anomaly)")
        
        # Professionals who should have Profession but don't
        prof_no_profession = professionals[professionals['Profession'].isnull()]
        print(f"Professionals with missing Profession: {len(prof_no_profession)} (potential anomaly)")
        
        # Anyone with missing core features (Gender, Age, etc.)
        core_features = ['Gender', 'Age', 'Sleep Duration', 'Dietary Habits', 
                        'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 
                        'Financial Stress', 'Family History of Mental Illness']
        
        print("\nMissing Core Features (should be minimal):")
        for feature in core_features:
            if feature in self.train_df.columns:
                missing_count = self.train_df[feature].isnull().sum()
                if missing_count > 0:
                    print(f"  {feature}: {missing_count} ({missing_count/len(self.train_df)*100:.2f}%)")
        
        print("\n" + "="*80 + "\n")
        
        # Visualization
        if len(missing_df) > 0:
            plt.figure(figsize=(12, 6))
            plt.barh(missing_df['Column'], missing_df['Missing_Percentage'], color='coral')
            plt.xlabel('Missing Percentage (%)')
            plt.title('Missing Values by Feature')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'missing_values.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def analyze_categorical_features(self):
        """Analyze categorical feature distributions"""
        print("CATEGORICAL FEATURES ANALYSIS:")
        print("-" * 80)
        
        categorical_cols = self.train_df.select_dtypes(include=['object']).columns.tolist()
        # Remove id and Name as they're not useful for analysis
        categorical_cols = [col for col in categorical_cols if col not in ['id', 'Name', 'Depression']]
        
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.train_df[col].value_counts())
            print("-" * 40)
            
        print("\n" + "="*80 + "\n")
        
    def analyze_numerical_features(self):
        """Analyze numerical feature distributions"""
        print("NUMERICAL FEATURES ANALYSIS:")
        print("-" * 80)
        
        numerical_cols = self.train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Remove id and Depression target
        numerical_cols = [col for col in numerical_cols if col not in ['id', 'Depression']]
        
        print("\nNumerical Features Statistics:")
        print(self.train_df[numerical_cols].describe())
        
        print("\n" + "="*80 + "\n")
        
        # Create distribution plots for key numerical features
        if len(numerical_cols) > 0:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if len(numerical_cols) == 1 else axes
            
            for idx, col in enumerate(numerical_cols):
                if idx < len(axes):
                    self.train_df[col].dropna().hist(bins=30, ax=axes[idx], edgecolor='black')
                    axes[idx].set_title(f'Distribution of {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
            
            # Hide unused subplots
            for idx in range(len(numerical_cols), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def analyze_correlations(self):
        """Analyze correlations between numerical features and target"""
        print("CORRELATION ANALYSIS:")
        print("-" * 80)
        
        # Select only numerical columns
        numerical_cols = self.train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in ['id']]
        
        if len(numerical_cols) > 1:
            corr_matrix = self.train_df[numerical_cols].corr()
            
            print("\nCorrelation with Depression (target):")
            if 'Depression' in corr_matrix.columns:
                depression_corr = corr_matrix['Depression'].sort_values(ascending=False)
                print(depression_corr)
            
            print("\n" + "="*80 + "\n")
            
            # Visualization
            plt.figure(figsize=(14, 12))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1)
            plt.title('Correlation Matrix of Numerical Features')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def analyze_feature_importance_by_depression(self):
        """Analyze how features differ between depression classes"""
        print("FEATURE DIFFERENCES BY DEPRESSION STATUS:")
        print("-" * 80)
        
        # Numerical features comparison
        numerical_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                         'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 
                         'Financial Stress']
        
        numerical_cols = [col for col in numerical_cols if col in self.train_df.columns]
        
        print("\nMean values by Depression status:")
        for col in numerical_cols:
            print(f"\n{col}:")
            print(self.train_df.groupby('Depression')[col].agg(['mean', 'median', 'std']))
            
        print("\n" + "="*80 + "\n")
        
        # Create boxplots
        if len(numerical_cols) > 0:
            n_cols = 3
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if len(numerical_cols) == 1 else axes
            
            for idx, col in enumerate(numerical_cols):
                if idx < len(axes):
                    self.train_df.boxplot(column=col, by='Depression', ax=axes[idx])
                    axes[idx].set_title(f'{col} by Depression Status')
                    axes[idx].set_xlabel('Depression')
                    axes[idx].set_ylabel(col)
                    plt.sca(axes[idx])
                    plt.xticks([1, 2], ['No Depression', 'Depression'])
            
            # Hide unused subplots
            for idx in range(len(numerical_cols), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'features_by_depression.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("="*80)
        print("SUMMARY REPORT: KEY FINDINGS")
        print("="*80)
        
        report = []
        
        # Dataset size
        report.append(f"1. Dataset Size:")
        report.append(f"   - Training samples: {len(self.train_df):,}")
        report.append(f"   - Test samples: {len(self.test_df):,}")
        report.append(f"   - Total features: {self.train_df.shape[1] - 2}")  # Exclude id and target
        
        # Class imbalance
        depression_counts = self.train_df['Depression'].value_counts()
        imbalance_ratio = depression_counts.max() / depression_counts.min()
        report.append(f"\n2. Class Imbalance:")
        report.append(f"   - No Depression (0): {depression_counts[0]:,} ({depression_counts[0]/len(self.train_df)*100:.1f}%)")
        report.append(f"   - Depression (1): {depression_counts[1]:,} ({depression_counts[1]/len(self.train_df)*100:.1f}%)")
        report.append(f"   - Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        # Missing values
        missing_total = self.train_df.isnull().sum().sum()
        report.append(f"\n3. Missing Values:")
        report.append(f"   - Total missing: {missing_total:,}")
        report.append(f"   - Columns with missing: {(self.train_df.isnull().sum() > 0).sum()}")
        
        # Feature types
        categorical_count = len(self.train_df.select_dtypes(include=['object']).columns) - 2
        numerical_count = len(self.train_df.select_dtypes(include=['int64', 'float64']).columns) - 2
        report.append(f"\n4. Feature Types:")
        report.append(f"   - Categorical features: {categorical_count}")
        report.append(f"   - Numerical features: {numerical_count}")
        
        # Recommendations
        report.append(f"\n5. Recommendations:")
        report.append(f"   - Handle class imbalance using SMOTE or class weights")
        report.append(f"   - Intelligently handle missing values based on context")
        report.append(f"   - Consider feature engineering for interaction effects")
        report.append(f"   - Use stratified K-fold for validation")
        report.append(f"   - Try tree-based models (XGBoost, LightGBM, CatBoost)")
        report.append(f"   - Implement SHAP for interpretability")
        
        report_text = "\n".join(report)
        print(report_text)
        print("\n" + "="*80 + "\n")
        
        # Save report
        with open(self.output_dir / 'eda_summary_report.txt', 'w') as f:
            f.write(report_text)
            
    def run_full_analysis(self):
        """Run complete EDA pipeline"""
        print("\n")
        print("="*80)
        print("DEPRESSION PREDICTION - EXPLORATORY DATA ANALYSIS")
        print("="*80)
        print("\n")
        
        self.load_data()
        self.basic_info()
        self.analyze_target_distribution()
        self.analyze_missing_values()
        self.analyze_categorical_features()
        self.analyze_numerical_features()
        self.analyze_correlations()
        self.analyze_feature_importance_by_depression()
        self.generate_summary_report()
        
        print(f"Analysis complete! Results saved to: {self.output_dir}/")
        print("\nKey outputs:")
        print(f"  - target_distribution.png")
        print(f"  - missing_values.png")
        print(f"  - numerical_distributions.png")
        print(f"  - correlation_matrix.png")
        print(f"  - features_by_depression.png")
        print(f"  - eda_summary_report.txt")


if __name__ == "__main__":
    # Initialize and run EDA
    eda = DepressionEDA(
        train_path='data/train.csv',
        test_path='data/test.csv'
    )
    eda.run_full_analysis()


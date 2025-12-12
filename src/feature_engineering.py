"""
Feature Engineering Module for Depression Prediction

This module creates additional features to improve model performance:
1. Interaction features (e.g., stress * hours)
2. Polynomial features for key predictors
3. Domain-specific features (stress index, sleep quality score)
4. Aggregated features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineer"""
        self.poly_features = None
        self.feature_names = None
        self.original_features = None
        self.training_statistics = {}  # Store statistics computed from training set only
        
    def create_stress_features(self, df):
        """
        Create stress-related features
        
        Args:
            df: Input dataframe with features
            
        Returns:
            DataFrame with additional stress features
        """
        df = df.copy()
        
        # Total stress index: combination of all stress factors
        stress_components = []
        
        if 'Academic Pressure' in df.columns:
            stress_components.append(df['Academic Pressure'])
        if 'Work Pressure' in df.columns:
            stress_components.append(df['Work Pressure'])
        if 'Financial Stress' in df.columns:
            stress_components.append(df['Financial Stress'])
        
        if len(stress_components) > 0:
            df['Total_Stress_Index'] = sum(stress_components) / len(stress_components)
            
            # Maximum stress (highest stress factor)
            df['Max_Stress'] = np.maximum.reduce(stress_components)
            
            # Stress variance (consistency of stress across domains)
            stress_array = np.column_stack(stress_components)
            df['Stress_Variance'] = np.var(stress_array, axis=1)
        
        # Work/Study intensity: stress * hours
        if 'Work/Study Hours' in df.columns:
            if 'Academic Pressure' in df.columns:
                df['Academic_Intensity'] = df['Academic Pressure'] * df['Work/Study Hours']
            if 'Work Pressure' in df.columns:
                df['Work_Intensity'] = df['Work Pressure'] * df['Work/Study Hours']
        
        return df
        
    def create_lifestyle_features(self, df):
        """
        Create lifestyle-related features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with lifestyle features
        """
        df = df.copy()
        
        # Sleep quality score: sleep duration is categorical, we'll use encoded values
        # Higher values of Sleep Duration (encoded) should correlate with better sleep
        
        # Work-life balance: inverse of work/study hours
        if 'Work/Study Hours' in df.columns:
            df['Work_Life_Balance'] = 1.0 / (df['Work/Study Hours'] + 1)
            
            # Overwork indicator: binary flag for excessive hours
            df['Overwork_Flag'] = (df['Work/Study Hours'] >= 10).astype(int)
        
        # Health lifestyle score: combination of sleep and diet
        lifestyle_components = []
        if 'Sleep Duration' in df.columns:
            lifestyle_components.append(df['Sleep Duration'])
        if 'Dietary Habits' in df.columns:
            lifestyle_components.append(df['Dietary Habits'])
        
        if len(lifestyle_components) > 0:
            df['Lifestyle_Score'] = sum(lifestyle_components) / len(lifestyle_components)
        
        return df
        
    def create_satisfaction_features(self, df, is_training=True):
        """
        Create satisfaction-related features
        
        Args:
            df: Input dataframe
            is_training: Whether this is training data (compute statistics)
            
        Returns:
            DataFrame with satisfaction features
        """
        df = df.copy()
        
        # Overall satisfaction: average of study and job satisfaction
        satisfaction_components = []
        
        if 'Study Satisfaction' in df.columns:
            satisfaction_components.append(df['Study Satisfaction'])
        if 'Job Satisfaction' in df.columns:
            satisfaction_components.append(df['Job Satisfaction'])
        
        if len(satisfaction_components) > 0:
            df['Overall_Satisfaction'] = sum(satisfaction_components) / len(satisfaction_components)
            
            # Satisfaction-stress ratio
            if 'Total_Stress_Index' in df.columns:
                df['Satisfaction_Stress_Ratio'] = (df['Overall_Satisfaction'] + 1) / (df['Total_Stress_Index'] + 1)
        
        # Achievement indicator: CGPA above median (use training median to prevent leakage)
        if 'CGPA' in df.columns:
            if is_training:
                # Training: Compute and store median
                median_cgpa = df['CGPA'].median()
                self.training_statistics['median_cgpa'] = median_cgpa
            else:
                # Validation/Test: Use training median
                median_cgpa = self.training_statistics.get('median_cgpa', df['CGPA'].median())
            
            df['High_Achiever'] = (df['CGPA'] >= median_cgpa).astype(int)
        
        return df
        
    def create_risk_features(self, df):
        """
        Create depression risk features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with risk features
        """
        df = df.copy()
        
        # Suicidal thoughts is a strong indicator
        # Already encoded, so we can use directly
        
        # Family history combined with personal stress
        if 'Family History of Mental Illness' in df.columns and 'Total_Stress_Index' in df.columns:
            df['Family_Stress_Interaction'] = df['Family History of Mental Illness'] * df['Total_Stress_Index']
        
        # Age-related risk: younger people might be more vulnerable
        if 'Age' in df.columns:
            df['Youth_Risk'] = (df['Age'] <= 30).astype(int)
            df['Age_Stress_Interaction'] = df['Age'] * df['Total_Stress_Index'] if 'Total_Stress_Index' in df.columns else df['Age']
        
        # Multiple risk factors indicator
        risk_factors = []
        
        if 'Have you ever had suicidal thoughts ?' in df.columns:
            risk_factors.append(df['Have you ever had suicidal thoughts ?'])
        if 'Family History of Mental Illness' in df.columns:
            risk_factors.append(df['Family History of Mental Illness'])
        if 'Overwork_Flag' in df.columns:
            risk_factors.append(df['Overwork_Flag'])
        
        if len(risk_factors) > 0:
            df['Total_Risk_Factors'] = sum(risk_factors)
        
        return df
        
    def create_interaction_features(self, df, key_features=None):
        """
        Create interaction features between key predictors
        
        Args:
            df: Input dataframe
            key_features: List of features to create interactions for
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        if key_features is None:
            # Default key features for interactions
            key_features = [
                'Academic Pressure', 'Work Pressure', 'Financial Stress',
                'Work/Study Hours', 'Age', 'Have you ever had suicidal thoughts ?'
            ]
        
        # Filter to features that exist in dataframe
        key_features = [f for f in key_features if f in df.columns]
        
        # Create pairwise interactions for most important features
        important_pairs = [
            ('Academic Pressure', 'Work/Study Hours'),
            ('Work Pressure', 'Work/Study Hours'),
            ('Financial Stress', 'Work/Study Hours'),
            ('Age', 'Academic Pressure'),
            ('Age', 'Work Pressure'),
            ('Have you ever had suicidal thoughts ?', 'Total_Stress_Index'),
        ]
        
        for feat1, feat2 in important_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f'{feat1}_x_{feat2}'.replace(' ', '_').replace('?', '').replace('/', '_')
                df[interaction_name] = df[feat1] * df[feat2]
        
        return df
        
    def create_polynomial_features(self, df, degree=2, key_features=None):
        """
        Create polynomial features for key predictors
        
        Args:
            df: Input dataframe
            degree: Polynomial degree
            key_features: Features to create polynomials for
            
        Returns:
            DataFrame with polynomial features
        """
        df = df.copy()
        
        if key_features is None:
            # Select numerical features that are good candidates for polynomials
            key_features = [
                'Age', 'Work/Study Hours', 'Total_Stress_Index'
            ]
        
        # Filter to features that exist
        key_features = [f for f in key_features if f in df.columns]
        
        # Create polynomial features
        for feature in key_features:
            for d in range(2, degree + 1):
                df[f'{feature}_pow{d}'] = df[feature] ** d
        
        return df
        
    def create_all_features(self, df, is_training=True):
        """
        Create all engineered features
        
        Args:
            df: Input dataframe
            is_training: Whether this is training data (compute statistics)
            
        Returns:
            DataFrame with all engineered features
        """
        print("Creating engineered features...")
        original_shape = df.shape
        
        # Create features in sequence
        df = self.create_stress_features(df)
        print(f"  - Stress features created. Shape: {df.shape}")
        
        df = self.create_lifestyle_features(df)
        print(f"  - Lifestyle features created. Shape: {df.shape}")
        
        df = self.create_satisfaction_features(df, is_training=is_training)
        print(f"  - Satisfaction features created. Shape: {df.shape}")
        
        df = self.create_risk_features(df)
        print(f"  - Risk features created. Shape: {df.shape}")
        
        df = self.create_interaction_features(df)
        print(f"  - Interaction features created. Shape: {df.shape}")
        
        df = self.create_polynomial_features(df, degree=2)
        print(f"  - Polynomial features created. Shape: {df.shape}")
        
        print(f"\nFeature engineering complete!")
        print(f"  Original features: {original_shape[1]}")
        print(f"  Total features: {df.shape[1]}")
        print(f"  New features added: {df.shape[1] - original_shape[1]}")
        
        if is_training:
            print(f"  Statistics stored for validation/test: {len(self.training_statistics)}")
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        return df
        
    def handle_inf_nan(self, df, is_training=True):
        """
        Handle infinite and NaN values that may result from feature engineering
        Uses training statistics to prevent leakage
        
        Args:
            df: Input dataframe
            is_training: Whether this is training data (compute statistics)
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Replace inf with large values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median (using training statistics)
        for col in df.columns:
            if df[col].isnull().any():
                if is_training:
                    # Training: Compute and store median
                    fill_value = df[col].median()
                    self.training_statistics[f'{col}_fillna'] = fill_value
                else:
                    # Validation/Test: Use training median
                    fill_value = self.training_statistics.get(f'{col}_fillna', 0.0)
                
                df[col] = df[col].fillna(fill_value)
        
        return df


def engineer_features(X, feature_names, is_training=True, engineer=None):
    """
    Apply feature engineering to feature matrix
    
    Args:
        X: Feature matrix (numpy array)
        feature_names: List of feature names
        is_training: Whether this is training data
        engineer: FeatureEngineer instance (for test data)
        
    Returns:
        Engineered feature matrix, feature names, engineer instance
    """
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Initialize or use existing engineer
    if is_training or engineer is None:
        engineer = FeatureEngineer()
    
    # Create features (pass is_training to prevent leakage)
    df_engineered = engineer.create_all_features(df, is_training=is_training)
    
    # Handle inf and nan (pass is_training to prevent leakage)
    df_engineered = engineer.handle_inf_nan(df_engineered, is_training=is_training)
    
    # Get feature names
    engineered_feature_names = df_engineered.columns.tolist()
    
    # Convert back to numpy
    X_engineered = df_engineered.values
    
    return X_engineered, engineered_feature_names, engineer


def engineer_pipeline(input_dir='processed_data', output_dir='engineered_data'):
    """
    Complete feature engineering pipeline
    
    Args:
        input_dir: Directory with preprocessed data
        output_dir: Directory to save engineered data
    """
    from pathlib import Path
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*80)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*80 + "\n")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load(input_path / 'X_train.npy')
    y_train = np.load(input_path / 'y_train.npy')
    X_val = np.load(input_path / 'X_val.npy')
    y_val = np.load(input_path / 'y_val.npy')
    X_test = np.load(input_path / 'X_test.npy')
    test_ids = np.load(input_path / 'test_ids.npy')
    
    # Load feature names
    with open(input_path / 'feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    
    print(f"Loaded data shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Original features: {len(feature_names)}\n")
    
    # Engineer training features
    print("="*80)
    print("ENGINEERING TRAINING FEATURES")
    print("="*80 + "\n")
    X_train_eng, train_feature_names, engineer = engineer_features(
        X_train, feature_names, is_training=True
    )
    
    # Engineer validation features
    print("\n" + "="*80)
    print("ENGINEERING VALIDATION FEATURES")
    print("="*80 + "\n")
    X_val_eng, _, _ = engineer_features(
        X_val, feature_names, is_training=False, engineer=engineer
    )
    
    # Engineer test features
    print("\n" + "="*80)
    print("ENGINEERING TEST FEATURES")
    print("="*80 + "\n")
    X_test_eng, _, _ = engineer_features(
        X_test, feature_names, is_training=False, engineer=engineer
    )
    
    # Save engineered data
    print("\n" + "="*80)
    print("SAVING ENGINEERED DATA")
    print("="*80 + "\n")
    
    np.save(output_path / 'X_train_eng.npy', X_train_eng)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'X_val_eng.npy', X_val_eng)
    np.save(output_path / 'y_val.npy', y_val)
    np.save(output_path / 'X_test_eng.npy', X_test_eng)
    np.save(output_path / 'test_ids.npy', test_ids)
    
    # Save feature names
    with open(output_path / 'feature_names_eng.txt', 'w') as f:
        f.write('\n'.join(train_feature_names))
    
    print(f"Engineered data saved to: {output_path}/")
    print(f"\nFinal feature count: {len(train_feature_names)}")
    print(f"New features created: {len(train_feature_names) - len(feature_names)}")
    
    return {
        'X_train': X_train_eng,
        'y_train': y_train,
        'X_val': X_val_eng,
        'y_val': y_val,
        'X_test': X_test_eng,
        'test_ids': test_ids,
        'feature_names': train_feature_names,
        'engineer': engineer
    }


if __name__ == "__main__":
    # Run feature engineering pipeline
    result = engineer_pipeline(
        input_dir='processed_data',
        output_dir='engineered_data'
    )
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"Training samples: {result['X_train'].shape[0]}")
    print(f"Validation samples: {result['X_val'].shape[0]}")
    print(f"Test samples: {result['X_test'].shape[0]}")
    print(f"Total features: {len(result['feature_names'])}")
    
    print("\n" + "="*80)
    print("NEW FEATURES CREATED")
    print("="*80)
    
    # Load original feature names for comparison
    with open('processed_data/feature_names.txt', 'r') as f:
        original_features = [line.strip() for line in f]
    
    new_features = [f for f in result['feature_names'] if f not in original_features]
    print(f"\nTotal new features: {len(new_features)}\n")
    
    for i, feat in enumerate(new_features, 1):
        print(f"  {i}. {feat}")


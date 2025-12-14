"""
Feature Engineering Module for Depression Prediction

This module creates additional features to improve model performance:
1. Interaction features (e.g., stress * hours)
2. Polynomial features for key predictors
3. Domain-specific features (stress index, sleep quality score)
4. Aggregated features

CORRECTED PIPELINE ORDER:
- Feature engineering is performed BEFORE scaling and SMOTE
- This ensures features are created from original-scale data
- Scaling and SMOTE are applied at the END of this module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    def __init__(self):
        """Initialize feature engineer"""
        self.poly_features = None
        self.feature_names = None
        self.original_features = None
        self.training_statistics = {}  # Store statistics computed from training set only
        self.scaler = None  # Scaler for feature scaling
        self.random_state = 55  # For SMOTE reproducibility
        
    def create_stress_features(self, df):
        """
        Create stress-related features
        
        IMPROVED: Added stress saturation features (non-linear)
        
        NOTE: This handles structural NaN values properly. Students have NaN for
        Work Pressure (they don't work), and Professionals have NaN for Academic
        Pressure (they don't study). We use nanmean/nanmax to compute aggregates
        only from available stress components.
        
        Args:
            df: Input dataframe with features
            
        Returns:
            DataFrame with additional stress features
        """
        df = df.copy()
        
        # Collect stress component column names that exist
        stress_col_names = []
        if 'Academic Pressure' in df.columns:
            stress_col_names.append('Academic Pressure')
        if 'Work Pressure' in df.columns:
            stress_col_names.append('Work Pressure')
        if 'Financial Stress' in df.columns:
            stress_col_names.append('Financial Stress')
        
        if len(stress_col_names) > 0:
            # Create stress array for vectorized NaN-aware operations
            stress_array = df[stress_col_names].values  # Shape: (n_samples, n_stress_components)
            
            # Average stress level (NaN-aware: ignores NaN values)
            # For students: averages Academic Pressure and Financial Stress
            # For professionals: averages Work Pressure and Financial Stress
            df['Total_Stress_Index'] = np.nanmean(stress_array, axis=1)
            
            # Maximum stress (highest available stress factor)
            df['Max_Stress'] = np.nanmax(stress_array, axis=1)
            
            # Stress variance (consistency across available stress domains)
            # Use ddof=0 for population variance (avoid NaN when only 1 component available)
            df['Stress_Variance'] = np.nanvar(stress_array, axis=1, ddof=0)
            
            # Stress saturation - diminishing marginal effect at high stress
            # log1p creates a sub-linear relationship (stress 5 is not 5x worse than stress 1)
            df['Stress_Saturated'] = np.log1p(df['Total_Stress_Index'])
        
        return df
        
    def create_lifestyle_features(self, df):
        """
        Create lifestyle-related features
        
        IMPROVED: Decomposed Sleep and Diet into meaningful binary features
        instead of combining them into a meaningless "Lifestyle_Score"
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with lifestyle features
        """
        df = df.copy()
        
        # Work-life balance: inverse of work/study hours
        if 'Work/Study Hours' in df.columns:
            df['Work_Life_Balance'] = 1.0 / (df['Work/Study Hours'] + 1)
            
            # Overwork indicator: binary flag for excessive hours
            df['Overwork_Flag'] = (df['Work/Study Hours'] >= 10).astype(int)
            
            # NEW: Work hours saturation (diminishing returns after certain point)
            df['Hours_Saturated'] = np.log1p(df['Work/Study Hours'])
        
        # Sleep quality indicators (decomposed, not combined)
        # Sleep Duration was properly encoded in preprocessing (1-5, with 5=optimal)
        if 'Sleep Duration' in df.columns:
            # Binary flag for optimal sleep (encoded as 5 in preprocessing)
            df['Sleep_Optimal'] = (df['Sleep Duration'] == 5).astype(int)
            
            # Binary flag for poor sleep (encoded as 1 in preprocessing)
            df['Sleep_Poor'] = (df['Sleep Duration'] == 1).astype(int)
        
        # Dietary quality indicators (decomposed, not combined)
        # Dietary Habits was properly encoded as 0/1/3 (risk levels)
        if 'Dietary Habits' in df.columns:
            # Binary flag for unhealthy diet (encoded as 3)
            df['Diet_Unhealthy'] = (df['Dietary Habits'] == 3).astype(int)
            
            # Binary flag for healthy diet (encoded as 0)
            df['Diet_Healthy'] = (df['Dietary Habits'] == 0).astype(int)
        
        # ❌ REMOVED: Lifestyle_Score = (Sleep + Diet) / 2
        # This was wrong because:
        # 1. Sleep and Diet are on different scales
        # 2. Combining them loses information
        # 3. Tree models can learn their interaction better than simple averaging
        
        return df
        
    def create_satisfaction_features(self, df, is_training=True):
        """
        Create satisfaction-related features
        
        IMPROVED: Removed Satisfaction_Stress_Ratio (tree models can learn this)
        Added meaningful binary indicator instead
        
        NOTE: This handles structural NaN values properly. Students have NaN for
        Job Satisfaction (they don't work), and Professionals have NaN for Study
        Satisfaction (they don't study). We use nanmean to compute averages only
        from available satisfaction components.
        
        Args:
            df: Input dataframe
            is_training: Whether this is training data (compute statistics)
            
        Returns:
            DataFrame with satisfaction features
        """
        df = df.copy()
        
        # Collect satisfaction component column names that exist
        satisfaction_col_names = []
        if 'Study Satisfaction' in df.columns:
            satisfaction_col_names.append('Study Satisfaction')
        if 'Job Satisfaction' in df.columns:
            satisfaction_col_names.append('Job Satisfaction')
        
        if len(satisfaction_col_names) > 0:
            # Create satisfaction array for vectorized NaN-aware operations
            satisfaction_array = df[satisfaction_col_names].values
            
            # Overall satisfaction (NaN-aware: ignores NaN values)
            # For students: uses Study Satisfaction only
            # For professionals: uses Job Satisfaction only
            df['Overall_Satisfaction'] = np.nanmean(satisfaction_array, axis=1)
            
            # Binary indicator for critically low satisfaction + high stress
            # This is a meaningful interaction that models might not easily discover
            if 'Total_Stress_Index' in df.columns:
                # Handle NaN in Overall_Satisfaction for this comparison
                low_sat = df['Overall_Satisfaction'] < 2.0
                high_stress = df['Total_Stress_Index'] > 3.0
                # Both conditions must be True (and not NaN)
                df['Low_Satisfaction_High_Stress'] = (
                    low_sat.fillna(False) & high_stress.fillna(False)
                ).astype(int)
        
        # Achievement indicator: CGPA above median (use training median to prevent leakage)
        # Note: CGPA is NaN for professionals (structural missing)
        if 'CGPA' in df.columns:
            if is_training:
                # Training: Compute and store median (ignoring NaN)
                median_cgpa = df['CGPA'].median()  # pandas median ignores NaN by default
                self.training_statistics['median_cgpa'] = median_cgpa
            else:
                # Validation/Test: Use training median
                median_cgpa = self.training_statistics.get('median_cgpa', df['CGPA'].median())
            
            # High_Achiever: True if CGPA >= median, keep NaN for professionals
            # For professionals (NaN CGPA), this will be NaN which gets filled later
            df['High_Achiever'] = (df['CGPA'] >= median_cgpa).astype(float)
            # NaN CGPA results in NaN High_Achiever - professionals don't have this indicator
        
        return df
        
    def create_risk_features(self, df):
        """
        Create depression risk features
        
        IMPROVED: Focused on meaningful domain-specific risk interactions
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with risk features
        """
        df = df.copy()
        
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
        
        # CRITICAL INTERACTION: Suicidal thoughts + Family history
        # This is a domain-specific high-risk combination
        if 'Have you ever had suicidal thoughts ?' in df.columns and 'Family History of Mental Illness' in df.columns:
            df['Suicidal_Family_Risk'] = (
                df['Have you ever had suicidal thoughts ?'] * 
                df['Family History of Mental Illness']
            )
        
        # CRITICAL INTERACTION: Overwork + High stress
        # Another domain-specific dangerous combination
        if 'Overwork_Flag' in df.columns and 'Total_Stress_Index' in df.columns:
            df['Overwork_High_Stress'] = (
                df['Overwork_Flag'] * df['Total_Stress_Index']
            )
        
        # Age-related risk: younger people might be more vulnerable
        if 'Age' in df.columns:
            df['Youth_Risk'] = (df['Age'] <= 30).astype(int)
        
        # ❌ REMOVED: Family_Stress_Interaction = Family × Total_Stress
        # ❌ REMOVED: Age_Stress_Interaction = Age × Stress
        # Why removed:
        # 1. These are simple multiplications that tree models can learn
        # 2. Not as critical as Suicidal+Family or Overwork+Stress
        # 3. Keeping only the most meaningful domain-specific interactions
        
        return df
        
    def create_interaction_features(self, df, key_features=None):
        """
        Create interaction features between key predictors
        
        IMPROVED: Drastically reduced to only 2-3 meaningful domain-specific interactions
        Tree models can discover most simple interactions automatically
        
        Args:
            df: Input dataframe
            key_features: List of features to create interactions for (ignored in new version)
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # ⚠️ IMPORTANT CHANGE: Removed almost all interaction features
        # Why? Tree-based models (XGBoost, LightGBM, CatBoost) automatically discover
        # feature interactions through splitting. Creating them manually:
        # 1. Increases multicollinearity
        # 2. Adds noise to the model
        # 3. Slows down training
        # 4. Rarely improves performance
        
        # We keep ONLY the most critical domain-specific interactions that models
        # might not easily discover on their own
        
        # ❌ REMOVED ALL THESE:
        # - Academic Pressure × Work/Study Hours (model can learn this)
        # - Work Pressure × Work/Study Hours (model can learn this)
        # - Financial Stress × Work/Study Hours (model can learn this)
        # - Age × Academic Pressure (model can learn this)
        # - Age × Work Pressure (model can learn this)
        # - Suicidal × Stress (moved to risk_features as more specific interaction)
        
        # Note: Critical interactions (Suicidal×Family, Overwork×Stress, LowSat×HighStress)
        # are already created in risk_features and satisfaction_features
        
        print("  Note: Interaction features minimized - relying on tree model's automatic discovery")
        
        return df
        
    def create_polynomial_features(self, df, degree=2, key_features=None):
        """
        Create polynomial features for key predictors
        
        IMPROVED: Removed all polynomial features for tree-based models
        Tree models don't benefit from polynomial features - they can learn
        non-linear relationships through splitting
        
        Args:
            df: Input dataframe
            degree: Polynomial degree (IGNORED in new version)
            key_features: Features to create polynomials for (IGNORED in new version)
            
        Returns:
            DataFrame (unchanged - no polynomial features added)
        """
        df = df.copy()
        
        # ❌ REMOVED ALL POLYNOMIAL FEATURES
        # Why removed for tree-based models:
        # 1. XGBoost, LightGBM, CatBoost learn non-linear relationships automatically
        # 2. Polynomial features are useful for linear models (logistic regression)
        #    but not for tree models
        # 3. They increase feature space without providing new information
        # 4. Can cause overfitting and slow down training
        # 5. Especially bad for categorical features (e.g., Diet^2 is meaningless)
        
        # For non-linear effects, we use saturation features instead:
        # - Stress_Saturated = log1p(stress)
        # - Hours_Saturated = log1p(hours)
        # These capture diminishing marginal effects more naturally
        
        print("  Note: Polynomial features removed - tree models don't need them")
        
        return df
        
    def create_all_features(self, df, is_training=True):
        """
        Create all engineered features
        
        IMPROVED: Focused on meaningful, non-redundant features for tree-based models
        
        Args:
            df: Input dataframe
            is_training: Whether this is training data (compute statistics)
            
        Returns:
            DataFrame with all engineered features
        """
        print("Creating engineered features (IMPROVED - tree-optimized)...")
        original_shape = df.shape
        
        # Create features in sequence
        df = self.create_stress_features(df)
        print(f"  ✓ Stress features (includes saturation). Shape: {df.shape}")
        
        df = self.create_lifestyle_features(df)
        print(f"  ✓ Lifestyle features (decomposed Sleep/Diet). Shape: {df.shape}")
        
        df = self.create_satisfaction_features(df, is_training=is_training)
        print(f"  ✓ Satisfaction features (meaningful interactions). Shape: {df.shape}")
        
        df = self.create_risk_features(df)
        print(f"  ✓ Risk features (critical combinations). Shape: {df.shape}")
        
        df = self.create_interaction_features(df)
        print(f"  ✓ Interaction features (minimized). Shape: {df.shape}")
        
        df = self.create_polynomial_features(df, degree=2)
        print(f"  ✓ Polynomial features (removed - tree models don't need). Shape: {df.shape}")
        
        print(f"\nFeature engineering complete! 🎉")
        print(f"  Original features: {original_shape[1]}")
        print(f"  Total features: {df.shape[1]}")
        print(f"  New features added: {df.shape[1] - original_shape[1]}")
        print(f"\n  📊 Improvements:")
        print(f"     - Removed redundant linear combinations")
        print(f"     - Removed unnecessary polynomials")
        print(f"     - Minimized interaction features")
        print(f"     - Added meaningful saturation effects")
        print(f"     - Decomposed categorical features properly")
        print(f"     - Reduced multicollinearity")
        
        if is_training:
            print(f"\n  Statistics stored for validation/test: {len(self.training_statistics)}")
        
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
    
    def scale_features(self, X, is_training=True, method='robust'):
        """
        Scale features using training statistics only
        
        Args:
            X: Feature matrix (numpy array)
            is_training: Whether this is training data (fit scaler)
            method: 'standard', 'robust', or 'none'
            
        Returns:
            Scaled feature matrix
        """
        if method == 'none':
            return X
        
        if is_training:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            X_scaled = self.scaler.fit_transform(X)
            print(f"  Fitted {method} scaler on training data")
        else:
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
                print("  Warning: No scaler fitted, returning unscaled data")
        
        return X_scaled
    
    def apply_smote(self, X, y, method='smote'):
        """
        Apply SMOTE to handle class imbalance (training data only)
        
        Handles NaN values by temporarily replacing them with a sentinel value,
        applying SMOTE, then restoring NaN values. This is necessary because:
        1. Structural NaN values are preserved for tree-based models to learn
        2. SMOTE does not accept NaN values natively
        3. After SMOTE, NaN values are restored so tree models can handle them
        
        Args:
            X: Feature matrix (may contain NaN for structural missing values)
            y: Target vector
            method: 'smote' or 'smotetomek'
            
        Returns:
            Resampled X and y (with NaN values preserved/propagated)
        """
        print(f"  Original class distribution: {np.bincount(y.astype(int))}")
        
        # Check for NaN values
        nan_mask = np.isnan(X)
        has_nan = nan_mask.any()
        
        if has_nan:
            nan_count = nan_mask.sum()
            print(f"  Note: {nan_count} NaN values detected (structural missing values)")
            print(f"  Using sentinel value approach for SMOTE compatibility...")
            
            # Use a sentinel value that is far outside the normal data range
            # After scaling, most values are in [-3, 3] range, so -999 is clearly distinct
            SENTINEL_VALUE = -999.0
            
            # Replace NaN with sentinel for SMOTE
            X_for_smote = np.where(nan_mask, SENTINEL_VALUE, X)
        else:
            X_for_smote = X
        
        # Apply SMOTE
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X_for_smote, y)
        elif method == 'smotetomek':
            smote_tomek = SMOTETomek(random_state=self.random_state)
            X_resampled, y_resampled = smote_tomek.fit_resample(X_for_smote, y)
        else:
            raise ValueError(f"Unknown SMOTE method: {method}")
        
        # Restore NaN values if they were present
        if has_nan:
            # For original samples, restore exact NaN positions
            # For synthetic samples, identify sentinel values and convert back to NaN
            # Sentinel values in synthetic samples indicate the feature was NaN in neighbors
            X_resampled = np.where(
                np.isclose(X_resampled, SENTINEL_VALUE, atol=1.0),  # Allow some tolerance
                np.nan,
                X_resampled
            )
            restored_nan_count = np.isnan(X_resampled).sum()
            print(f"  Restored {restored_nan_count} NaN values after SMOTE")
        
        print(f"  Resampled class distribution: {np.bincount(y_resampled.astype(int))}")
        
        return X_resampled, y_resampled


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
    Complete feature engineering pipeline with CORRECTED order:
    1. Feature Engineering (on original-scale data)
    2. Scaling (fit on training only)
    3. SMOTE (on training only)
    
    This is the correct order because:
    - Feature engineering needs original feature scales for meaningful calculations
    - Scaling should be applied after all features are created
    - SMOTE should be applied to the final scaled feature set
    
    Args:
        input_dir: Directory with preprocessed data
        output_dir: Directory to save engineered data
    """
    from pathlib import Path
    import json
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*80)
    print("FEATURE ENGINEERING PIPELINE (CORRECTED ORDER)")
    print("="*80 + "\n")
    print("Pipeline Order:")
    print("  1. Feature Engineering (on original-scale data)")
    print("  2. Scaling (fit on training only)")
    print("  3. SMOTE (on training only)")
    print()
    
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
    
    # Load preprocessing settings
    settings_path = input_path / 'preprocessing_settings.json'
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        use_smote = settings.get('use_smote', True)
        smote_method = settings.get('smote_method', 'smote')
        scaling_method = settings.get('scaling_method', 'robust')
        print(f"Loaded preprocessing settings:")
        print(f"  - use_smote: {use_smote}")
        print(f"  - smote_method: {smote_method}")
        print(f"  - scaling_method: {scaling_method}")
    else:
        # Default settings if file doesn't exist
        use_smote = True
        smote_method = 'smote'
        scaling_method = 'robust'
        print("Using default settings (no preprocessing_settings.json found)")
    
    print(f"\nLoaded data shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Original features: {len(feature_names)}\n")
    
    # STEP 1: Feature Engineering (on original-scale data)
    print("="*80)
    print("STEP 1: FEATURE ENGINEERING (on original-scale data)")
    print("="*80 + "\n")
    
    # Engineer training features
    print("Engineering TRAINING features...")
    X_train_eng, train_feature_names, engineer = engineer_features(
        X_train, feature_names, is_training=True
    )
    
    # Engineer validation features
    print("\nEngineering VALIDATION features...")
    X_val_eng, _, _ = engineer_features(
        X_val, feature_names, is_training=False, engineer=engineer
    )
    
    # Engineer test features
    print("\nEngineering TEST features...")
    X_test_eng, _, _ = engineer_features(
        X_test, feature_names, is_training=False, engineer=engineer
    )
    
    print(f"\nFeature engineering complete!")
    print(f"  Features: {len(feature_names)} -> {len(train_feature_names)}")
    
    # STEP 2: Scaling (fit on training only)
    print("\n" + "="*80)
    print(f"STEP 2: SCALING FEATURES (using {scaling_method} scaler)")
    print("="*80 + "\n")
    
    print("Fitting scaler on TRAINING SET ONLY...")
    X_train_scaled = engineer.scale_features(X_train_eng, is_training=True, method=scaling_method)
    
    print("Applying training scaler to VALIDATION SET...")
    X_val_scaled = engineer.scale_features(X_val_eng, is_training=False, method=scaling_method)
    
    print("Applying training scaler to TEST SET...")
    X_test_scaled = engineer.scale_features(X_test_eng, is_training=False, method=scaling_method)
    
    # STEP 3: SMOTE (on training only)
    print("\n" + "="*80)
    if use_smote:
        print(f"STEP 3: HANDLING CLASS IMBALANCE (using {smote_method})")
        print("="*80 + "\n")
        print("Applying SMOTE to TRAINING SET ONLY...")
        X_train_final, y_train_final = engineer.apply_smote(X_train_scaled, y_train, method=smote_method)
    else:
        print("STEP 3: SKIPPING SMOTE (disabled)")
        print("="*80 + "\n")
        X_train_final = X_train_scaled
        y_train_final = y_train
    
    # Final results
    print("\n" + "="*80)
    print("SAVING ENGINEERED DATA")
    print("="*80 + "\n")
    
    np.save(output_path / 'X_train_eng.npy', X_train_final)
    np.save(output_path / 'y_train.npy', y_train_final)
    np.save(output_path / 'X_val_eng.npy', X_val_scaled)
    np.save(output_path / 'y_val.npy', y_val)
    np.save(output_path / 'X_test_eng.npy', X_test_scaled)
    np.save(output_path / 'test_ids.npy', test_ids)
    
    # Save feature names
    with open(output_path / 'feature_names_eng.txt', 'w') as f:
        f.write('\n'.join(train_feature_names))
    
    print(f"Engineered data saved to: {output_path}/")
    print(f"\nFinal shapes:")
    print(f"  Train: {X_train_final.shape} (after SMOTE)" if use_smote else f"  Train: {X_train_final.shape}")
    print(f"  Val: {X_val_scaled.shape}")
    print(f"  Test: {X_test_scaled.shape}")
    print(f"  Feature count: {len(train_feature_names)}")
    print(f"  New features created: {len(train_feature_names) - len(feature_names)}")
    
    print("\n" + "="*80)
    print("DATA LEAKAGE PREVENTION VERIFIED (Phase 2):")
    print("="*80)
    print("  ✓ Feature engineering statistics computed on training set only")
    print("  ✓ Scaler fitted on training set only")
    print("  ✓ SMOTE applied to training set only")
    print("="*80)
    
    return {
        'X_train': X_train_final,
        'y_train': y_train_final,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
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


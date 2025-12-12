"""
Data Preprocessing Module for Depression Prediction

This module handles:
1. Intelligent missing value imputation (context-aware)
2. Data cleaning (remove noisy entries)
3. Feature encoding (categorical to numerical)
4. Feature scaling and normalization
5. Class imbalance handling (SMOTE, class weights)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, random_state=55, remove_features=[], encoding_method='target'):
        """
        Initialize the data preprocessor
        
        Args:
            random_state: Random seed for reproducibility
            remove_features: List of feature names to remove, or None to use default removal
                           Examples:
                           - ['Name', 'City', 'Degree'] to remove these specific features
                           - [] to keep all features (no removal)
                           - None (default) to use default removal ['Name', 'City', 'Degree']
            encoding_method: Method for encoding high-cardinality categorical features (Profession)
                           - 'target': Target encoding (default) - uses target mean for each category
                           - 'label': Label encoding - simple integer encoding
        """
        self.random_state = random_state
        self.remove_features = remove_features if remove_features is not None else ['Name', 'City', 'Degree']
        self.encoding_method = encoding_method  # 'target' or 'label'
        self.label_encoders = {}
        self.target_encoders = {}  # Store target encoding mappings
        self.scaler = None
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        self.imputation_values = {}  # Store imputation statistics from training set
    
    def drop_irrelevant_columns(self, df):
        """
        Drop columns specified in self.remove_features configuration
        
        By default removes:
        - Name: Personal identifiers with no causal relationship to mental health
        - City: High-cardinality geographic data, signal captured by other features
        - Degree: High-cardinality educational data, signal captured by Academic Pressure and CGPA
        
        These features add noise and increase dimensionality without meaningful predictive power.
        
        Can be controlled via remove_features parameter in __init__:
        - Set to [] to keep all features
        - Set to custom list to remove specific features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with specified columns removed
        """
        df = df.copy()
        
        # If no features to remove, return dataframe as-is
        if not self.remove_features:
            print(f"  Feature removal disabled - keeping all features")
            return df
        
        # Drop configured columns that exist in the dataframe
        existing_columns = [col for col in self.remove_features if col in df.columns]
        if existing_columns:
            df = df.drop(columns=existing_columns)
            print(f"  Dropped configured columns: {existing_columns}")
        else:
            print(f"  No configured columns found to drop")
        
        return df
    
    def target_encode_fit(self, df, target_col, categorical_col, smoothing=10):
        """
        Apply target encoding directly on training data (no K-Fold CV needed)
        
        Since we now split data BEFORE encoding, we can directly compute target means
        from the training data without any CV - there's no leakage because validation
        data is not included in this computation.
        
        This method is CRITICAL for high-cardinality categorical features like Profession.
        It encodes each category with the target mean (depression rate) with smoothing.
        
        Args:
            df: Training dataframe with both features and target
            target_col: Name of target column
            categorical_col: Name of categorical column to encode
            smoothing: Smoothing parameter for rare categories (default 10)
            
        Returns:
            Encoded series, encoding_map for applying to validation/test
        """
        # Compute global mean
        global_mean = df[target_col].mean()
        
        # Group by category and calculate mean + count
        category_stats = df.groupby(categorical_col)[target_col].agg(['mean', 'count'])
        
        # Apply smoothing (Bayesian average)
        # Formula: (category_mean * count + global_mean * smoothing) / (count + smoothing)
        category_stats['smoothed_mean'] = (
            (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
            (category_stats['count'] + smoothing)
        )
        
        # Create encoding map
        encoding_map = category_stats['smoothed_mean'].to_dict()
        encoding_map['__GLOBAL_MEAN__'] = global_mean  # Store global mean for unseen categories
        
        # Apply encoding to training data
        encoded = df[categorical_col].map(encoding_map).fillna(global_mean)
        
        return encoded, encoding_map
    
    def target_encode_with_cv(self, df, target_col, categorical_col, n_splits=5, smoothing=10):
        """
        Apply target encoding with K-Fold cross-validation to prevent data leakage
        
        NOTE: This method is kept for backward compatibility but is no longer used
        in the main pipeline since we now split data BEFORE encoding.
        
        This method is CRITICAL for high-cardinality categorical features like Profession.
        It encodes each category with the target mean (depression rate) using proper CV
        to avoid information leakage.
        
        Strategy:
        1. Use K-Fold CV on training data
        2. For each fold, compute target mean on OTHER folds only
        3. Apply smoothing for rare categories (Bayesian average)
        4. Store global mean for test data
        
        Args:
            df: Dataframe with both features and target
            target_col: Name of target column
            categorical_col: Name of categorical column to encode
            n_splits: Number of CV folds (default 5)
            smoothing: Smoothing parameter for rare categories (default 10)
            
        Returns:
            Encoded series for the categorical column
        """
        # Initialize encoded column with global mean
        global_mean = df[target_col].mean()
        encoded = pd.Series(index=df.index, data=global_mean, dtype=np.float64)
        
        # Perform K-Fold encoding
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        for train_idx, val_idx in kf.split(df):
            # Calculate target mean for each category using ONLY training fold
            train_fold = df.iloc[train_idx]
            
            # Group by category and calculate mean + count
            category_stats = train_fold.groupby(categorical_col)[target_col].agg(['mean', 'count'])
            
            # Apply smoothing (Bayesian average)
            # Formula: (category_mean * count + global_mean * smoothing) / (count + smoothing)
            category_stats['smoothed_mean'] = (
                (category_stats['mean'] * category_stats['count'] + global_mean * smoothing) /
                (category_stats['count'] + smoothing)
            )
            
            # Map to validation fold
            val_categories = df.iloc[val_idx][categorical_col]
            encoded.iloc[val_idx] = val_categories.map(category_stats['smoothed_mean']).fillna(global_mean)
        
        # Store encoding mapping for test data (using entire training set)
        category_stats_full = df.groupby(categorical_col)[target_col].agg(['mean', 'count'])
        category_stats_full['smoothed_mean'] = (
            (category_stats_full['mean'] * category_stats_full['count'] + global_mean * smoothing) /
            (category_stats_full['count'] + smoothing)
        )
        
        encoding_map = category_stats_full['smoothed_mean'].to_dict()
        encoding_map['__GLOBAL_MEAN__'] = global_mean  # Store global mean for unseen categories
        
        return encoded, encoding_map
    
    def apply_target_encoding(self, df, categorical_col, encoding_map):
        """
        Apply pre-computed target encoding to test data
        
        Args:
            df: Dataframe to encode
            categorical_col: Name of categorical column
            encoding_map: Dictionary with encoding values from training
            
        Returns:
            Encoded series
        """
        global_mean = encoding_map.get('__GLOBAL_MEAN__', 0.5)
        encoded = df[categorical_col].map(encoding_map).fillna(global_mean)
        return encoded
    
    def remove_anomalous_missing_values(self, df):
        """
        Remove rows with EXTREME anomalous missing values (VERY CONSERVATIVE)
        
        Only remove rows that are completely corrupted or have too many missing critical fields
        Most missing values will be handled by imputation instead of deletion
        
        Conservative strategy:
        - Only remove rows with >= 50% missing values across ALL columns
        - Keep almost all data for imputation
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe, number of rows removed
        """
        df = df.copy()
        original_size = len(df)
        
        # Calculate missing value percentage per row
        missing_pct = df.isnull().sum(axis=1) / len(df.columns)
        
        # Only remove rows with >= 50% missing values (VERY conservative)
        anomalous_mask = missing_pct >= 0.5
        
        # Remove anomalous rows
        df_cleaned = df[~anomalous_mask].copy()
        removed_count = original_size - len(df_cleaned)
        
        if removed_count > 0:
            print(f"  Removed {removed_count:,} rows with >= 50% missing values (extremely corrupted data)")
            print(f"  Remaining: {len(df_cleaned):,} rows ({100*len(df_cleaned)/original_size:.2f}%)")
        else:
            print(f"  No extremely corrupted rows found. Keeping all {len(df):,} rows.")
        
        return df_cleaned, removed_count
    
    def clean_data_quality_issues(self, df):
        """
        Clean data quality issues in specific columns (CONSERVATIVE STRATEGY)
        
        Philosophy: Only remove OBVIOUS errors, preserve all potentially valid data
        - Use exact match for invalid values, NOT substring matching
        - Prefer setting to NaN rather than deleting rows
        - Let imputation handle the NaN values
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        print("  Cleaning data quality issues (conservative strategy)...")
        
        # 1. Clean Sleep Duration - keep only valid categories
        if 'Sleep Duration' in df.columns:
            valid_sleep = ['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', 'More than 8 hours']
            invalid_sleep = ~df['Sleep Duration'].isin(valid_sleep) & df['Sleep Duration'].notna()
            if invalid_sleep.sum() > 0:
                print(f"    Sleep Duration: Setting {invalid_sleep.sum()} invalid values to NaN")
                df.loc[invalid_sleep, 'Sleep Duration'] = np.nan
        
        # 2. Clean Dietary Habits - keep only valid categories
        if 'Dietary Habits' in df.columns:
            valid_diet = ['Healthy', 'Moderate', 'Unhealthy']
            invalid_diet = ~df['Dietary Habits'].isin(valid_diet) & df['Dietary Habits'].notna()
            if invalid_diet.sum() > 0:
                print(f"    Dietary Habits: Setting {invalid_diet.sum()} invalid values to NaN")
                df.loc[invalid_diet, 'Dietary Habits'] = np.nan
        
        # 3. Clean Profession - CONSERVATIVE: Use EXACT MATCH only
        #    Only remove values that are EXACTLY a city/degree/gender name
        #    This preserves 'HR Manager', 'Marketing Manager', 'Pharmacist', etc.
        if 'Profession' in df.columns:
            # EXACT match for obviously wrong values
            exact_invalid_professions = [
                # Degree names (exact match)
                'Class', 'BSc', 'MSc', 'BA', 'MA', 'BBA', 'MBA', 'B.Com', 'M.Com',
                'BCA', 'MCA', 'BE', 'ME', 'BTech', 'MTech', 'B.Tech', 'M.Tech',
                'LLB', 'LLM', 'MD', 'MBBS', 'BDS', 'PhD',
                # Cities
                'City', 'Bhopal', 'Indore', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai',
                # Genders
                'Gender', 'Male', 'Female', 'M', 'F',
                # Ages/Numbers
                'Age', '18', '19', '20', '21', '22', '23', '24', '25',
                # Family/Personal
                'Family', 'Father', 'Mother', 'Brother', 'Sister',
                # Names (common Indian names that appear as errors)
                'Yuvraj', 'Navya', 'Kavya', 'Rohan', 'Priya', 'Ananya', 'Arjun',
                'Soham', 'Aarav', 'Vivaan', 'Aditya', 'Naina', 'Ishaan', 'Krishna',
                # Other
                'Student', 'Yes', 'No', 'None', 'Unknown', 'NA', 'N/A'
            ]
            
            # Use exact match (isin) instead of contains
            invalid_profession = df['Profession'].isin(exact_invalid_professions)
            if invalid_profession.sum() > 0:
                print(f"    Profession: Setting {invalid_profession.sum()} exact-match invalid values to NaN")
                df.loc[invalid_profession, 'Profession'] = np.nan
        
        print("  Data quality cleaning completed (conservative approach)")
        return df
        
    def handle_missing_values(self, df, is_training=True, imputation_values=None):
        """
        Intelligently handle missing values based on context (IMPROVED STRATEGY)
        
        Key improvements:
        1. Use MEDIAN (not mean) for numerical features (more robust to skewness)
        2. Profession handling: Students -> 'Student', Professionals -> 'Unknown'
        3. More conservative imputation strategy
        
        CRITICAL: If is_training=True, compute statistics and store in self.imputation_values
                  If is_training=False, use provided imputation_values or self.imputation_values
        
        This prevents data leakage by ensuring validation/test sets use only training statistics
        
        Args:
            df: Input dataframe
            is_training: Whether this is training data (compute new statistics)
            imputation_values: Pre-computed imputation values (for validation/test)
            
        Returns:
            DataFrame with imputed missing values
        """
        df = df.copy()
        
        # Separate students and working professionals
        is_student = df['Working Professional or Student'] == 'Student'
        is_professional = df['Working Professional or Student'] == 'Working Professional'
        
        if is_training:
            # Training mode: Compute statistics and store them
            print("  Computing imputation statistics from training data...")
            self.imputation_values = {}
            
            # For Students: Work-related features should be 0 (not applicable)
            student_work_features = ['Work Pressure', 'Job Satisfaction']
            for feature in student_work_features:
                if feature in df.columns:
                    self.imputation_values[f'{feature}_student'] = 0.0
            
            # For Professionals: Academic features should be 0 (not applicable)
            prof_academic_features = ['Academic Pressure', 'Study Satisfaction', 'CGPA']
            for feature in prof_academic_features:
                if feature in df.columns:
                    self.imputation_values[f'{feature}_professional'] = 0.0
            
            # For Students with missing CGPA: use MEDIAN from training students
            if 'CGPA' in df.columns:
                student_cgpa_median = df.loc[is_student, 'CGPA'].median()
                self.imputation_values['CGPA_student'] = student_cgpa_median
                print(f"    Student CGPA median: {student_cgpa_median:.2f}")
            
            # Numerical features: compute MEDIAN from training data (robust to outliers)
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numerical_cols:
                if col not in ['id', 'Depression']:
                    median_val = df[col].median()
                    self.imputation_values[f'{col}_median'] = median_val
            
            # Categorical features: compute MODE from training data
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in ['id', 'Name', 'Profession']:  # Profession handled separately
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        self.imputation_values[f'{col}_mode'] = mode_value[0]
                    else:
                        self.imputation_values[f'{col}_mode'] = 'Unknown'
            
            # Profession: compute mode for professionals only
            if 'Profession' in df.columns:
                prof_mode = df.loc[is_professional, 'Profession'].mode()
                if len(prof_mode) > 0:
                    self.imputation_values['Profession_professional_mode'] = prof_mode[0]
                else:
                    self.imputation_values['Profession_professional_mode'] = 'Unknown'
            
            imputation_values = self.imputation_values
            print(f"  Computed imputation values for {len(self.imputation_values)} statistics")
        else:
            # Validation/Test mode: Use provided statistics
            if imputation_values is None:
                imputation_values = self.imputation_values
            if not imputation_values:
                raise ValueError("No imputation values available for test/validation data")
        
        # Apply imputation using the computed (or provided) statistics
        print("  Applying imputation...")
        
        # For Students: Fill work-related features with 0
        student_work_features = ['Work Pressure', 'Job Satisfaction']
        for feature in student_work_features:
            if feature in df.columns:
                df.loc[is_student, feature] = df.loc[is_student, feature].fillna(
                    imputation_values.get(f'{feature}_student', 0.0)
                )
        
        # For Students: Fill missing Profession with 'Student'
        if 'Profession' in df.columns:
            student_prof_missing = is_student & df['Profession'].isnull()
            if student_prof_missing.sum() > 0:
                print(f"    Filling {student_prof_missing.sum()} missing Profession for Students -> 'Student'")
                df.loc[student_prof_missing, 'Profession'] = 'Student'
        
        # For Professionals: Fill academic-related features with 0
        prof_academic_features = ['Academic Pressure', 'Study Satisfaction', 'CGPA']
        for feature in prof_academic_features:
            if feature in df.columns:
                df.loc[is_professional, feature] = df.loc[is_professional, feature].fillna(
                    imputation_values.get(f'{feature}_professional', 0.0)
                )
        
        # For Professionals: Fill missing Profession with mode or 'Unknown'
        if 'Profession' in df.columns:
            prof_prof_missing = is_professional & df['Profession'].isnull()
            if prof_prof_missing.sum() > 0:
                fill_value = imputation_values.get('Profession_professional_mode', 'Unknown')
                print(f"    Filling {prof_prof_missing.sum()} missing Profession for Professionals -> '{fill_value}'")
                df.loc[prof_prof_missing, 'Profession'] = fill_value
        
        # Handle anomalous missing values in CGPA for students (use training median)
        if 'CGPA' in df.columns:
            student_cgpa_fill = imputation_values.get('CGPA_student', 0.0)
            student_cgpa_missing = is_student & df['CGPA'].isnull()
            if student_cgpa_missing.sum() > 0:
                print(f"    Filling {student_cgpa_missing.sum()} missing CGPA for Students -> {student_cgpa_fill:.2f}")
                df.loc[student_cgpa_missing, 'CGPA'] = student_cgpa_fill
        
        # Fill remaining missing values using training statistics
        # Numerical features: use training MEDIAN (robust to outliers)
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if col not in ['id', 'Depression']:
                fill_value = imputation_values.get(f'{col}_median', 0.0)
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    print(f"    Filling {missing_count} missing {col} -> median: {fill_value:.2f}")
                    df[col] = df[col].fillna(fill_value)
        
        # Categorical features: use training mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['id', 'Name', 'Profession']:  # Profession already handled
                fill_value = imputation_values.get(f'{col}_mode', 'Unknown')
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    print(f"    Filling {missing_count} missing {col} -> mode: '{fill_value}'")
                    df[col] = df[col].fillna(fill_value)
        
        print("  Imputation completed")
        return df
    
    def handle_outliers(self, df, is_training=True, clip_values=None):
        """
        Handle outliers using IQR-based clipping (CONSERVATIVE)
        
        Only clip extreme outliers in specific numerical features
        - Work/Study Hours: Clip to reasonable range
        - CGPA: Clip to valid range (0-10)
        - Financial Stress: Already in 1-5 range
        
        Uses clipping (not deletion) to preserve data
        
        Args:
            df: Input dataframe
            is_training: Whether this is training data
            clip_values: Pre-computed clipping bounds (for validation/test)
            
        Returns:
            DataFrame with clipped outliers
        """
        df = df.copy()
        
        # Features to apply IQR clipping
        features_to_clip = ['Work/Study Hours', 'CGPA']
        
        if is_training:
            print("  Computing outlier clipping bounds from training data...")
            self.clip_values = {}
            
            for feature in features_to_clip:
                if feature in df.columns:
                    Q1 = df[feature].quantile(0.25)
                    Q3 = df[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Use 3*IQR for very conservative clipping (only extreme outliers)
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    # Additional bounds for specific features
                    if feature == 'CGPA':
                        lower_bound = max(lower_bound, 0.0)  # CGPA cannot be negative
                        upper_bound = min(upper_bound, 10.0)  # CGPA max is 10
                    elif feature == 'Work/Study Hours':
                        lower_bound = max(lower_bound, 0.0)  # Hours cannot be negative
                        upper_bound = min(upper_bound, 24.0)  # Max hours in a day (24 hours)
                    
                    self.clip_values[feature] = (lower_bound, upper_bound)
                    print(f"    {feature}: clip to [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            clip_values = self.clip_values
        else:
            # Validation/Test mode: Use provided bounds
            if clip_values is None:
                clip_values = self.clip_values
            if not clip_values:
                print("  Warning: No clipping values available, skipping outlier handling")
                return df
        
        # Apply clipping
        print("  Applying outlier clipping...")
        for feature in features_to_clip:
            if feature in df.columns and feature in clip_values:
                lower, upper = clip_values[feature]
                original_outliers = ((df[feature] < lower) | (df[feature] > upper)).sum()
                
                if original_outliers > 0:
                    df[feature] = df[feature].clip(lower=lower, upper=upper)
                    print(f"    {feature}: Clipped {original_outliers} outliers")
        
        print("  Outlier handling completed")
        return df
        
    def encode_categorical_features(self, df, fit=True, target_col='Depression'):
        """
        Encode categorical features to numerical
        
        CONFIGURABLE STRATEGY:
        - Ordinal encoding for Sleep Duration and Dietary Habits (meaningful order)
        - Target encoding OR Label encoding for Profession (configurable via self.encoding_method)
        - Label encoding for Gender and Working Professional or Student (low cardinality)
        
        The encoding method for Profession can be configured:
        - 'target': Target encoding - uses target mean for each category (default)
        - 'label': Label encoding - simple integer encoding
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for test)
            target_col: Target column name (needed for target encoding)
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        # 1. ORDINAL ENCODING for Sleep Duration (U-shaped relationship)
        # 7-8 hours is optimal, too little or too much is bad
        if 'Sleep Duration' in df.columns:
            sleep_quality_map = {
                'Less than 5 hours': 1,     # Poor quality (severe sleep deprivation)
                '5-6 hours': 2,              # Below optimal (insufficient sleep)
                '6-7 hours': 4,              # Good (slightly below optimal)
                '7-8 hours': 5,              # Optimal (best for most people)
                'More than 8 hours': 2       # Excessive (moderately suboptimal, possible oversleeping)
            }
            df['Sleep Duration'] = df['Sleep Duration'].map(sleep_quality_map)
            print(f"  Encoded 'Sleep Duration' with quality mapping (1=poor, 5=optimal)")
        
        # 2. ORDINAL ENCODING for Dietary Habits (risk-based, non-linear)
        # Not linear: Unhealthy is much worse than the gap between Healthy and Moderate
        if 'Dietary Habits' in df.columns:
            diet_risk_map = {
                'Healthy': 0,      # Low risk
                'Moderate': 1,     # Medium risk
                'Unhealthy': 3     # High risk (3x, not 2x, to show severity)
            }
            df['Dietary Habits'] = df['Dietary Habits'].map(diet_risk_map)
            print(f"  Encoded 'Dietary Habits' with risk-based mapping (0=healthy, 3=unhealthy)")
        
        # 3. PROFESSION ENCODING - CONFIGURABLE (target or label)
        # High cardinality: 64 unique values
        if 'Profession' in df.columns:
            if self.encoding_method == 'target':
                # TARGET ENCODING: Uses target mean for each category
                # Better for tree-based models, captures relationship with target
                if fit:
                    if target_col in df.columns:
                        print(f"  Applying TARGET ENCODING to 'Profession' (fit on training data)...")
                        encoded_profession, encoding_map = self.target_encode_fit(
                            df, target_col, 'Profession', smoothing=10
                        )
                        df['Profession'] = encoded_profession
                        self.target_encoders['Profession'] = encoding_map
                        print(f"    Encoded {len(encoding_map)-1} unique professions with target means")
                    else:
                        # Fallback if target not available
                        print(f"  WARNING: Target column not found, using LabelEncoder for Profession")
                        le = LabelEncoder()
                        df['Profession'] = df['Profession'].fillna('Unknown')
                        df['Profession'] = le.fit_transform(df['Profession'].astype(str))
                        self.label_encoders['Profession'] = le
                else:
                    # Test/Validation: Apply pre-computed target encoding
                    if 'Profession' in self.target_encoders:
                        print(f"  Applying pre-computed TARGET ENCODING to 'Profession'...")
                        df['Profession'] = self.apply_target_encoding(
                            df, 'Profession', self.target_encoders['Profession']
                        )
                    else:
                        print(f"  WARNING: No target encoding found, setting Profession to 0")
                        df['Profession'] = 0
            else:
                # LABEL ENCODING: Simple integer encoding
                # Simpler but doesn't capture relationship with target
                if fit:
                    print(f"  Applying LABEL ENCODING to 'Profession' (fit on training data)...")
                    le = LabelEncoder()
                    df['Profession'] = df['Profession'].fillna('Unknown').astype(str)
                    df['Profession'] = le.fit_transform(df['Profession'])
                    self.label_encoders['Profession'] = le
                    print(f"    Encoded {len(le.classes_)} unique professions with integer labels")
                else:
                    # Test/Validation: Apply pre-computed label encoding
                    if 'Profession' in self.label_encoders:
                        print(f"  Applying pre-computed LABEL ENCODING to 'Profession'...")
                        le = self.label_encoders['Profession']
                        df['Profession'] = df['Profession'].fillna('Unknown').astype(str)
                        df['Profession'] = df['Profession'].apply(
                            lambda x: x if x in le.classes_ else 'Unknown'
                        )
                        if 'Unknown' not in le.classes_:
                            le.classes_ = np.append(le.classes_, 'Unknown')
                        df['Profession'] = le.transform(df['Profession'])
                    else:
                        print(f"  WARNING: No label encoding found, setting Profession to 0")
                        df['Profession'] = 0
        
        # 4. LABEL ENCODING for low-cardinality categorical features
        # (Gender, Working Professional or Student - only 2 values each)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        # Exclude id, target, already-encoded features, and Profession
        categorical_cols = [col for col in categorical_cols 
                          if col not in ['id', 'Depression', 'Sleep Duration', 'Dietary Habits', 'Profession']]
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    # Add 'Unknown' to classes if needed
                    if 'Unknown' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'Unknown')
                    df[col] = le.transform(df[col])
                else:
                    df[col] = 0
        
        self.categorical_features = categorical_cols + ['Sleep Duration', 'Dietary Habits']
        return df
        
    def create_validation_split(self, X, y, val_size=0.2, stratify=True):
        """
        Create train/validation split with stratification
        
        Args:
            X: Features
            y: Target
            val_size: Validation set size (default 0.2)
            stratify: Whether to use stratified split
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        if stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state
            )
        
        return X_train, X_val, y_train, y_val
        
    def handle_class_imbalance(self, X, y, method='smote'):
        """
        Handle class imbalance using SMOTE or SMOTETomek
        
        Args:
            X: Feature matrix
            y: Target vector
            method: 'smote' or 'smotetomek'
            
        Returns:
            Resampled X and y
        """
        print(f"Original class distribution: {np.bincount(y)}")
        
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif method == 'smotetomek':
            smote_tomek = SMOTETomek(random_state=self.random_state)
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
        
    def scale_features(self, df, fit=True, method='robust'):
        """
        Scale numerical features
        
        Args:
            df: Input dataframe
            fit: Whether to fit scaler
            method: 'standard', 'robust', or 'none'
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Exclude id and target
        numerical_cols = [col for col in numerical_cols if col not in ['id', 'Depression']]
        
        # If no scaling is requested, just store numerical features and return
        if method == 'none':
            self.scaler = None
            self.numerical_features = numerical_cols
            return df
        
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            if self.scaler is not None:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        self.numerical_features = numerical_cols
        return df
        
    def prepare_features(self, df, target_col='Depression', drop_cols=None):
        """
        Prepare feature matrix and target vector
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            drop_cols: Additional columns to drop
            
        Returns:
            X (features), y (target if exists)
        """
        df = df.copy()
        
        # Default columns to drop
        default_drop = ['id', 'Name']
        if drop_cols:
            default_drop.extend(drop_cols)
        
        # Drop specified columns
        df = df.drop(columns=[col for col in default_drop if col in df.columns])
        
        # Separate features and target
        if target_col in df.columns:
            y = df[target_col].values
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df
        
        self.feature_names = X.columns.tolist()
        
        return X, y
        
    def preprocess_train(self, df, use_smote=True, smote_method='smote', 
                        val_size=0.2, scaling_method='robust'):
        """
        Complete preprocessing pipeline for training data with NO DATA LEAKAGE (FIXED)
        
        CRITICAL ORDER TO PREVENT LEAKAGE (Corrected Flow):
        1. Remove ONLY extremely corrupted rows (>= 50% missing)
        2. Clean data quality issues (conservative, exact match only)
        3. SPLIT DATA FIRST (before any encoding that uses target!)
        4. Encode categorical features (fit on training only, apply to validation)
        5. Handle outliers (IQR clipping - fit on training set only)
        6. Compute imputation statistics ONLY on training set
        7. Apply imputation to both train and validation separately
        
        NOTE: Scaling and SMOTE are moved to AFTER feature engineering
        This is the correct order because:
        - Feature engineering needs original feature scales for meaningful calculations
        - SMOTE should be applied to the final feature set
        
        Args:
            df: Training dataframe
            use_smote: Whether to apply SMOTE (will be stored for later use)
            smote_method: SMOTE variant to use (will be stored for later use)
            val_size: Validation split size
            scaling_method: Scaling method to use (will be stored for later use)
            
        Returns:
            Dictionary containing preprocessed train/val sets (BEFORE scaling and SMOTE)
        """
        print("=" * 80)
        print("STARTING TRAINING DATA PREPROCESSING (DATA LEAKAGE FIXED)")
        print("=" * 80)
        print(f"Original shape: {df.shape}")
        print("\n⚠️  NOTE: Scaling and SMOTE will be applied AFTER feature engineering")
        print("    This is the correct order for proper feature construction.\n")
        
        # Store SMOTE and scaling settings for later use in feature engineering pipeline
        self.use_smote = use_smote
        self.smote_method = smote_method
        self.scaling_method = scaling_method
        
        # Step 0: Drop configured columns (configurable via remove_features parameter)
        print("\n" + "=" * 80)
        print("STEP 0: Feature removal (configurable)")
        print("=" * 80)
        df = self.drop_irrelevant_columns(df)
        print(f"  Shape after feature removal: {df.shape}")
        
        # Step 1: Remove ONLY extremely corrupted rows (>= 50% missing)
        print("\n" + "=" * 80)
        print("STEP 1: Removing extremely corrupted rows (>= 50% missing)")
        print("=" * 80)
        df, removed_count = self.remove_anomalous_missing_values(df)
        
        # Step 2: Clean data quality issues (conservative - exact match only)
        print("\n" + "=" * 80)
        print("STEP 2: Cleaning data quality issues (conservative strategy)")
        print("=" * 80)
        df = self.clean_data_quality_issues(df)
        
        # Step 3: SPLIT DATA FIRST - This is critical to prevent leakage
        # Must split BEFORE encoding to prevent target leakage from validation data
        print("\n" + "=" * 80)
        print(f"STEP 3: Creating train/validation split FIRST (val_size={val_size})")
        print("=" * 80)
        print("  ⚠️  IMPORTANT: Splitting BEFORE encoding to prevent data leakage!")
        
        # Extract target before splitting
        y = df['Depression'].values
        
        # Use stratified split
        train_idx, val_idx = train_test_split(
            np.arange(len(df)), test_size=val_size, 
            random_state=self.random_state, stratify=y
        )
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        print(f"  Train set shape: {train_df.shape}")
        print(f"  Validation set shape: {val_df.shape}")
        print(f"  Train target distribution: {np.bincount(y_train)}")
        print(f"  Val target distribution: {np.bincount(y_val)}")
        
        # Step 4: Encode categorical features (fit on training only)
        # Target encoding uses only training data's target values
        print("\n" + "=" * 80)
        print("STEP 4: Encoding categorical features (fit on TRAINING only)")
        print("=" * 80)
        print("  Fitting encoders on TRAINING SET ONLY...")
        train_df = self.encode_categorical_features(train_df, fit=True, target_col='Depression')
        print(f"  Training encoded. Shape: {train_df.shape}")
        
        print("  Applying encoders to VALIDATION SET...")
        val_df = self.encode_categorical_features(val_df, fit=False, target_col='Depression')
        print(f"  Validation encoded. Shape: {val_df.shape}")
        
        # Step 5: Prepare features
        print("\n" + "=" * 80)
        print("STEP 5: Preparing features")
        print("=" * 80)
        X_train, _ = self.prepare_features(train_df, target_col='Depression')
        
        # Store feature names from training data
        train_feature_names = self.feature_names.copy()
        
        X_val, _ = self.prepare_features(val_df, target_col='Depression')
        
        # Ensure feature names are consistent
        self.feature_names = train_feature_names
        
        print(f"  Feature matrix shape: {X_train.shape}")
        print(f"  Number of features: {len(self.feature_names)}")
        
        # Convert to DataFrames for processing
        X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
        X_val_df = pd.DataFrame(X_val, columns=self.feature_names)
        
        # Step 6: Handle outliers - FIT ON TRAINING SET ONLY
        print("\n" + "=" * 80)
        print("STEP 6: Handling outliers (IQR clipping)")
        print("=" * 80)
        print("  Computing clipping bounds from TRAINING SET ONLY...")
        X_train_df = self.handle_outliers(X_train_df, is_training=True)
        
        print("  Applying training bounds to VALIDATION SET...")
        X_val_df = self.handle_outliers(X_val_df, is_training=False, 
                                        clip_values=self.clip_values)
        
        # Step 7: Handle missing values - COMPUTE ON TRAINING SET ONLY
        print("\n" + "=" * 80)
        print("STEP 7: Handling missing values (median/mode imputation)")
        print("=" * 80)
        X_train_df = self.handle_missing_values(X_train_df, is_training=True)
        print(f"  Training missing values after imputation: {X_train_df.isnull().sum().sum()}")
        
        print("  Applying training statistics to VALIDATION SET...")
        X_val_df = self.handle_missing_values(X_val_df, is_training=False, 
                                               imputation_values=self.imputation_values)
        print(f"  Validation missing values after imputation: {X_val_df.isnull().sum().sum()}")
        
        # NOTE: Scaling and SMOTE are NOT done here anymore
        # They will be applied AFTER feature engineering in the engineer_pipeline
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE (Phase 1 - Before Feature Engineering)")
        print("=" * 80)
        print(f"Training set shape: {X_train_df.shape}")
        print(f"Validation set shape: {X_val_df.shape}")
        print("\n" + "=" * 80)
        print("DATA LEAKAGE PREVENTION VERIFIED (Phase 1):")
        print("=" * 80)
        print("  ✓ Only extremely corrupted rows removed (>= 50% missing)")
        print("  ✓ Conservative data cleaning (exact match only)")
        print("  ✓ Train/val split done BEFORE encoding (target leakage prevented)")
        print("  ✓ Target encoding fitted on training set only")
        print("  ✓ Outlier clipping bounds fitted on training set only")
        print("  ✓ Imputation statistics computed on training set only")
        print("\n⚠️  Scaling and SMOTE will be applied AFTER feature engineering")
        print("=" * 80)
        
        return {
            'X_train': X_train_df.values,
            'y_train': y_train,
            'X_val': X_val_df.values,
            'y_val': y_val,
            'feature_names': self.feature_names,
            'imputation_values': self.imputation_values,
            'removed_anomalies': removed_count,
            # Pass settings for feature engineering pipeline
            'use_smote': use_smote,
            'smote_method': smote_method,
            'scaling_method': scaling_method
        }
        
    def preprocess_test(self, df, scaling_method='robust'):
        """
        Complete preprocessing pipeline for test data (IMPROVED)
        Uses statistics computed from training data to prevent leakage
        
        Uses conservative cleaning strategy and training statistics for all transformations
        
        NOTE: Scaling is NOT applied here anymore - it will be done after feature engineering
        
        Args:
            df: Test dataframe
            scaling_method: Scaling method (stored for later use)
            
        Returns:
            Preprocessed feature matrix (BEFORE scaling) and ids
        """
        print("=" * 80)
        print("STARTING TEST DATA PREPROCESSING (CORRECTED FLOW)")
        print("=" * 80)
        print(f"Original shape: {df.shape}")
        print("\n⚠️  NOTE: Scaling will be applied AFTER feature engineering")
        
        # Save ids for submission
        test_ids = df['id'].values
        
        # Store scaling method for later use
        self.scaling_method = scaling_method
        
        # Step 0: Drop configured columns (configurable via remove_features parameter)
        print("\n" + "=" * 80)
        print("STEP 0: Feature removal (configurable)")
        print("=" * 80)
        df = self.drop_irrelevant_columns(df)
        print(f"  Shape after feature removal: {df.shape}")
        
        # Step 1: Clean data quality issues (conservative - exact match only)
        print("\n" + "=" * 80)
        print("STEP 1: Cleaning data quality issues (conservative strategy)")
        print("=" * 80)
        df = self.clean_data_quality_issues(df)
        
        # Step 2: Encode categorical features (using training encoders)
        print("\n" + "=" * 80)
        print("STEP 2: Encoding categorical features (using training encoders)")
        print("=" * 80)
        df = self.encode_categorical_features(df, fit=False)
        print(f"  Encoded features. Shape: {df.shape}")
        
        # Step 3: Prepare features
        print("\n" + "=" * 80)
        print("STEP 3: Preparing features")
        print("=" * 80)
        X, _ = self.prepare_features(df, target_col='Depression')
        print(f"  Feature matrix shape: {X.shape}")
        
        # Step 4: Handle outliers (using TRAINING clipping bounds)
        print("\n" + "=" * 80)
        print("STEP 4: Handling outliers (using training bounds)")
        print("=" * 80)
        X_df = pd.DataFrame(X, columns=self.feature_names)
        X_df = self.handle_outliers(X_df, is_training=False, 
                                    clip_values=self.clip_values)
        
        # Step 5: Handle missing values (using TRAINING statistics)
        print("\n" + "=" * 80)
        print("STEP 5: Handling missing values (using training statistics)")
        print("=" * 80)
        X_df = self.handle_missing_values(X_df, is_training=False, 
                                          imputation_values=self.imputation_values)
        print(f"  Test missing values after imputation: {X_df.isnull().sum().sum()}")
        
        # NOTE: Scaling is NOT applied here - it will be done after feature engineering
        
        print("\n" + "=" * 80)
        print("TEST PREPROCESSING COMPLETE (Phase 1 - Before Feature Engineering)")
        print("=" * 80)
        print(f"Test set shape: {X_df.shape}")
        print("\n" + "=" * 80)
        print("DATA LEAKAGE PREVENTION VERIFIED (Phase 1):")
        print("=" * 80)
        print("  ✓ Conservative data cleaning (exact match only)")
        print("  ✓ Using training set category encoders")
        print("  ✓ Using training set outlier clipping bounds")
        print("  ✓ Using training set imputation statistics")
        print("\n⚠️  Scaling will be applied AFTER feature engineering")
        print("=" * 80)
        
        return X_df.values, test_ids


def preprocess_pipeline(train_path, test_path, output_dir='processed_data',
                       use_smote=True, smote_method='smote', 
                       val_size=0.2, scaling_method='robust',
                       encoding_method='target'):
    """
    Complete preprocessing pipeline for both train and test data
    
    NOTE: This is Phase 1 of preprocessing - BEFORE feature engineering
    Scaling and SMOTE will be applied in the feature engineering pipeline
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        output_dir: Directory to save processed data
        use_smote: Whether to use SMOTE (passed to feature engineering)
        smote_method: SMOTE variant (passed to feature engineering)
        val_size: Validation split size
        scaling_method: Scaling method (passed to feature engineering)
        encoding_method: Encoding method for high-cardinality features ('target' or 'label')
        
    Returns:
        Dictionary with all preprocessed data
    """
    from pathlib import Path
    import json
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_state=55, encoding_method=encoding_method)
    print(f"Using encoding method: {encoding_method}")
    
    # Preprocess training data
    print("\n" + "="*80)
    print("PREPROCESSING TRAINING DATA (Phase 1 - Before Feature Engineering)")
    print("="*80 + "\n")
    train_data = preprocessor.preprocess_train(
        train_df,
        use_smote=use_smote,
        smote_method=smote_method,
        val_size=val_size,
        scaling_method=scaling_method
    )
    
    # Preprocess test data
    print("\n" + "="*80)
    print("PREPROCESSING TEST DATA (Phase 1 - Before Feature Engineering)")
    print("="*80 + "\n")
    X_test, test_ids = preprocessor.preprocess_test(test_df, scaling_method=scaling_method)
    
    # Save processed data
    print("\nSaving processed data (Phase 1)...")
    np.save(output_path / 'X_train.npy', train_data['X_train'])
    np.save(output_path / 'y_train.npy', train_data['y_train'])
    np.save(output_path / 'X_val.npy', train_data['X_val'])
    np.save(output_path / 'y_val.npy', train_data['y_val'])
    np.save(output_path / 'X_test.npy', X_test)
    np.save(output_path / 'test_ids.npy', test_ids)
    
    # Save feature names
    with open(output_path / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(train_data['feature_names']))
    
    # Save preprocessing settings for feature engineering pipeline
    settings = {
        'use_smote': use_smote,
        'smote_method': smote_method,
        'scaling_method': scaling_method,
        'encoding_method': encoding_method
    }
    with open(output_path / 'preprocessing_settings.json', 'w') as f:
        json.dump(settings, f, indent=2)
    
    print(f"Processed data saved to: {output_path}/")
    print(f"Preprocessing settings saved for feature engineering phase.")
    
    return {
        **train_data,
        'X_test': X_test,
        'test_ids': test_ids,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    # Run preprocessing pipeline
    result = preprocess_pipeline(
        train_path='data/train.csv',
        test_path='data/test.csv',
        use_smote=True,
        smote_method='smote',
        val_size=0.2,
        scaling_method='robust'
    )
    
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"Training samples: {result['X_train'].shape[0]}")
    print(f"Validation samples: {result['X_val'].shape[0]}")
    print(f"Test samples: {result['X_test'].shape[0]}")
    print(f"Number of features: {len(result['feature_names'])}")
    print("\nFeature names:")
    for i, name in enumerate(result['feature_names'], 1):
        print(f"  {i}. {name}")


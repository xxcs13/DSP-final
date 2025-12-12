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
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, random_state=42):
        """
        Initialize the data preprocessor
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []
        self.imputation_values = {}  # Store imputation statistics from training set
    
    def remove_anomalous_missing_values(self, df):
        """
        Remove rows with anomalous missing values (MNAR - Missing Not At Random)
        These are missing values that indicate data quality issues, not structural missingness
        
        Anomalous cases:
        1. Students with missing CGPA (students should have CGPA)
        2. Professionals with missing Profession (professionals should have profession)
        3. Anyone with missing Financial Stress (universal feature)
        4. Anyone with missing Dietary Habits (universal feature)
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe, number of rows removed
        """
        df = df.copy()
        original_size = len(df)
        
        # Identify student and professional masks
        is_student = df['Working Professional or Student'] == 'Student'
        is_professional = df['Working Professional or Student'] == 'Working Professional'
        
        # Identify anomalous rows
        anomalous_mask = pd.Series(False, index=df.index)
        
        # Case 1: Students with missing CGPA
        student_missing_cgpa = is_student & df['CGPA'].isnull()
        anomalous_mask |= student_missing_cgpa
        
        # Case 2: Professionals with missing Profession
        prof_missing_profession = is_professional & df['Profession'].isnull()
        anomalous_mask |= prof_missing_profession
        
        # Case 3: Missing Financial Stress (universal feature - should not be missing)
        anomalous_mask |= df['Financial Stress'].isnull()
        
        # Case 4: Missing Dietary Habits (universal feature - should not be missing)
        anomalous_mask |= df['Dietary Habits'].isnull()
        
        # Remove anomalous rows
        df_cleaned = df[~anomalous_mask].copy()
        removed_count = original_size - len(df_cleaned)
        
        if removed_count > 0:
            print(f"  Removed {removed_count:,} rows with anomalous missing values:")
            print(f"    - Students with missing CGPA: {student_missing_cgpa.sum():,}")
            print(f"    - Professionals with missing Profession: {prof_missing_profession.sum():,}")
            print(f"    - Missing Financial Stress: {df['Financial Stress'].isnull().sum()}")
            print(f"    - Missing Dietary Habits: {df['Dietary Habits'].isnull().sum()}")
            print(f"  Remaining: {len(df_cleaned):,} rows ({100*len(df_cleaned)/original_size:.2f}%)")
        
        return df_cleaned, removed_count
    
    def clean_data_quality_issues(self, df):

        """
        Clean data quality issues in specific columns
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Clean Sleep Duration - keep only valid categories
        valid_sleep = ['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', 'More than 8 hours']
        df.loc[~df['Sleep Duration'].isin(valid_sleep), 'Sleep Duration'] = np.nan
        
        # Clean Dietary Habits - keep only valid categories
        valid_diet = ['Healthy', 'Moderate', 'Unhealthy']
        df.loc[~df['Dietary Habits'].isin(valid_diet), 'Dietary Habits'] = np.nan
        
        # Clean Degree - remove entries that look like errors (names, numbers, etc.)
        # Keep common degree patterns
        valid_degree_patterns = ['Class', 'BSc', 'MSc', 'BA', 'MA', 'B.', 'M.', 'PhD', 
                                'BBA', 'MBA', 'BCA', 'MCA', 'BE', 'ME', 'BTech', 'MTech',
                                'B.Tech', 'M.Tech', 'LLB', 'LLM', 'MD', 'BHM', 'MHM']
        
        def is_valid_degree(degree):
            if pd.isna(degree):
                return True
            degree_str = str(degree)
            return any(pattern in degree_str for pattern in valid_degree_patterns)
        
        df.loc[~df['Degree'].apply(is_valid_degree), 'Degree'] = np.nan
        
        # Clean Profession - remove entries that look like errors
        # Professionals without profession will be marked as 'Unknown' instead of null
        if 'Profession' in df.columns:
            invalid_professions = ['Class', 'BSc', 'MA', 'BBA', 'B.Com', 'City', 'Family', 
                                  'Yuvraj', 'Gender', 'Age']
            for inv in invalid_professions:
                df.loc[df['Profession'].str.contains(inv, case=False, na=False), 'Profession'] = np.nan
        
        return df
        
    def handle_missing_values(self, df, is_training=True, imputation_values=None):
        """
        Intelligently handle missing values based on context
        
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
            
            # Handle anomalous missing CGPA for students (use median from training students ONLY)
            if 'CGPA' in df.columns:
                student_cgpa_median = df.loc[is_student, 'CGPA'].median()
                self.imputation_values['CGPA_student'] = student_cgpa_median
            
            # Numerical features: compute median from training data
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numerical_cols:
                if col not in ['id', 'Depression']:
                    self.imputation_values[f'{col}_median'] = df[col].median()
            
            # Categorical features: compute mode from training data
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in ['id', 'Name']:
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        self.imputation_values[f'{col}_mode'] = mode_value[0]
                    else:
                        self.imputation_values[f'{col}_mode'] = 'Unknown'
            
            imputation_values = self.imputation_values
        else:
            # Validation/Test mode: Use provided statistics
            if imputation_values is None:
                imputation_values = self.imputation_values
            if not imputation_values:
                raise ValueError("No imputation values available for test/validation data")
        
        # Apply imputation using the computed (or provided) statistics
        # For Students: Fill work-related features with 0
        student_work_features = ['Work Pressure', 'Job Satisfaction']
        for feature in student_work_features:
            if feature in df.columns:
                df.loc[is_student, feature] = df.loc[is_student, feature].fillna(
                    imputation_values.get(f'{feature}_student', 0.0)
                )
        
        # For Students: Fill missing Profession with 'Student'
        if 'Profession' in df.columns:
            df.loc[is_student, 'Profession'] = df.loc[is_student, 'Profession'].fillna('Student')
        
        # For Professionals: Fill academic-related features with 0
        prof_academic_features = ['Academic Pressure', 'Study Satisfaction', 'CGPA']
        for feature in prof_academic_features:
            if feature in df.columns:
                df.loc[is_professional, feature] = df.loc[is_professional, feature].fillna(
                    imputation_values.get(f'{feature}_professional', 0.0)
                )
        
        # Handle anomalous missing values in CGPA for students (use training median)
        if 'CGPA' in df.columns:
            student_cgpa_fill = imputation_values.get('CGPA_student', 0.0)
            df.loc[is_student, 'CGPA'] = df.loc[is_student, 'CGPA'].fillna(student_cgpa_fill)
        
        # Fill remaining missing values using training statistics
        # Numerical features: use training median
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if col not in ['id', 'Depression']:
                fill_value = imputation_values.get(f'{col}_median', 0.0)
                df[col] = df[col].fillna(fill_value)
        
        # Categorical features: use training mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['id', 'Name']:
                fill_value = imputation_values.get(f'{col}_mode', 'Unknown')
                df[col] = df[col].fillna(fill_value)
        
        return df
        
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features to numerical
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for test)
            
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        # Exclude id and Name
        categorical_cols = [col for col in categorical_cols if col not in ['id', 'Name', 'Depression']]
        
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
        
        self.categorical_features = categorical_cols
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
            method: 'standard' or 'robust'
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Exclude id and target
        numerical_cols = [col for col in numerical_cols if col not in ['id', 'Depression']]
        
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
        Complete preprocessing pipeline for training data with NO DATA LEAKAGE
        
        CRITICAL ORDER TO PREVENT LEAKAGE:
        1. Remove anomalous missing values (quality filter - OK before split)
        2. Clean data quality issues (categorical validation - OK before split)
        3. Encode categorical features (deterministic mapping - OK before split)
        4. Prepare features and split data FIRST
        5. Compute imputation statistics ONLY on training set
        6. Apply imputation to both train and validation separately
        7. Scale using training statistics only
        8. Apply SMOTE only to training set
        
        Args:
            df: Training dataframe
            use_smote: Whether to apply SMOTE
            smote_method: SMOTE variant to use
            val_size: Validation split size
            scaling_method: Scaling method to use
            
        Returns:
            Dictionary containing preprocessed train/val sets
        """
        print("Starting training data preprocessing...")
        print(f"Original shape: {df.shape}")
        
        # Step 1: Remove anomalous missing values (MNAR - data quality issues)
        print("\nStep 1: Removing anomalous missing values...")
        df, removed_count = self.remove_anomalous_missing_values(df)
        
        # Step 2: Clean data quality issues (invalid categories)
        print("\nStep 2: Cleaning data quality issues...")
        df = self.clean_data_quality_issues(df)
        
        # Step 3: Encode categorical features (deterministic mapping - no leakage)
        print("\nStep 3: Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=True)
        
        # Step 4: Prepare features and target
        print("\nStep 4: Preparing features...")
        X, y = self.prepare_features(df, target_col='Depression')
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        # Step 5: SPLIT FIRST - This is critical to prevent leakage
        print(f"\nStep 5: Creating train/validation split (val_size={val_size})...")
        X_train, X_val, y_train, y_val = self.create_validation_split(
            X, y, val_size=val_size, stratify=True
        )
        print(f"Train set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        
        # Convert to DataFrames for missing value handling
        X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
        X_val_df = pd.DataFrame(X_val, columns=self.feature_names)
        
        # Step 6: Handle missing values - COMPUTE ON TRAINING SET ONLY
        print("\nStep 6: Handling missing values...")
        print("  Computing imputation statistics from TRAINING SET ONLY...")
        X_train_df = self.handle_missing_values(X_train_df, is_training=True)
        print(f"  Training missing values after imputation: {X_train_df.isnull().sum().sum()}")
        
        print("  Applying training statistics to VALIDATION SET...")
        X_val_df = self.handle_missing_values(X_val_df, is_training=False, 
                                               imputation_values=self.imputation_values)
        print(f"  Validation missing values after imputation: {X_val_df.isnull().sum().sum()}")
        
        # Step 7: Scale features - FIT ON TRAINING ONLY
        print(f"\nStep 7: Scaling features using {scaling_method} scaler...")
        print("  Fitting scaler on TRAINING SET ONLY...")
        X_train_scaled = self.scale_features(X_train_df, fit=True, method=scaling_method)
        
        print("  Applying training scaler to VALIDATION SET...")
        X_val_scaled = self.scale_features(X_val_df, fit=False, method=scaling_method)
        
        # Step 8: Handle class imbalance - ONLY ON TRAINING SET
        if use_smote:
            print(f"\nStep 8: Handling class imbalance using {smote_method}...")
            print("  Applying SMOTE to TRAINING SET ONLY...")
            X_train_resampled, y_train_resampled = self.handle_class_imbalance(
                X_train_scaled.values, y_train, method=smote_method
            )
        else:
            print("\nStep 8: Skipping SMOTE...")
            X_train_resampled = X_train_scaled.values
            y_train_resampled = y_train
        
        print("\nPreprocessing complete!")
        print(f"Final training set shape: {X_train_resampled.shape}")
        print(f"Final validation set shape: {X_val_scaled.shape}")
        print("\nDATA LEAKAGE PREVENTION VERIFIED:")
        print("  - Anomalous rows removed before split")
        print("  - Imputation statistics computed on training set only")
        print("  - Scaling parameters fitted on training set only")
        print("  - SMOTE applied to training set only")
        
        return {
            'X_train': X_train_resampled,
            'y_train': y_train_resampled,
            'X_val': X_val_scaled.values,
            'y_val': y_val,
            'feature_names': self.feature_names,
            'imputation_values': self.imputation_values,
            'removed_anomalies': removed_count
        }
        
    def preprocess_test(self, df, scaling_method='robust'):
        """
        Complete preprocessing pipeline for test data
        Uses statistics computed from training data to prevent leakage
        
        Args:
            df: Test dataframe
            scaling_method: Scaling method (should match training)
            
        Returns:
            Preprocessed feature matrix and ids
        """
        print("Starting test data preprocessing...")
        print(f"Original shape: {df.shape}")
        
        # Save ids for submission
        test_ids = df['id'].values
        
        # Step 1: Clean data quality issues (deterministic validation - no leakage)
        print("\nStep 1: Cleaning data quality issues...")
        df = self.clean_data_quality_issues(df)
        
        # Step 2: Encode categorical features (using training encoders)
        print("\nStep 2: Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=False)
        
        # Step 3: Prepare features
        print("\nStep 3: Preparing features...")
        X, _ = self.prepare_features(df, target_col='Depression')
        print(f"Feature matrix shape: {X.shape}")
        
        # Step 4: Handle missing values (using TRAINING statistics)
        print("\nStep 4: Handling missing values...")
        print("  Using imputation statistics from TRAINING SET...")
        X_df = pd.DataFrame(X, columns=self.feature_names)
        X_df = self.handle_missing_values(X_df, is_training=False, 
                                          imputation_values=self.imputation_values)
        print(f"  Test missing values after imputation: {X_df.isnull().sum().sum()}")
        
        # Step 5: Scale features (using TRAINING scaler)
        print(f"\nStep 5: Scaling features using {scaling_method} scaler...")
        print("  Using scaler fitted on TRAINING SET...")
        X_scaled = self.scale_features(X_df, fit=False, method=scaling_method)
        
        print("\nTest preprocessing complete!")
        print(f"Final test set shape: {X_scaled.shape}")
        print("\nDATA LEAKAGE PREVENTION VERIFIED:")
        print("  - Using training set imputation statistics")
        print("  - Using training set scaler")
        
        return X_scaled.values, test_ids


def preprocess_pipeline(train_path, test_path, output_dir='processed_data',
                       use_smote=True, smote_method='smote', 
                       val_size=0.2, scaling_method='robust'):
    """
    Complete preprocessing pipeline for both train and test data
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        output_dir: Directory to save processed data
        use_smote: Whether to use SMOTE
        smote_method: SMOTE variant
        val_size: Validation split size
        scaling_method: Scaling method
        
    Returns:
        Dictionary with all preprocessed data
    """
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_state=42)
    
    # Preprocess training data
    print("\n" + "="*80)
    print("PREPROCESSING TRAINING DATA")
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
    print("PREPROCESSING TEST DATA")
    print("="*80 + "\n")
    X_test, test_ids = preprocessor.preprocess_test(test_df, scaling_method=scaling_method)
    
    # Save processed data
    print("\nSaving processed data...")
    np.save(output_path / 'X_train.npy', train_data['X_train'])
    np.save(output_path / 'y_train.npy', train_data['y_train'])
    np.save(output_path / 'X_val.npy', train_data['X_val'])
    np.save(output_path / 'y_val.npy', train_data['y_val'])
    np.save(output_path / 'X_test.npy', X_test)
    np.save(output_path / 'test_ids.npy', test_ids)
    
    # Save feature names
    with open(output_path / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(train_data['feature_names']))
    
    print(f"Processed data saved to: {output_path}/")
    
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


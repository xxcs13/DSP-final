"""
Original Feature Engineering Module for Depression Prediction

This module provides a pass-through pipeline that uses ONLY the original features
without any feature engineering. This allows testing model performance on raw features.
"""

import numpy as np
from pathlib import Path


def engineer_pipeline(input_dir='processed_data', output_dir='engineered_data'):
    """
    Pass-through feature engineering pipeline using only original features
    
    This function loads preprocessed data and saves it to the engineered_data directory
    without creating any new features. This allows direct comparison of model performance
    with and without feature engineering.
    
    Args:
        input_dir: Directory with preprocessed data
        output_dir: Directory to save data (with original features only)
        
    Returns:
        Dictionary with data and metadata
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*80)
    print("ORIGINAL FEATURE PIPELINE (NO FEATURE ENGINEERING)")
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
    print(f"  Features: {len(feature_names)}\n")
    
    print("="*80)
    print("USING ORIGINAL FEATURES ONLY (NO ENGINEERING)")
    print("="*80 + "\n")
    print("  No feature engineering applied")
    print("  Using preprocessed features directly")
    print(f"  Feature count: {len(feature_names)}")
    
    # Save data to engineered_data directory (but without engineering)
    print("\n" + "="*80)
    print("SAVING DATA WITH ORIGINAL FEATURES")
    print("="*80 + "\n")
    
    np.save(output_path / 'X_train_eng.npy', X_train)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'X_val_eng.npy', X_val)
    np.save(output_path / 'y_val.npy', y_val)
    np.save(output_path / 'X_test_eng.npy', X_test)
    np.save(output_path / 'test_ids.npy', test_ids)
    
    # Save feature names
    with open(output_path / 'feature_names_eng.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    print(f"Data saved to: {output_path}/")
    print(f"\nFinal feature count: {len(feature_names)}")
    print(f"New features created: 0 (using original features only)")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'test_ids': test_ids,
        'feature_names': feature_names,
        'engineer': None
    }


if __name__ == "__main__":
    # Run original feature pipeline
    result = engineer_pipeline(
        input_dir='processed_data',
        output_dir='engineered_data'
    )
    
    print("\n" + "="*80)
    print("ORIGINAL FEATURE PIPELINE SUMMARY")
    print("="*80)
    print(f"Training samples: {result['X_train'].shape[0]}")
    print(f"Validation samples: {result['X_val'].shape[0]}")
    print(f"Test samples: {result['X_test'].shape[0]}")
    print(f"Total features: {len(result['feature_names'])}")
    print("\nNote: This pipeline uses original features only without any engineering.")
    print("To use engineered features, import from new_feature_engineering instead.")

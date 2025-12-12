"""
Submission File Validator

Validates that submission files match the required format
"""

import pandas as pd
import numpy as np
import sys


def validate_submission(submission_path, sample_path):
    """
    Validate submission file format
    
    Args:
        submission_path: Path to submission file
        sample_path: Path to sample submission file
        
    Returns:
        Boolean indicating if valid
    """
    print(f"Validating: {submission_path}")
    print("="*80)
    
    try:
        # Load files
        submission = pd.read_csv(submission_path)
        sample = pd.read_csv(sample_path)
        
        # Check 1: Column names
        if list(submission.columns) != list(sample.columns):
            print("ERROR: Column names do not match")
            print(f"  Expected: {list(sample.columns)}")
            print(f"  Got: {list(submission.columns)}")
            return False
        print("✓ Column names match")
        
        # Check 2: Number of rows
        if len(submission) != len(sample):
            print("ERROR: Number of rows do not match")
            print(f"  Expected: {len(sample)}")
            print(f"  Got: {len(submission)}")
            return False
        print(f"✓ Number of rows match: {len(submission)}")
        
        # Check 3: ID column
        if not (submission['id'] == sample['id']).all():
            print("ERROR: ID column values do not match")
            return False
        print("✓ ID column values match")
        
        # Check 4: Depression values are binary
        unique_values = submission['Depression'].unique()
        if not set(unique_values).issubset({0, 1}):
            print("ERROR: Depression values must be 0 or 1")
            print(f"  Found: {unique_values}")
            return False
        print("✓ Depression values are binary (0 or 1)")
        
        # Check 5: No missing values
        if submission.isnull().any().any():
            print("ERROR: Submission contains missing values")
            return False
        print("✓ No missing values")
        
        # Check 6: Data types
        if submission['id'].dtype != sample['id'].dtype:
            print("WARNING: ID column data type differs")
            print(f"  Expected: {sample['id'].dtype}")
            print(f"  Got: {submission['id'].dtype}")
        
        # Print statistics
        print("\n" + "="*80)
        print("SUBMISSION STATISTICS")
        print("="*80)
        print(f"Total predictions: {len(submission)}")
        print(f"No Depression (0): {(submission['Depression'] == 0).sum()} ({(submission['Depression'] == 0).sum() / len(submission) * 100:.2f}%)")
        print(f"Depression (1): {(submission['Depression'] == 1).sum()} ({(submission['Depression'] == 1).sum() / len(submission) * 100:.2f}%)")
        
        print("\n" + "="*80)
        print("VALIDATION PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SUBMISSION FILE VALIDATION")
    print("="*80 + "\n")
    
    # Validate both submission files
    files_to_validate = [
        'submission_catboost.csv',
        'submission_ensemble.csv'
    ]
    
    sample_path = 'data/sample_submission.csv'
    
    results = {}
    for file in files_to_validate:
        print()
        valid = validate_submission(file, sample_path)
        results[file] = valid
        print()
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_valid = True
    for file, valid in results.items():
        status = "VALID" if valid else "INVALID"
        symbol = "✓" if valid else "✗"
        print(f"{symbol} {file}: {status}")
        if not valid:
            all_valid = False
    
    print("\n" + "="*80)
    
    if all_valid:
        print("ALL SUBMISSION FILES ARE VALID")
        print("\nRecommended for submission: submission_ensemble.csv")
        print("="*80)
        sys.exit(0)
    else:
        print("SOME SUBMISSION FILES ARE INVALID")
        print("="*80)
        sys.exit(1)


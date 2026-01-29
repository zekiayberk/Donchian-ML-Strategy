
import pandas as pd
import hashlib

def generate_fold_ranges(df, config):
    """
    Generates the exact same date ranges as src/ml/train_wfo.py.
    """
    timestamps = df.index
    start_date = timestamps.min()
    end_date = timestamps.max()
    
    train_days = config['ml'].get('train_days', 180)
    valid_days = config['ml'].get('valid_days', 30)
    test_days = config['ml'].get('test_days', 30)
    
    folds = []
    current_test_start = start_date + pd.Timedelta(days=train_days + valid_days)
    
    fold_idx = 1
    while current_test_start < end_date:
        test_end = current_test_start + pd.Timedelta(days=test_days)
        valid_start = current_test_start - pd.Timedelta(days=valid_days)
        train_start = valid_start - pd.Timedelta(days=train_days)
        
        # Check min samples requirement (loose check to mimic training logic)
        train_mask = (timestamps >= train_start) & (timestamps < valid_start)
        valid_mask = (timestamps >= valid_start) & (timestamps < current_test_start)
        
        if train_mask.sum() < 20 or valid_mask.sum() < 5:
             current_test_start += pd.Timedelta(days=test_days)
             continue
             
        folds.append({
            'fold': fold_idx,
            'train_range': (train_start, valid_start),
            'valid_range': (valid_start, current_test_start),
            'test_range': (current_test_start, test_end)
        })
        
        fold_idx += 1
        current_test_start += pd.Timedelta(days=test_days)
        
    return folds

def get_data_hash(df):
    """
    Computes a SHA256 hash of the dataframe content to ensure consistency.
    """
    # Simply hashing values/index bytes
    content = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(content).hexdigest()

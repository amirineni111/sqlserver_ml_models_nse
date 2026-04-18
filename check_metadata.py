"""Check NSE training metadata"""
import pickle
import numpy as np
from pathlib import Path

metadata_path = Path('data/nse_models/nse_training_metadata.pkl')
if metadata_path.exists():
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print('=' * 60)
    print('NSE TRAINING METADATA')
    print('=' * 60)
    print(f'Training date: {metadata.get("training_timestamp", "unknown")}')
    print(f'Best classifier: {metadata.get("best_clf_model", "unknown")}')
    print(f'Feature count: {len(metadata.get("feature_columns", []))}')
    print()
    
    # Check if training distribution is stored
    if 'training_distribution' in metadata:
        dist = metadata['training_distribution']
        print(f'TRAINING DATA DISTRIBUTION:')
        for key, val in dist.items():
            print(f'  {key}: {val}')
        print()
    
    # Model results
    clf_results = metadata.get('clf_results', {})
    if clf_results:
        print('MODEL F1 SCORES:')
        for name, results in clf_results.items():
            f1 = results.get('f1_score', 0)
            acc = results.get('accuracy', 0)
            print(f'  {name:25s}: F1={f1:.4f}, Acc={acc:.4f}')
        print()
    
    # Check encoder classes
    if 'direction_encoder_classes' in metadata:
        print(f'Direction classes: {metadata["direction_encoder_classes"]}')
    
    print('=' * 60)
else:
    print('Metadata file not found')

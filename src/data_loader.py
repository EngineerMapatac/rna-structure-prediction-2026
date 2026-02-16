import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# Import our configuration variables
try:
    from src import config
except ImportError:
    # Fallback for running script directly
    import config

# Dictionary to map RNA bases to integers
# 0 is usually reserved for padding, so we start at 1
BASE_TO_INT = {'A': 1, 'C': 2, 'G': 3, 'U': 4, 'N': 5}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}

def load_csv_data(split='train'):
    """
    Loads the raw CSV data using paths from config.py.
    
    Args:
        split (str): 'train' or 'test'
    """
    if split == 'train':
        print(f"Loading training data from {config.TRAIN_CSV}...")
        df = pd.read_csv(config.TRAIN_CSV)
    elif split == 'test':
        print(f"Loading test data from {config.TEST_CSV}...")
        df = pd.read_csv(config.TEST_CSV)
    else:
        raise ValueError("Split must be 'train' or 'test'")
        
    return df

def tokenize_sequence(sequence, max_len=None):
    """
    Converts a string sequence (e.g., 'ACGU') to a list of integers.
    """
    tokens = [BASE_TO_INT.get(base, 5) for base in sequence] # 5 for unknown 'N'
    
    if max_len:
        # Pad or truncate if a max_len is specified (common for batching)
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens)) # 0 is padding
        else:
            tokens = tokens[:max_len]
            
    return torch.tensor(tokens, dtype=torch.long)

class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Sequences.
    """
    def __init__(self, df, mode='train'):
        self.df = df
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Get the Sequence ID and content
        target_id = row['target_id']
        sequence = row['sequence']
        
        # 2. Tokenize the sequence
        tokenized_seq = tokenize_sequence(sequence)
        
        # 3. Prepare the sample dictionary
        sample = {
            'target_id': target_id,
            'sequence_str': sequence,
            'sequence_tokens': tokenized_seq,
            'length': len(sequence)
        }
        
        # Note: We will add 3D coordinate loading here later
        # when we integrate the parser.
        
        return sample

# --- Usage Example (if run directly) ---
if __name__ == "__main__":
    # Test the loader
    try:
        df_train = load_csv_data('train')
        print(f"Loaded {len(df_train)} samples.")
        
        # Test the Dataset class
        dataset = RNADataset(df_train)
        print("Sample 0:", dataset[0])
        
    except FileNotFoundError:
        print("Files not found. Check config paths.")
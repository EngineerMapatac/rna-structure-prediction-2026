import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import os

try:
    from src import config
except ImportError:
    import config

# --- MAPPING CONSTANTS ---
# 1. Define the mapping from Letter -> Number
BASE_TO_INT = {'A': 1, 'C': 2, 'G': 3, 'U': 4, 'N': 5}

# 2. Define the reverse mapping from Number -> Letter (CRITICAL FOR INFERENCE)
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}

# 3. Padding token
PAD_TOKEN = 0

class RNADataset(Dataset):
    def __init__(self, mode='train', max_len=None):
        self.mode = mode
        self.max_len = max_len
        
        if mode == 'train':
            self.seq_df = pd.read_csv(config.TRAIN_CSV)
            coords_path = os.path.join(config.PROCESSED_DATA_DIR, 'coords.csv')
            if os.path.exists(coords_path):
                self.coords_df = pd.read_csv(coords_path)
            else:
                self.coords_df = pd.DataFrame()
        else:
            # For inference, use test_sequences.csv
            self.seq_df = pd.read_csv(config.TEST_CSV)
            self.coords_df = pd.DataFrame()

    def __len__(self):
        return len(self.seq_df)
    
    def tokenize(self, sequence):
        tokens = [BASE_TO_INT.get(base, 5) for base in sequence]
        return torch.tensor(tokens, dtype=torch.long)

    def get_coords(self, target_id, length):
        coords = np.zeros((length, 3), dtype=np.float32)
        mask = np.zeros((length,), dtype=np.float32)
        
        if self.coords_df.empty:
            return torch.tensor(coords), torch.tensor(mask)
            
        subset = self.coords_df[self.coords_df['target_id'] == target_id]
        if subset.empty:
            return torch.tensor(coords), torch.tensor(mask)
            
        indices = subset['residue_number'].values - 1 
        valid = (indices >= 0) & (indices < length)
        
        if valid.sum() > 0:
            coords[indices[valid]] = subset.loc[valid, ['x', 'y', 'z']].values
            mask[indices[valid]] = 1.0
            
        return torch.tensor(coords), torch.tensor(mask)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]
        target_id = row['target_id']
        sequence = row['sequence']
        length = len(sequence)
        
        seq_tokens = self.tokenize(sequence)
        coords, mask = self.get_coords(target_id, length)
        
        return {
            'target_id': target_id,
            'sequence': seq_tokens,
            'coords': coords,
            'mask': mask
        }
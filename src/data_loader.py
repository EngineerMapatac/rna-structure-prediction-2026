import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

try:
    from src import config
except ImportError:
    import config

# Mapping constants
BASE_TO_INT = {'A': 1, 'C': 2, 'G': 3, 'U': 4, 'N': 5}
PAD_TOKEN = 0

class RNADataset(Dataset):
    """
    PyTorch Dataset that returns:
    - sequence (Input)
    - coordinates (Target)
    - mask (Which residues define the loss)
    """
    def __init__(self, mode='train', max_len=None):
        self.mode = mode
        self.max_len = max_len
        
        # 1. Load Sequences
        if mode == 'train':
            self.seq_df = pd.read_csv(config.TRAIN_CSV)
            # Load the coordinates we processed earlier
            coords_path = os.path.join(config.PROCESSED_DATA_DIR, 'coords.csv')
            if os.path.exists(coords_path):
                self.coords_df = pd.read_csv(coords_path)
                print(f"Loaded {len(self.coords_df)} coordinate labels.")
            else:
                print("WARNING: coords.csv not found! Run src/cif_parser.py first.")
                self.coords_df = pd.DataFrame()
        else:
            self.seq_df = pd.read_csv(config.TEST_CSV)
            self.coords_df = pd.DataFrame()

    def __len__(self):
        return len(self.seq_df)
    
    def tokenize(self, sequence):
        """Converts string 'ACGU' to tensor [1, 2, 3, 4]"""
        tokens = [BASE_TO_INT.get(base, 5) for base in sequence]
        return torch.tensor(tokens, dtype=torch.long)

    def get_coords(self, target_id, length):
        """
        Retrieves (L, 3) coordinates for a specific target_id.
        Returns coordinates and a mask (1 = valid data, 0 = missing).
        """
        # Create empty containers
        coords = np.zeros((length, 3), dtype=np.float32)
        mask = np.zeros((length,), dtype=np.float32)
        
        if self.coords_df.empty:
            return torch.tensor(coords), torch.tensor(mask)
            
        # Filter for this specific RNA molecule
        # We assume target_id in sequence matches target_id in coords
        subset = self.coords_df[self.coords_df['target_id'] == target_id]
        
        if subset.empty:
            return torch.tensor(coords), torch.tensor(mask)
            
        # Map residue numbers to indices (1-based index -> 0-based index)
        # Note: This is a simplified alignment. Real production code checks sequence alignment carefully.
        indices = subset['residue_number'].values - 1 
        
        # Filter out indices that might be out of bounds (safety check)
        valid = (indices >= 0) & (indices < length)
        
        safe_indices = indices[valid]
        
        if len(safe_indices) > 0:
            coords[safe_indices] = subset.loc[valid, ['x', 'y', 'z']].values
            mask[safe_indices] = 1.0
            
        return torch.tensor(coords, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]
        target_id = row['target_id']
        sequence = row['sequence']
        length = len(sequence)
        
        # Inputs
        seq_tokens = self.tokenize(sequence)
        
        # Targets (Labels)
        coords, mask = self.get_coords(target_id, length)
        
        return {
            'target_id': target_id,
            'sequence': seq_tokens,
            'coords': coords,   # Shape: (L, 3)
            'mask': mask        # Shape: (L,)
        }

# --- Test Block ---
if __name__ == "__main__":
    # Test the loader with a small batch
    print("Testing Data Loader...")
    
    dataset = RNADataset(mode='train')
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nTarget ID: {sample['target_id']}")
        print(f"Sequence Shape: {sample['sequence'].shape}")
        print(f"Coords Shape:   {sample['coords'].shape}")
        print(f"Mask Sum:       {sample['mask'].sum()} (Valid residues found)")
        
        if sample['mask'].sum() == 0:
            print("\nNOTE: Mask sum is 0. This is expected if the first sequence in train_sequences.csv")
            print("is NOT one of the 5 PDBs we downloaded. Try iterating to find a match.")
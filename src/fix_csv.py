import pandas as pd
import os
try:
    from src import config
except ImportError:
    import config

def reconstruct_training_csv():
    # 1. Load the coordinates you extracted earlier
    coords_path = os.path.join(config.PROCESSED_DATA_DIR, 'coords.csv')
    
    if not os.path.exists(coords_path):
        print("Error: coords.csv not found. Run src/cif_parser.py first.")
        return

    print(f"Reading from {coords_path}...")
    df = pd.read_csv(coords_path)
    
    # 2. Group by Target ID to reconstruct the sequence
    # We want to turn the individual rows (A, C, G...) back into a string "ACG..."
    targets = []
    
    for target_id, group in df.groupby('target_id'):
        # Sort by residue number to ensure correct order
        group = group.sort_values('residue_number')
        
        # Join the characters to form the sequence
        sequence = "".join(group['residue_name'].tolist())
        
        targets.append({
            'target_id': target_id,
            'sequence': sequence,
            'description': 'Reconstructed from PDB' # Dummy column
        })
    
    # 3. Save as the new train_sequences.csv
    seq_df = pd.DataFrame(targets)
    output_path = config.TRAIN_CSV
    
    seq_df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Created {output_path}")
    print(f"Generated {len(seq_df)} sequences matching your downloaded PDBs.")
    print("\nSample:")
    print(seq_df.head())

if __name__ == "__main__":
    reconstruct_training_csv()
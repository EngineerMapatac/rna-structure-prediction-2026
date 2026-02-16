import os
import pandas as pd
from Bio.PDB import MMCIFParser
import warnings
from tqdm import tqdm
import torch

# Import project configuration
try:
    from src import config
except ImportError:
    import config

# Suppress Biopython warnings about discontinuous chains
warnings.filterwarnings('ignore')

def parse_single_cif(file_path):
    """
    Parses a single .cif file to extract C1' coordinates.
    Returns a DataFrame of residues and coordinates.
    """
    parser = MMCIFParser(QUIET=True)
    
    try:
        structure_id = os.path.basename(file_path).split('.')[0]
        structure = parser.get_structure(structure_id, file_path)
        
        extracted_data = []

        # Iterate through structure hierarchy
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Filter for standard RNA residues
                    if residue.get_resname().strip() in ['A', 'C', 'G', 'U']:
                        
                        # We specifically need the C1' atom
                        if config.TARGET_ATOM in residue:
                            atom = residue[config.TARGET_ATOM]
                            coord = atom.get_coord()
                            
                            extracted_data.append({
                                'target_id': structure_id,
                                'chain_id': chain.id,
                                'residue_name': residue.get_resname(),
                                'residue_number': residue.id[1],
                                'x': coord[0],
                                'y': coord[1],
                                'z': coord[2]
                            })
                            
        return pd.DataFrame(extracted_data)

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return pd.DataFrame()

def process_directory(input_dir=config.PDB_DIR, limit=None):
    """
    Processes all .cif files in the directory.
    
    Args:
        input_dir (str): Directory containing .cif files.
        limit (int): Optional limit for testing (e.g., process only first 10 files).
    """
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return pd.DataFrame()

    all_files = [f for f in os.listdir(input_dir) if f.endswith('.cif')]
    
    if limit:
        all_files = all_files[:limit]
        
    print(f"Found {len(all_files)} structures. Starting extraction...")
    
    results = []
    for f in tqdm(all_files):
        full_path = os.path.join(input_dir, f)
        df = parse_single_cif(full_path)
        if not df.empty:
            results.append(df)
            
    if results:
        final_df = pd.concat(results, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()

# --- Execution Block ---
if __name__ == "__main__":
    # When run as a script, process the raw data and save it
    
    # 1. Process the files
    print(f"Reading from: {config.PDB_DIR}")
    df_coords = process_directory(limit=None) # Set limit=10 for quick test
    
    # 2. Save to processed folder
    if not df_coords.empty:
        output_path = os.path.join(config.PROCESSED_DATA_DIR, 'coords.csv')
        df_coords.to_csv(output_path, index=False)
        print(f"Success! Saved {len(df_coords)} residues to {output_path}")
    else:
        print("No data extracted. Did you download the .cif files to data/raw/PDB_RNA?")
import os
import urllib.request
try:
    from src import config
except ImportError:
    import config

# List of real RNA PDB IDs to download (small examples)
SAMPLE_PDBS = ['1y26', '1fka', '430d', '1xvj', '1csl']

def download_pdb(pdb_id, output_dir):
    """
    Downloads a .cif file from the RCSB PDB database.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    output_path = os.path.join(output_dir, f"{pdb_id}.cif")
    
    print(f"Downloading {pdb_id} from {url}...")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f" -> Saved to {output_path}")
    except Exception as e:
        print(f" -> Failed: {e}")

if __name__ == "__main__":
    # Ensure the directory exists
    os.makedirs(config.PDB_DIR, exist_ok=True)
    
    print(f"Downloading samples to: {config.PDB_DIR}")
    
    for pdb_id in SAMPLE_PDBS:
        download_pdb(pdb_id, config.PDB_DIR)
        
    print("\nDone! Now run: python src/cif_parser.py")
# Create a temporary test file named test_loader.py
import src.data_loader as dl

ds = dl.RNADataset(mode='train')

print("Searching for a match...")
found = False
for i in range(len(ds)):
    sample = ds[i]
    if sample['mask'].sum() > 0:
        print(f"\nMATCH FOUND at index {i}!")
        print(f"Target: {sample['target_id']}")
        print(f"Coords: \n{sample['coords'][0:5]}") # Print first 5 coords
        found = True
        break

if not found:
    print("No matches found. Make sure the PDBs you downloaded match the IDs in train_sequences.csv!")
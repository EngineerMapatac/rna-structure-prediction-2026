import torch
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import our project modules
try:
    from src import config
    from src.model import RNAFoldingModel
    from src.data_loader import RNADataset, INT_TO_BASE
except ImportError:
    import config
    from model import RNAFoldingModel
    from data_loader import RNADataset, INT_TO_BASE

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Point to the last model you saved (e.g., epoch 10)
MODEL_PATH = os.path.join(config.MODELS_DIR, "model_epoch_10.pt") 

def run_inference():
    print(f"🚀 Loading model from {MODEL_PATH}...")
    
    # 1. Load the Model
    model = RNAFoldingModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("Did you run src/train.py?")
        return

    model.eval() # Set to evaluation mode (turns off training specifics)

    # 2. Load Test Data
    print(f"Loading test sequences from {config.TEST_CSV}...")
    test_dataset = RNADataset(mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    submission_rows = []

    print("Generating predictions...")
    with torch.no_grad(): # No need to calculate gradients for inference
        for batch in tqdm(test_loader):
            
            # Get data
            seq = batch['sequence'].to(DEVICE)      # (1, Len)
            target_id = batch['target_id'][0]       # String ID
            
            # Predict coordinates (1, Len, 3)
            preds = model(seq)
            
            # Move to CPU for processing
            preds = preds.cpu().numpy()[0]          # (Len, 3)
            seq_tokens = seq.cpu().numpy()[0]       # (Len,)
            
            # 3. Format Output for Kaggle
            # We need to create a row for EVERY residue
            for i, (token, coord) in enumerate(zip(seq_tokens, preds)):
                residue_num = i + 1
                residue_name = INT_TO_BASE.get(token, 'N')
                
                # Create the unique ID: "TargetID_ResidueNum"
                row_id = f"{target_id}_{residue_num}"
                
                # Extract x, y, z
                x, y, z = coord[0], coord[1], coord[2]
                
                # HACK: Duplicate the prediction 5 times to meet submission rules
                row_data = {
                    'ID': row_id,
                    'resname': residue_name,
                    'resid': residue_num,
                    'x_1': x, 'y_1': y, 'z_1': z,
                    'x_2': x, 'y_2': y, 'z_2': z,
                    'x_3': x, 'y_3': y, 'z_3': z,
                    'x_4': x, 'y_4': y, 'z_4': z,
                    'x_5': x, 'y_5': y, 'z_5': z,
                }
                submission_rows.append(row_data)

    # 4. Save Submission File
    df_sub = pd.DataFrame(submission_rows)
    
    # Reorder columns to match sample_submission.csv exactly
    cols = ['ID', 'resname', 'resid']
    for k in range(1, 6):
        cols.extend([f'x_{k}', f'y_{k}', f'z_{k}'])
        
    df_sub = df_sub[cols]
    
    output_file = os.path.join(config.SUBMISSIONS_DIR, 'submission.csv')
    df_sub.to_csv(output_file, index=False)
    
    print(f"\n✅ Submission generated: {output_file}")
    print(f"Total rows: {len(df_sub)}")
    print(df_sub.head())

if __name__ == "__main__":
    run_inference()
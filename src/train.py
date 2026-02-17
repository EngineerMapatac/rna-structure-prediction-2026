import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# Import our custom modules
try:
    from src import config
    from src.model import RNAFoldingModel
    from src.data_loader import RNADataset
except ImportError:
    import config
    from model import RNAFoldingModel
    from data_loader import RNADataset

# --- Hyperparameters ---
BATCH_SIZE = 1          # Keep to 1 for now to avoid padding complex sequences
LEARNING_RATE = 1e-3    # How fast the model learns
EPOCHS = 10             # How many times to loop through the data
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. Prepare Data
    print(f"🚀 Starting Training on {DEVICE}...")
    train_dataset = RNADataset(mode='train')
    
    # DataLoader handles shuffling and batching
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    model = RNAFoldingModel().to(DEVICE)
    
    # 3. Define Loss Function & Optimizer
    # MSELoss: Calculates (Predicted_Coord - Real_Coord)^2
    criterion = nn.MSELoss(reduction='none') 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. The Loop
    model.train() # Set model to training mode
    
    for epoch in range(EPOCHS):
        total_loss = 0
        valid_batches = 0
        
        # Progress bar for visual feedback
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            # Move data to GPU (if available)
            seq = batch['sequence'].to(DEVICE)          # Input: (Batch, Len)
            coords = batch['coords'].to(DEVICE)         # Target: (Batch, Len, 3)
            mask = batch['mask'].to(DEVICE)             # Mask: (Batch, Len)
            
            # A. Zero the gradients (reset from previous step)
            optimizer.zero_grad()
            
            # B. Forward Pass (Make a prediction)
            preds = model(seq)                          # Output: (Batch, Len, 3)
            
            # C. Calculate Loss
            # We only want to learn from valid residues (mask = 1)
            raw_loss = criterion(preds, coords)         # Calculate error for all points
            
            # Sum errors (x+y+z) and apply mask
            # If mask is 0 (missing data), the error becomes 0
            masked_loss = raw_loss.sum(dim=2) * mask    
            
            # Normalize by number of valid residues
            num_valid = mask.sum()
            if num_valid > 0:
                loss = masked_loss.sum() / num_valid
                
                # D. Backward Pass (Calculate corrections)
                loss.backward()
                
                # E. Optimizer Step (Update weights)
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                valid_batches += 1
                progress_bar.set_postfix({'loss': loss.item()})
            else:
                # Skip batch if no valid coordinate data exists
                continue
        
        # End of Epoch: Save the model
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            print(f"Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.4f}")
            
            # Save checkpoint
            save_path = os.path.join(config.MODELS_DIR, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
        else:
            print("Warning: No valid training data found in this epoch.")

    print(f"\n✅ Training Complete! Models saved in {config.MODELS_DIR}")

if __name__ == "__main__":
    train()
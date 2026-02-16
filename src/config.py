import os
import sys

# --- 1. Environment Detection ---
# Kaggle notebooks always mount input data to /kaggle/input
IS_KAGGLE = os.path.exists('/kaggle/input')

# --- 2. Project Root Setup ---
if not IS_KAGGLE:
    # Local: Root is one level up from 'src'
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    # Kaggle: Root is the working directory where we can write files
    PROJECT_ROOT = '/kaggle/working'

# --- 3. Input Data Paths (Read-Only) ---
if IS_KAGGLE:
    # Point to the official competition dataset
    DATA_ROOT = '/kaggle/input/stanford-rna-3d-folding-2'
else:
    # Point to your local 'data/raw' folder
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw')

# Specific file pointers
TRAIN_CSV = os.path.join(DATA_ROOT, 'train_sequences.csv')
TEST_CSV = os.path.join(DATA_ROOT, 'test_sequences.csv')
SAMPLE_SUBMISSION = os.path.join(DATA_ROOT, 'sample_submission.csv')

# Directories for heavy data
PDB_DIR = os.path.join(DATA_ROOT, 'PDB_RNA')  # Contains .cif files
MSA_DIR = os.path.join(DATA_ROOT, 'MSA')      # Contains alignment files

# --- 4. Output Paths (Writable) ---
# We store processed data and models here
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SUBMISSIONS_DIR = os.path.join(PROJECT_ROOT, 'submissions')

# Automatically create these folders if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# --- 5. Global Constants ---
SEED = 42
NUM_PREDICTIONS = 5    # Competition requires 5 structural predictions
TARGET_ATOM = "C1'"    # The specific atom we must predict
TIMEOUT_HOURS = 8      # Kaggle runtime limit
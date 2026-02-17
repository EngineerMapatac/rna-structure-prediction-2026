# Stanford RNA 3D Folding Part 2 🧬

![Status](https://img.shields.io/badge/Status-Active-success)
![Competition](https://img.shields.io/badge/Kaggle-Stanford_RNA_3D_Folding_2-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

---

## 📌 Project Overview
---
This repository contains the source code and deep learning pipeline for the **Stanford RNA 3D Folding Part 2** competition on Kaggle.
---

**The Goal:**

---

Predict the 3D spatial coordinates (x, y, z) of the **C1' atom** for every residue in a given RNA sequence. The model must generate 5 distinct structural conformations to account for RNA flexibility.

----

**Current Architecture:**

* **Input:** RNA Sequence (A, C, G, U)
* **Model:** Transformer Encoder (Self-Attention mechanism)
* **Output:** 3D Coordinates (x, y, z) for every base.
  
----

## 📂 Repository Structure

```text
stanford-rna-folding-2/
│
├── data/                   # Data files (Excluded from Git)
│   ├── raw/                # Original Kaggle datasets & PDB files
│   └── processed/          # Parsed coordinate tensors
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_visualize_prediction.ipynb   # 3D plotting of results
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── config.py           # Environment path configuration
│   ├── data_loader.py      # PyTorch Dataset & Tokenizer
│   ├── cif_parser.py       # PDB/CIF parsing logic
│   ├── download_samples.py # Automated RCSB downloader
│   ├── create_test_data.py # Generates dummy test sequences
│   ├── fix_csv.py          # Syncs CSVs with downloaded PDBs
│   ├── model.py            # Transformer Architecture
│   ├── train.py            # Training Loop (MSE Loss)
│   └── inference.py        # Submission generation
│
├── outputs/                # Model checkpoints
├── submissions/            # Final Kaggle submission files
└── requirements.txt        # Python dependencies

```

---

🚀 Getting Started

---

1. Installation
   
Clone the repository and install the dependencies:

```

git clone [https://github.com/YOUR_USERNAME/stanford-rna-folding-2.git](https://github.com/YOUR_USERNAME/stanford-rna-folding-2.git)
cd stanford-rna-folding-2
pip install -r requirements.txt
```
--- 

2. Data Setup (Automated)
   
We have included a script to download real RNA structure samples directly from the Protein Data Bank (RCSB), so you don't need to download the full 300GB dataset to start development.

```
# A. Download sample 3D structures from PDB
python src/download_samples.py

# B. Parse the .cif files into coordinate tensors
python src/cif_parser.py

# C. Generate matching training/test CSVs (Fixes empty file errors)
python src/fix_csv.py
python src/create_test_data.py

```
---

3. Training the Model
   
Train the Transformer model on the processed samples. Checkpoints are saved to models/.

```
python -m src.train

```
---

4. Inference & Submissions
   
Generate predictions for the test set. This creates submissions/submission.csv.
```
# Note: Use -m to run as a module (avoids ModuleNotFoundError)
python -m src.inference

```

---

## 🛠️ Usage

Testing the Data Pipeline

To verify that the data loader is correctly reading sequences and aligning them with 3D coordinates:

```
python test_loader.py
(Expected Output: ✅ MATCH FOUND)
```
--- 

## Testing the Model Architecture

To verify the Transformer model accepts RNA sequences and outputs 3D coordinates of the correct shape:

```
python test_model.py
(Expected Output: ✅ SUCCESS: Model produced (x, y, z) coordinates...)

```
---

## 📊 Visualization

To verify the model is actually learning structure:

- Open notebooks/02_visualize_prediction.ipynb.

- Run all cells.

- You should see an interactive 3D plot of the predicted RNA backbone.

---

## ⚠️ Troubleshooting / Known Issues

Disk Space Error: The data/ folder is heavy. Ensure .gitignore includes data/, models/, and *.pt before committing.

ModuleNotFoundError: Always run scripts from the root directory using the -m flag (e.g., python -m src.inference instead of python src/inference.py).

Empty CSV Error: If train_sequences.csv is empty, run python src/fix_csv.py to regenerate it from your downloaded PDBs.

---

## 📝 Roadmap
[x] Repository Setup & Environment

[x] Automated Data Downloader

[x] CIF Parsing & Coordinate Extraction

[x] PyTorch Dataset & DataLoader

[x] Transformer Model Implementation

[ ] Training Loop (Loss Function & Optimizer)

[ ] Validation Pipeline (TM-Score Metric)

[ ] Inference & Submission Generation

---

## 📚 Citation & Acknowledgments
---
Competition: Stanford RNA 3D Folding Part 2

Organizers: Stanford University School of Medicine, NVIDIA Healthcare, AI@HHMI.

---

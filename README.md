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
├── data/                   # Data files (Not synced to GitHub)
│   ├── raw/                # Original Kaggle datasets
│   │   └── PDB_RNA/        # 3D Structure files (.cif)
│   └── processed/          # Parsed tensors and features
│
├── notebooks/              # Jupyter notebooks for experimentation
│   └── 01_data_exploration.ipynb
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── config.py           # Path configurations (Local vs Kaggle)
│   ├── data_loader.py      # PyTorch Dataset & Tokenizer
│   ├── cif_parser.py       # PDB/CIF file parsing logic
│   ├── download_samples.py # Script to fetch sample data from RCSB
│   ├── fix_csv.py          # Utility to sync CSVs with downloaded PDBs
│   └── model.py            # Transformer Neural Network Architecture
│
├── outputs/                # Model checkpoints and submission files
├── .gitignore              # Ignores large data files
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies

```

---
🚀 Getting Started
1. Installation
Clone the repository and install the dependencies:

Bash
git clone [https://github.com/YOUR_USERNAME/stanford-rna-folding-2.git](https://github.com/YOUR_USERNAME/stanford-rna-folding-2.git)
cd stanford-rna-folding-2
pip install -r requirements.txt
2. Data Setup (Automated)
We have included a script to download real RNA structure samples directly from the Protein Data Bank (RCSB), so you don't need to download the full 300GB dataset to start development.

```
# 1. Download sample .cif files
python src/download_samples.py

# 2. Parse the .cif files into coordinate tensors
python src/cif_parser.py

# 3. Generate a matching training CSV file
python src/fix_csv.py

```

## 🛠️ Usage

Testing the Data Pipeline

To verify that the data loader is correctly reading sequences and aligning them with 3D coordinates:

```
python test_loader.py
(Expected Output: ✅ MATCH FOUND)
```
--- 

Testing the Model Architecture

To verify the Transformer model accepts RNA sequences and outputs 3D coordinates of the correct shape:

```
python test_model.py
(Expected Output: ✅ SUCCESS: Model produced (x, y, z) coordinates...)

```

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

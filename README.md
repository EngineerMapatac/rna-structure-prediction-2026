# Stanford RNA 3D Folding Part 2 🧬

![Status](https://img.shields.io/badge/Status-Active-success)
![Competition](https://img.shields.io/badge/Kaggle-Stanford_RNA_3D_Folding_2-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)

## 📌 Project Overview
This repository contains the source code and development pipeline for the **Stanford RNA 3D Folding Part 2** competition on Kaggle.

**The Grand Challenge:**
RNA is essential to life’s core functions, but predicting its 3D structure from sequence alone remains a massive unsolved problem in biology. Unlike proteins (solved by AlphaFold), RNA lacks the massive structural datasets required for easy modeling.

**The Goal:**
Predict the 3D spatial coordinates (x, y, z) of the **C1' atom** for every residue in a given RNA sequence. The model must generate 5 distinct structural conformations to account for RNA flexibility.

## 🏆 Evaluation Metric
Submissions are evaluated using the **TM-score** (Template Modeling score), which measures the topological similarity between the predicted structure and the ground truth (experimental structure).
* **Score Range:** 0.0 (random) to 1.0 (perfect match).
* **Key Constraint:** 8-hour runtime limit for inference on Kaggle.

## 📂 Repository Structure

```text
stanford-rna-folding-2/
│
├── data/                   # Data files (Not synced to GitHub)
│   ├── raw/                # Original Kaggle datasets (.csv, .cif)
│   └── processed/          # Parsed tensors and features
│
├── notebooks/              # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.ipynb
│   └── 02_cif_parsing_demo.ipynb
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── config.py           # Path configurations
│   ├── data_loader.py      # Custom dataset classes
│   └── cif_parser.py       # PDB/CIF file parsing logic
│
├── outputs/                # Model checkpoints and submission files
├── .gitignore              # Ignores large data files
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies


🚀 Getting Started
1. Prerequisites
Python 3.10+

Biopython

PyTorch (or TensorFlow/JAX)

```

2. Installation
Clone the repository and install the dependencies:

Bash
git clone [https://github.com/YOUR_USERNAME/stanford-rna-folding-2.git](https://github.com/YOUR_USERNAME/stanford-rna-folding-2.git)
cd stanford-rna-folding-2
pip install -r requirements.txt
3. Data Setup (Important!)
The dataset for this competition is ~300GB. Do not try to download it all to your local machine unless you have massive storage.

Local Dev: Download a small sample (5-10 .cif files) from the Kaggle Data Page and place them in data/raw/PDB_RNA/.

Kaggle Dev: Use the Kaggle API or Notebook environment to access the full dataset.

🛠️ Usage
Parsing the 3D Structures
To extract the C1' coordinates from the raw .cif files:

Bash
python src/cif_parser.py --input_dir data/raw/PDB_RNA --output_file data/processed/coords.csv
(Note: Ensure your src/config.py points to the correct directories)

📝 Roadmap
[x] Repository Setup & Environment

[ ] Data Pipeline (CIF Parsing & Coordinate Extraction)

[ ] Feature Engineering (MSA Processing)

[ ] Model Architecture (Graph Neural Network / Transformer)

[ ] Training Loop Implementation

[ ] Inference Pipeline (Submission generation)

📚 Citation & Acknowledgments
Competition: Stanford RNA 3D Folding Part 2

Organizers: Stanford University School of Medicine, NVIDIA Healthcare, AI@HHMI.

Citation: Rhiju Das, Youhan Lee, et al. Stanford RNA 3D Folding Part 2. Kaggle, 2026.

Created by John Mapatac
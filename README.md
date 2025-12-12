NSL-KDD CNN–LSTM Intrusion Detection Pipeline

This repository contains a complete experimental pipeline for training, evaluating, and reproducing a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) model on the NSL-KDD intrusion detection dataset. The workflow includes data acquisition, preprocessing, temporal sequence construction, model development, evaluation, and results export.

The project is designed for academic research, enabling transparent reproducibility of all experiments reported in the accompanying study.

1. Project Overview

The NSL-KDD dataset is a widely used benchmark for network intrusion detection. This project implements a modern deep learning approach that transforms the traditionally tabular dataset into temporal sequences using sliding windows, enabling the CNN–LSTM architecture to capture both spatial and temporal patterns that correlate with malicious network activity.

The repository includes:

Complete preprocessing pipeline

Merging and cleaning of the original NSL-KDD CSV files

Label encoding and feature scaling

Temporal sequence construction

CNN–LSTM model implementation

Training scripts with metric logging

Test-time evaluation

Export of processed data and results

2. Dataset Files

The following source files from the NSL-KDD dataset are required:

KDDTrain+.csv

KDDTest+.csv

These files are loaded, validated, and merged during the preprocessing stage. A fully processed evaluation set is exported to:

results/nsl_kdd_eval.csv

3. Environment and Dependencies

The project uses Python 3.10+ and relies on the following core libraries:

TensorFlow / Keras

Scikit-learn

Pandas

NumPy

Matplotlib

Seaborn (optional for visualizations)

Create the environment using:

python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows


Install dependencies:

pip install -r requirements.txt

4. Project Structure
nsl_kdd_dataset/
│
├── data/
│   └── nsl_kdd/
│       ├── KDDTrain+.csv
│       └── KDDTest+.csv
│
├── results/
│   ├── nsl_kdd_eval.csv
│   ├── nsl_eval.json
│   └── plots/
│
├── models/
│   └── lstm_full_final.keras
│
├── notebooks/
│   └── 01_nsl_kdd_ctm_lstm.ipynb
│
└── src/
    ├── preprocessing/
    ├── sequences/
    ├── training/
    └── evaluation/

5. Pipeline Summary
Dataseof different cvs file


Validate schema, check class distribution, inspect missing values

Merge datasets for unified preprocessing

Confirmed zero missing values

Step 2. Encoding and Scaling

Label-encode all categorical variables

Apply MinMax scaling to all numeric and encoded features

Final feature dimension: 135

Step 3. Sequence Construction

Convert tabular rows into sliding-window sequences:

Window length: 30

Stride: 1

Output shape:

X: (35808, 30, 135)

y: (35808,)

Step 4. CNN–LSTM Model

The architecture consists of:

Two 1D convolution layers

A 128-unit LSTM layer

Dropout for regularization

Sigmoid dense output for binary classification

Total trainable parameters: 137,281.

Step 5. Training and Validation

Model trained for 12 epochs with strong and stable convergence.

Step 6. Evaluation

The final model achieved:

Accuracy: 0.9990

F1-score: 0.9993

ROC-AUC: 1.0000

Results saved at:

results/nsl_eval.json

6. Reproducing the Experiment

Run preprocessing:

python src/preprocessing/run_preprocessing.py


Generate sequences:

python src/sequences/build_sequences.py


Train the model:

python src/training/train_lstm.py


Evaluate:

python src/evaluation/evaluate_model.py

7. Results

The CNN–LSTM model demonstrated near-perfect performance on the NSL-KDD test set. Temporal sequences materially improved classification fidelity relative to traditional tabular baselines. Full metrics and plots are available in the results/ directory.

8. Citation

If you use this repository in academic work, please cite:

NSL-KDD Dataset:
M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, "A Detailed Analysis of the KDD CUP 99 Data Set," 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications.

9. Contact

For research collaboration inquiries or questions related to the pipeline, please open an issue or submit a pull request through the GitHub repository.

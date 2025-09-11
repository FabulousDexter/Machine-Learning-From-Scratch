# Logistic Regression for Breast Cancer Diagnosis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

This project implements a logistic regression model from scratch using Python, `numpy`, and `cupy` for GPU acceleration. The model is trained on the Wisconsin Breast Cancer dataset to classify tumors as malignant or benign.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
  
## Project Overview

The primary goal of this project is to build and understand the mechanics of logistic regression. This includes:
- **Data Preprocessing**: Loading and scaling the data using `StandardScaler`.
- **Model Implementation**: Building the core components of logistic regression from the ground up:
    - Sigmoid function
    - Cost function (Binary Cross-Entropy)
    - Gradient Descent for optimization
- **GPU Acceleration**: Using `cupy` to speed up computations.
- **Model Evaluation**: Assessing performance using a confusion matrix and classification report (precision, recall, F1-score).
- **Model Improvement**:
    - Adjusting the prediction threshold to improve recall for the malignant class.
    - Comparing the custom implementation with `scikit-learn`'s `LogisticRegression` for a performance baseline.

## Dataset

The dataset used is the [Wisconsin Breast Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), which is included in the `dataset/` directory. This dataset contains:

- **569 samples** of breast cancer biopsies
- **30 features** computed from digitized images of cell nuclei
- **2 classes**: Malignant (M) and Benign (B)
- Features include radius, texture, perimeter, area, smoothness, compactness, concavity, etc.

## Features

- ✅ **From-scratch implementation** of logistic regression
- ✅ **GPU acceleration** using CuPy for faster computations
- ✅ **Comprehensive data preprocessing** with StandardScaler
- ✅ **Custom optimization** using gradient descent
- ✅ **Model evaluation** with confusion matrix and classification metrics
- ✅ **Threshold tuning** for optimal recall
- ✅ **Comparison** with scikit-learn's implementation
- ✅ **Visualization** of training progress and results

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for CuPy acceleration)
- Jupyter Notebook or VS Code with Python extension

### Setup
1. Clone the repository:
```bash
git clone https://github.com/FabulousDartier/python-logistic-regression.git
cd python-logistic-regression
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
# For GPU acceleration (replace XX with your CUDA version, e.g., cuda11x):
pip install cupy-cudaXX
```

## Usage

## Usage

1. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Open the notebook:**
Navigate to `notebooks/logistic_regression.ipynb` and run all cells sequentially.

3. **Alternative - VS Code:**
Open the project folder in VS Code and run the notebook cells directly.

The notebook will automatically:
- Load and preprocess the breast cancer dataset
- Train the logistic regression model from scratch
- Evaluate model performance
- Compare with scikit-learn's implementation
- Display visualizations and metrics

## Results

### Model Performance
- **Accuracy**: 97% on test set
- **Precision**: 96% (malignant class)
- **Recall**: 98% (malignant class) - optimized to minimize false negatives
- **F1-Score**: 97%

### Key Findings
- The from-scratch implementation performs comparably to scikit-learn's LogisticRegression
- GPU acceleration with CuPy provides significant speedup for large datasets
- Threshold tuning successfully improved recall for the critical malignant class
- The model effectively distinguishes between malignant and benign tumors

## Project Structure

```
python-logistic-regression/
│
├── README.md                 # Project documentation
├── dataset/
│   └── data.csv             # Wisconsin Breast Cancer dataset
├── notebooks/
│   └── logistic_regression.ipynb  # Main implementation notebook
└── .gitignore               # Git ignore file
```

## Technologies Used

- **Python 3.8+** - Programming language
- **NumPy** - Numerical computing
- **CuPy** - GPU-accelerated computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning utilities and comparison
- **Jupyter Notebook** - Interactive development environment

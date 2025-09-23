# Machine Learning From Scratch 🤖

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive collection of machine learning algorithms implemented from scratch in Python. This repository serves as an educational resource for understanding the fundamental concepts and mathematical foundations behind popular ML algorithms.

## 🎯 Project Overview

This project aims to implement core machine learning algorithms without relying on high-level libraries like scikit-learn. Each implementation focuses on:

- **Mathematical Understanding**: Clear explanation of the underlying mathematics
- **From-Scratch Implementation**: Using only NumPy and basic Python libraries
- **Educational Value**: Detailed comments and step-by-step explanations
- **Practical Applications**: Real-world datasets and use cases
- **Performance Analysis**: Visualization and evaluation metrics

## 📊 Progress Tracker

### Supervised Learning

#### Regression
- [x] **[Linear Regression](Supervised%20Learning/Regression/python-linear-regression/linear_regression.ipynb)** ✅
  - Salary prediction based on years of experience
  - Gradient descent optimization
  - Cost function visualization

#### Classification
- [x] **[Logistic Regression](Supervised%20Learning/Classification/python-logistic-regression/logistic_regression.ipynb)** ✅
  - Breast cancer diagnosis (Wisconsin dataset)
  - Sigmoid function implementation
  - GPU acceleration with CuPy
- [x] **[K-Nearest Neighbors (KNN)](Supervised%20Learning/Classification/python-k-nearest-neighbors/knn.ipynb)** ✅
  - Breast cancer classification
  - Custom distance metrics
  - Feature scaling and preprocessing
- [x] **[Naive Bayes](Supervised%20Learning/Classification/naive-bayes-model/naive_bayes.ipynb)** ✅
  - Gaussian Naive Bayes implementation
  - Probability calculations from scratch
  - Comparison with scikit-learn
- [x] **[Decision Trees](Supervised%20Learning/Classification/decision-trees/decision_trees.ipynb)** ✅
  - Recursive tree building algorithm
  - Information gain and entropy calculations
  - Tree visualization and pruning techniques
- [x] **[Support Vector Machine (SVM)](Supervised%20Learning/Classification/support-vector-machines/svm.ipynb)** ✅
  - Gradient descent optimization
  - Hinge loss implementation
  - Margin maximization and hyperplane finding
- [x] **[Random Forest](Supervised%20Learning/Classification/random-forest/random-forest.ipynb)** ✅
  - Bootstrap sampling for diverse training sets
  - Feature randomness at each split
  - Ensemble voting for robust predictions

### Unsupervised Learning
- [ ] **Principal Component Analysis (PCA)**
- [ ] **K-Means Clustering**

### Deep Learning
- [ ] **Neural Network**
  - Multi-layer perceptron
  - Backpropagation algorithm
  - Activation functions

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib jupyter seaborn
```

### Running the Projects

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Machine-Learning-From-Scratch.git
   cd Machine-Learning-From-Scratch
   ```

2. **Navigate to any algorithm folder:**
   ```bash
   cd "Supervised Learning/Classification/python-logistic-regression"
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Open and run the implementation notebook**

## 📁 Project Structure

```
Machine-Learning-From-Scratch/
├── README.md
├── Supervised Learning/
│   ├── Regression/
│   │   └── python-linear-regression/
│   │       ├── linear_regression.ipynb
│   │       ├── Salary_dataset.csv
│   │       └── README.md
│   └── Classification/
│       ├── python-logistic-regression/
│       │   ├── logistic_regression.ipynb
│       │   ├── dataset.csv
│       │   └── README.md
│       ├── python-k-nearest-neighbors/
│       │   ├── knn.ipynb
│       │   ├── dataset.csv
│       │   └── README.md
│       ├── naive-bayes-model/
│       │   ├── naive_bayes.ipynb
│       │   ├── data.csv
│       │   └── README.md
│       ├── decision-trees/
│       │   ├── decision_trees.ipynb
│       │   ├── data.csv
│       │   └── README.md
│       └── support-vector-machines/
│           ├── svm.ipynb
│           ├── data.csv
│           └── README.md
├── Unsupervised Learning/
│   └── placeholder.txt
└── Neural Network/
    └── placeholder.txt
```

## 🔬 Implemented Algorithms

### 1. [Linear Regression](Supervised%20Learning/Regression/python-linear-regression/linear_regression.ipynb)
- **Dataset**: Salary prediction dataset
- **Key Features**: Gradient descent, cost function optimization
- **Use Case**: Predicting employee salaries based on experience
- **Location**: `Supervised Learning/Regression/python-linear-regression/`

### 2. [Logistic Regression](Supervised%20Learning/Classification/python-logistic-regression/logistic_regression.ipynb)
- **Dataset**: Wisconsin Breast Cancer dataset
- **Key Features**: Sigmoid function, binary classification, GPU acceleration
- **Use Case**: Medical diagnosis (malignant vs benign tumors)
- **Location**: `Supervised Learning/Classification/python-logistic-regression/`

### 3. [K-Nearest Neighbors (KNN)](Supervised%20Learning/Classification/python-k-nearest-neighbors/knn.ipynb)
- **Dataset**: Breast cancer diagnostic dataset
- **Key Features**: Custom distance metrics, feature scaling
- **Use Case**: Tumor classification based on nearest neighbors
- **Location**: `Supervised Learning/Classification/python-k-nearest-neighbors/`

### 4. [Naive Bayes](Supervised%20Learning/Classification/naive-bayes-model/naive_bayes.ipynb)
- **Dataset**: Breast cancer dataset
- **Key Features**: Gaussian probability distributions, Bayes theorem
- **Use Case**: Probabilistic classification with assumption of feature independence
- **Location**: `Supervised Learning/Classification/naive-bayes-model/`

### 5. [Decision Trees](Supervised%20Learning/Classification/decision-trees/decision_trees.ipynb)
- **Dataset**: Breast cancer dataset
- **Key Features**: Information gain, entropy calculations, recursive tree building
- **Use Case**: Rule-based classification with interpretable decision paths
- **Location**: `Supervised Learning/Classification/decision-trees/`

### 6. [Support Vector Machine (SVM)](Supervised%20Learning/Classification/support-vector-machines/svm.ipynb)
- **Dataset**: Breast cancer dataset
- **Key Features**: Margin maximization, hinge loss, gradient descent optimization
- **Use Case**: Maximum margin classification for linearly separable data
- **Location**: `Supervised Learning/Classification/support-vector-machines/`

### 7. [Random Forest](Supervised%20Learning/Classification/random-forest/random-forest.ipynb)
- **Dataset**: Breast cancer dataset
- **Key Features**: Bootstrap sampling, feature randomness, ensemble voting
- **Use Case**: Robust classification using multiple decision trees with voting
- **Location**: `Supervised Learning/Classification/random-forest/`

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment
- **CuPy**: GPU acceleration (for logistic regression)

## 📚 Learning Resources

Each algorithm implementation includes:

- **Mathematical Foundation**: Detailed explanation of the underlying mathematics
- **Step-by-Step Implementation**: Code broken down into digestible functions
- **Visualization**: Plots showing algorithm behavior and results
- **Performance Metrics**: Accuracy, precision, recall, and other relevant metrics
- **Comparison**: Performance comparison with scikit-learn implementations

## 🎓 Educational Value

This repository is designed for:

- **Students** learning machine learning fundamentals
- **Developers** wanting to understand algorithms beyond black-box usage
- **Researchers** needing to modify or extend existing algorithms
- **Interview Preparation** for data science and ML engineering roles

## 🏆 Roadmap

### Phase 1: Core Algorithms (Completed ✅)
- [x] Linear Regression
- [x] Logistic Regression  
- [x] K-Nearest Neighbors
- [x] Naive Bayes
- [x] Decision Trees
- [x] Random Forest

### Phase 2: Advanced Algorithms
- [x] Support Vector Machines
- [ ] Principal Component Analysis
- [ ] K-Means Clustering

### Phase 3: Deep Learning
- [ ] Neural Networks (Multi-layer Perceptron)
- [ ] Backpropagation Algorithm
- [ ] Various Activation Functions

*Building ML algorithms from scratch to truly understand the magic behind the models.*


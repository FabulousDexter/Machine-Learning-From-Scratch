# Machine Learning From Scratch 🤖

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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
- [x] **Linear Regression** ✅
  - Salary prediction based on years of experience
  - Gradient descent optimization
  - Cost function visualization

#### Classification
- [x] **Logistic Regression** ✅
  - Breast cancer diagnosis (Wisconsin dataset)
  - Sigmoid function implementation
  - GPU acceleration with CuPy
- [x] **K-Nearest Neighbors (KNN)** ✅
  - Breast cancer classification
  - Custom distance metrics
  - Feature scaling and preprocessing
- [x] **Naive Bayes** ✅
  - Gaussian Naive Bayes implementation
  - Probability calculations from scratch
  - Comparison with scikit-learn

#### Upcoming Implementations
- [ ] **Decision Trees**
- [ ] **Random Forests**
- [ ] **Support Vector Machine (SVM)**

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
│       └── naive-bayes-model/
│           ├── naive_bayes.ipynb
│           ├── data.csv
│           └── README.md
├── Unsupervised Learning/
│   └── placeholder.txt
└── Neural Network/
    └── placeholder.txt
```

## 🔬 Implemented Algorithms

### 1. Linear Regression
- **Dataset**: Salary prediction dataset
- **Key Features**: Gradient descent, cost function optimization
- **Use Case**: Predicting employee salaries based on experience
- **Location**: `Supervised Learning/Regression/python-linear-regression/`

### 2. Logistic Regression
- **Dataset**: Wisconsin Breast Cancer dataset
- **Key Features**: Sigmoid function, binary classification, GPU acceleration
- **Use Case**: Medical diagnosis (malignant vs benign tumors)
- **Location**: `Supervised Learning/Classification/python-logistic-regression/`

### 3. K-Nearest Neighbors (KNN)
- **Dataset**: Breast cancer diagnostic dataset
- **Key Features**: Custom distance metrics, feature scaling
- **Use Case**: Tumor classification based on nearest neighbors
- **Location**: `Supervised Learning/Classification/python-k-nearest-neighbors/`

### 4. Naive Bayes
- **Dataset**: Breast cancer dataset
- **Key Features**: Gaussian probability distributions, Bayes theorem
- **Use Case**: Probabilistic classification with assumption of feature independence
- **Location**: `Supervised Learning/Classification/naive-bayes-model/`

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Implement your algorithm following the existing structure
4. Add comprehensive documentation and comments
5. Include a detailed README for your implementation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Roadmap

### Phase 1: Core Algorithms (In Progress)
- [x] Linear Regression
- [x] Logistic Regression  
- [x] K-Nearest Neighbors
- [x] Naive Bayes
- [ ] Decision Trees
- [ ] Random Forests

### Phase 2: Advanced Algorithms
- [ ] Support Vector Machines
- [ ] Principal Component Analysis
- [ ] K-Means Clustering

### Phase 3: Deep Learning
- [ ] Neural Networks (Multi-layer Perceptron)
- [ ] Backpropagation Algorithm
- [ ] Various Activation Functions

### Phase 4: Enhancements
- [ ] Performance optimizations
- [ ] Additional datasets
- [ ] Interactive visualizations
- [ ] Algorithm comparison framework

## 📬 Contact

Feel free to reach out if you have questions or suggestions!

---

⭐ **Star this repository if you find it helpful!** ⭐

*Building ML algorithms from scratch to truly understand the magic behind the models.*


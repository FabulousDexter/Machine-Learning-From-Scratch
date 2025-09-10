# Naive Bayes Classification Model

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project implementing Gaussian Naive Bayes classifier from scratch for breast cancer diagnosis, featuring both custom implementation and scikit-learn comparison with detailed performance analysis.

## 📋 Project Overview

This project demonstrates the implementation of a Naive Bayes classifier for binary classification of breast cancer tumors as either benign (B) or malignant (M). The implementation focuses on understanding the mathematical foundations and provides educational value through:

- **Custom Naive Bayes Implementation**: Built from scratch using only NumPy and basic Python
- **Scikit-learn Comparison**: Performance benchmarking against GaussianNB
- **Data Visualization**: Comprehensive scatter plots showing data distribution and decision boundaries
- **Performance Evaluation**: Detailed accuracy metrics and confusion matrices
- **Mathematical Explanation**: Step-by-step breakdown of Bayes' theorem application

## 📊 Dataset

The project uses the Wisconsin Breast Cancer dataset with the following characteristics:

- **Total Samples**: 569 cases
- **Features**: 30 numerical features (mean, standard error, and worst values for 10 real-valued features)
- **Target Classes**: 
  - `B` - Benign (non-cancerous) - 357 cases (62.7%)
  - `M` - Malignant (cancerous) - 212 cases (37.3%)
- **Train/Test Split**: 80% training (455 samples), 20% testing (114 samples)

### Key Features Include:
- **Geometric Features**: Radius, perimeter, area
- **Texture Features**: Texture, smoothness, symmetry
- **Shape Features**: Compactness, concavity, concave points, fractal dimension

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn jupyter scikit-learn
```

### Running the Project

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Supervised Learning/Classification/naive-bayes-model"
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook naive_bayes.ipynb
   ```

3. **Run all cells** to see the complete analysis including:
   - Data exploration and visualization
   - Custom Naive Bayes implementation
   - Model training and evaluation
   - Comparison with scikit-learn

## 🔧 Implementation Details

### Custom Naive Bayes Algorithm

The custom implementation follows these key steps:

#### 1. **Training Phase**:
   - Calculate prior probabilities for each class: P(Benign) and P(Malignant)
   - Compute mean (μ) and standard deviation (σ) for each feature per class
   - Store class statistics for prediction phase

#### 2. **Prediction Phase**:
   - Apply Gaussian probability density function for each feature
   - Calculate likelihood: P(features|class) using Gaussian PDF
   - Apply Bayes' theorem to compute posterior probabilities
   - Predict class with highest posterior probability

### Mathematical Foundation

The classifier implements Bayes' theorem:

```
P(class|features) = P(features|class) × P(class) / P(features)
```

**Where:**
- `P(class|features)` - Posterior probability (what we want to find)
- `P(features|class)` - Likelihood (Gaussian PDF)
- `P(class)` - Prior probability (class frequency in training data)
- `P(features)` - Evidence (normalization constant)

**Gaussian Probability Density Function:**
```
P(x|class) = (1/√(2πσ²)) × exp(-((x-μ)²)/(2σ²))
```

## 📈 Results

### Performance Metrics

Both implementations achieve excellent accuracy on the breast cancer dataset:

| Implementation | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| **Custom Naive Bayes** | 96.49% | 96.2% | 96.5% | 96.3% |
| **Scikit-learn GaussianNB** | 97.37% | 97.1% | 97.4% | 97.2% |

### Key Insights

- **High Performance**: Both implementations achieve >96% accuracy
- **Minimal Difference**: Less than 1% accuracy difference between custom and scikit-learn
- **Robust Classification**: Strong performance on medical diagnostic data
- **Educational Value**: Custom implementation demonstrates algorithm mechanics

## 📁 Project Structure

```
naive-bayes-model/
├── naive_bayes.ipynb          # Main implementation notebook
├── data.csv                   # Breast cancer dataset
├── README.md                  # This documentation
└── requirements.txt           # Python dependencies (if needed)
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and preprocessing
- **Matplotlib/Seaborn**: Data visualization and plotting
- **Scikit-learn**: Performance comparison and data preprocessing
- **Jupyter Notebook**: Interactive development environment

## 🎓 Learning Objectives

This project teaches:

1. **Bayes' Theorem**: Practical application in machine learning
2. **Probability Theory**: Understanding of conditional probabilities
3. **Gaussian Distribution**: Implementation of probability density functions
4. **Feature Independence**: Naive assumption and its implications
5. **Model Evaluation**: Accuracy metrics and performance comparison
6. **Data Preprocessing**: Handling real-world medical datasets

## 🔍 Algorithm Assumptions

**Naive Bayes makes the following assumptions:**

1. **Feature Independence**: Features are conditionally independent given the class
2. **Gaussian Distribution**: Continuous features follow normal distribution
3. **No Missing Values**: Complete dataset required for training
4. **Sufficient Training Data**: Adequate samples for reliable statistics

## 🧪 Experimental Setup

### Data Preprocessing
- No missing values in the dataset
- Features used as-is (no normalization required for Naive Bayes)
- Random train-test split with fixed seed for reproducibility

### Hyperparameters
- **Train/Test Ratio**: 80/20
- **Random State**: 42 (for reproducible results)
- **Smoothing**: None (sufficient data available)

## 🚧 Future Enhancements

- [ ] **Multinomial Naive Bayes**: Implementation for categorical features
- [ ] **Laplace Smoothing**: Handle zero probabilities
- [ ] **Cross-Validation**: More robust performance evaluation
- [ ] **Feature Selection**: Identify most important diagnostic features
- [ ] **Hyperparameter Tuning**: Optimize smoothing parameters

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional evaluation metrics
- Feature importance analysis
- Visualization enhancements
- Documentation improvements

## 📄 License

This project is part of the Machine Learning From Scratch collection and follows the same MIT License.

---

**📝 Note**: This implementation is for educational purposes to understand the inner workings of Naive Bayes classification. For production use, consider using optimized libraries like scikit-learn.

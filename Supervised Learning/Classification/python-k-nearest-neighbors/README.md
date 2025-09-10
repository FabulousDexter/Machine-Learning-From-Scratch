# K-Nearest Neighbors (k-NN) Classification

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project implementing the k-Nearest Neighbors (k-NN) algorithm from scratch for breast cancer diagnosis, featuring custom distance calculations, feature scaling, and detailed performance analysis.

## 📋 Project Overview

This project demonstrates the implementation of a k-Nearest Neighbors classifier for binary classification of breast cancer tumors as either benign (B) or malignant (M). The implementation focuses on understanding the algorithm mechanics and provides educational value through:

- **Custom k-NN Implementation**: Built from scratch using only NumPy and basic Python
- **Distance Metrics**: Implementation of Euclidean distance calculations
- **Data Preprocessing**: Feature scaling and normalization for optimal performance
- **Visualization**: Comprehensive scatter plots showing data distribution and predictions
- **Performance Evaluation**: Detailed accuracy metrics and model evaluation
- **Hyperparameter Analysis**: Effect of different k values on classification performance

## 📊 Dataset

The project uses the Wisconsin Breast Cancer dataset with the following characteristics:

- **Total Samples**: 569 cases
- **Features**: 30 numerical features (mean, standard error, and worst values for 10 real-valued features)
- **Target Classes**: 
  - `B` - Benign (non-cancerous) - 357 cases (62.7%)
  - `M` - Malignant (cancerous) - 212 cases (37.3%)
- **Train/Test Split**: 80% training, 20% testing

### Key Features Include:
- **Geometric Features**: Radius, perimeter, area
- **Texture Features**: Texture, smoothness, symmetry
- **Shape Features**: Compactness, concavity, concave points, fractal dimension

### Preprocessing Steps
1. **Data Cleaning**: Dropped irrelevant columns (e.g., `Unnamed: 32`, `id`)
2. **Train/Test Split**: Random 80/20 split with fixed seed for reproducibility
3. **Feature Scaling**: Applied StandardScaler to normalize features for distance calculations

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn jupyter scikit-learn
```

### Running the Project

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Supervised Learning/Classification/python-k-nearest-neighbors"
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook knn.ipynb
   ```

3. **Run all cells** to see the complete analysis including:
   - Data exploration and preprocessing
   - Custom k-NN implementation
   - Model training and evaluation
   - Visualization of results

## 🔧 Implementation Details

### Custom k-NN Algorithm

The custom implementation includes a complete `KNN` class with the following methods:

#### **Core Components:**

1. **`__init__(self, k=3)`**: Initialize the classifier with k value
2. **`fit(self, X_train, y_train)`**: Store training data (lazy learning)
3. **`predict(self, X_test)`**: Predict labels for test samples
4. **`_euclidean_distance(self, point1, point2)`**: Calculate distance between points

#### **Algorithm Steps:**

1. **Training Phase** (Lazy Learning):
   - Store all training samples and labels
   - No explicit model training required

2. **Prediction Phase**:
   - For each test sample:
     - Calculate Euclidean distance to all training samples
     - Find k nearest neighbors
     - Assign majority class among k neighbors
     - Return predicted class

### Mathematical Foundation

**Euclidean Distance Formula:**
```
d(p,q) = √(Σ(pi - qi)²)
```

**k-NN Decision Rule:**
```
ŷ = argmax(Σ I(yi = c))
```

Where:
- `d(p,q)` is the Euclidean distance between points p and q
- `ŷ` is the predicted class
- `I(yi = c)` is an indicator function (1 if neighbor i has class c, 0 otherwise)

## 📈 Results

### Performance Metrics

Both custom and scikit-learn implementations achieve excellent accuracy on the breast cancer dataset:

| Implementation | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| **Custom k-NN (k=5)** | 95.61% | 95.3% | 95.6% | 95.4% |
| **Scikit-learn KNeighborsClassifier** | 96.49% | 96.2% | 96.5% | 96.3% |

### Performance Analysis

The k-NN implementation achieves excellent accuracy on the breast cancer dataset:

| Metric | Custom Implementation | Scikit-learn |
|--------|----------------------|--------------|
| **Best Accuracy** | 95.61% (k=5) | 96.49% (k=5) |
| **Optimal k Range** | 3-7 | 3-7 |
| **Training Time** | O(1) - Lazy learning | O(1) - Lazy learning |
| **Prediction Time** | O(n·d) per sample | O(n·d) per sample |

### Hyperparameter Analysis

**Effect of k Value:**
- **k=1**: High variance, prone to overfitting
- **k=3-7**: Optimal balance between bias and variance
- **k>10**: Increased bias, potential underfitting

### Key Insights

- **Comparable Performance**: Custom implementation achieves 95.61% vs scikit-learn's 96.49%
- **Minimal Difference**: Less than 1% accuracy difference between implementations
- **Distance Sensitivity**: Feature scaling crucial for both implementations
- **Local Decision Boundaries**: Both adapt well to local patterns
- **Educational Value**: Custom implementation demonstrates algorithm mechanics without sacrificing much performance

## 📁 Project Structure

```
python-k-nearest-neighbors/
├── knn.ipynb                  # Main implementation notebook
├── dataset.csv                # Breast cancer dataset
├── README.md                  # This documentation
└── requirements.txt           # Python dependencies (if needed)
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and preprocessing
- **Matplotlib/Seaborn**: Data visualization and plotting
- **Scikit-learn**: Data preprocessing (StandardScaler)
- **Jupyter Notebook**: Interactive development environment

## 🎓 Learning Objectives

This project teaches:

1. **Instance-Based Learning**: Understanding lazy learning algorithms
2. **Distance Metrics**: Implementation of Euclidean distance
3. **Hyperparameter Tuning**: Effect of k value on model performance
4. **Feature Scaling**: Importance of normalization in distance-based algorithms
5. **Bias-Variance Tradeoff**: How k affects model complexity
6. **Classification Boundaries**: Non-linear decision boundaries

## 🔍 Algorithm Characteristics

**k-NN Algorithm Properties:**

1. **Lazy Learning**: No training phase, stores all data
2. **Non-Parametric**: Makes no assumptions about data distribution
3. **Instance-Based**: Decisions based on local neighborhood
4. **Memory-Intensive**: Stores entire training dataset
5. **Distance-Sensitive**: Requires feature scaling for optimal performance

## 🧪 Experimental Setup

### Data Preprocessing
- **Missing Values**: None in the dataset
- **Feature Scaling**: StandardScaler applied to all features
- **Train/Test Split**: 80/20 with random_state=42

### Hyperparameters
- **k Value**: Tested multiple values (1, 3, 5, 7, 9)
- **Distance Metric**: Euclidean distance
- **Scaling Method**: StandardScaler (zero mean, unit variance)

## 🚧 Future Enhancements

- [ ] **Multiple Distance Metrics**: Manhattan, Minkowski, Cosine distance
- [ ] **Weighted k-NN**: Distance-weighted voting
- [ ] **Cross-Validation**: More robust k selection
- [ ] **Dimensionality Reduction**: PCA preprocessing
- [ ] **Approximate Nearest Neighbors**: For large datasets
- [ ] **Performance Optimization**: KD-tree or Ball-tree implementation

## 📚 Algorithm Complexity

### Time Complexity
- **Training**: O(1) - Lazy learning
- **Prediction**: O(n·d·m) where n=training samples, d=features, m=test samples
- **Space**: O(n·d) - Store all training data

### Space Complexity
- **Memory Usage**: O(n·d) for storing training data
- **Scalability**: Limited by memory for large datasets

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Alternative distance metrics implementation
- Performance optimizations (KD-tree, LSH)
- Advanced visualization techniques
- Cross-validation framework

## 📄 License

This project is part of the Machine Learning From Scratch collection and follows the same MIT License.

---

**📝 Note**: This implementation is for educational purposes to understand the mechanics of k-NN classification. For production use with large datasets, consider optimized libraries like scikit-learn with efficient data structures.

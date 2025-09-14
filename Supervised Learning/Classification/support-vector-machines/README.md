# Support Vector Machine (SVM) Classification Model

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive machine learning project implementing Support Vector Machine classifier from scratch for breast cancer diagnosis, featuring both custom implementation and scikit-learn comparison with detailed mathematical foundations and performance analysis.

## 📋 Project Overview

This project demonstrates the implementation of a Support Vector Machine classifier for binary classification of breast cancer tumors as either benign (B) or malignant (M). The implementation focuses on understanding the mathematical foundations and provides educational value through:

- **Custom SVM Implementation**: Built from scratch using gradient descent and hinge loss
- **Scikit-learn Comparison**: Performance benchmarking against SVC with RBF kernel
- **Mathematical Theory**: Comprehensive explanation of SVM optimization and margin maximization
- **Data Visualization**: Scatter plots showing data distribution and class separation
- **Performance Evaluation**: Detailed accuracy metrics and algorithm comparison
- **Feature Scaling**: Demonstration of preprocessing importance for SVM performance

## 📊 Dataset

The project uses the Wisconsin Breast Cancer dataset with the following characteristics:

- **Total Samples**: 569 cases
- **Features**: 30 numerical features (mean, standard error, and worst values for 10 real-valued features)
- **Target Classes**: 
  - `B` - Benign (non-cancerous) - 357 cases (62.7%)
  - `M` - Malignant (cancerous) - 212 cases (37.3%)
- **Train/Test Split**: 80% training (455 samples), 20% testing (114 samples)
- **Encoding**: Binary encoding (Malignant = 1, Benign = -1)

### Key Features Include:
- **Geometric Features**: Radius, perimeter, area
- **Texture Features**: Texture, smoothness, symmetry
- **Shape Features**: Compactness, concavity, concave points, fractal dimension

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Running the Project

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Supervised Learning/Classification/support-vector-machines"
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook svm.ipynb
   ```

3. **Run all cells** to see the complete analysis including:
   - Data exploration and visualization
   - Custom SVM implementation from scratch
   - Model training with gradient descent
   - Performance evaluation and comparison
   - Mathematical theory with detailed explanations

## 🔧 Implementation Details

### Custom SVM Algorithm

The custom implementation follows these key steps:

#### 1. **Initialization**:
   - Random weight vector initialization: `w ~ N(0, 0.01)`
   - Bias term initialization: `b = 0`
   - Hyperparameter setup: learning rate, regularization, iterations

#### 2. **Training Phase (Gradient Descent)**:
   - **Mini-batch Processing**: Efficient batch-wise gradient computation
   - **Margin Calculation**: Compute `margin = y_i(w^T x_i + b)` for each sample
   - **Gradient Computation**: Apply hinge loss derivatives
   - **Parameter Updates**: Update weights and bias using computed gradients

#### 3. **Prediction Phase**:
   - Calculate decision function: `f(x) = w^T x + b`
   - Apply sign function: `prediction = sign(f(x))`
   - Return class labels: {-1, 1}

### Mathematical Foundation

#### **SVM Optimization Problem**

The SVM aims to find the optimal hyperplane that maximizes the margin between classes:

```
Minimize: (1/2)||w||² + C∑ξᵢ
Subject to: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

**Where:**
- `w` - Weight vector (normal to hyperplane)
- `b` - Bias term
- `C` - Regularization parameter (trade-off between margin and errors)
- `ξᵢ` - Slack variables (allow soft margin)

#### **Decision Function**

For a data point **x**, the SVM decision function is:
```
f(x) = w^T x + b
```

**Classification Rule:**
```
Class = sign(f(x)) = {
  +1  if f(x) ≥ 0  (Malignant)
  -1  if f(x) < 0   (Benign)
}
```

#### **Margin and Support Vectors**

The margin is the distance from the hyperplane to the nearest data points:
```
Margin Width = 2/||w||
```

**Support Vectors** are points that lie on the margin boundary:
```
yᵢ(w^T xᵢ + b) = 1
```

#### **Hinge Loss Function**

The hinge loss measures classification error with margin penalty:
```
L_hinge(y, f(x)) = max(0, 1 - y·f(x))
```

**Loss Interpretation:**
- `L = 0` when margin ≥ 1 (correct classification with sufficient margin)
- `L > 0` when margin < 1 (support vector or misclassification)

#### **Gradient Computation**

For gradient descent optimization:

**Weight Gradient:**
```
∂L/∂w = {
  λw                    if yᵢ(w^T xᵢ + b) ≥ 1
  λw - yᵢxᵢ           if yᵢ(w^T xᵢ + b) < 1
}
```

**Bias Gradient:**
```
∂L/∂b = {
  0                     if yᵢ(w^T xᵢ + b) ≥ 1
  -yᵢ                  if yᵢ(w^T xᵢ + b) < 1
}
```

**Parameter Updates:**
```
w ← w - η(∂L/∂w)
b ← b - η(∂L/∂b)
```

Where `η` is the learning rate and `λ` is the regularization parameter.

## 📈 Results

### Performance Metrics

Both implementations achieve excellent accuracy on the breast cancer dataset:

| Implementation | Accuracy | Algorithm | Kernel | Key Features |
|----------------|----------|-----------|---------|--------------|
| **Custom SVM** | 98.25% | Gradient Descent | Linear | Hinge Loss, Mini-batch |
| **Scikit-learn SVC** | 98.25% | SMO Algorithm | RBF | Optimized C++ Implementation |

### Key Insights

- **Excellent Performance**: Both implementations achieve 98.25% accuracy
- **Identical Results**: Perfect match between custom and scikit-learn implementations
- **Robust Classification**: Outstanding performance on medical diagnostic data
- **Educational Value**: Custom implementation demonstrates core SVM mechanics
- **Scaling Importance**: Feature standardization crucial for convergence

### Hyperparameters Used

**Custom Implementation:**
- Learning Rate: 0.0001
- Regularization (λ): 0.1
- Iterations: 1000
- Batch Size: 64

**Scikit-learn SVC:**
- Kernel: RBF (Radial Basis Function)
- Regularization (C): 1.0
- Default gamma: 'scale'

## 📁 Project Structure

```
support-vector-machines/
├── svm.ipynb                  # Main implementation notebook
├── README.md                  # This comprehensive documentation
└── requirements.txt           # Python dependencies (if needed)
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and preprocessing
- **Matplotlib**: Data visualization and plotting
- **Scikit-learn**: Performance comparison and preprocessing utilities
- **Jupyter Notebook**: Interactive development environment

## 🎓 Learning Objectives

This project teaches:

1. **SVM Theory**: Understanding hyperplanes, margins, and support vectors
2. **Optimization**: Gradient descent for convex optimization problems
3. **Hinge Loss**: Loss functions for margin-based classification
4. **Regularization**: Balancing model complexity and generalization
5. **Feature Scaling**: Preprocessing importance for distance-based algorithms
6. **Mathematical Modeling**: Translating theory into practical implementation

## 🔍 Algorithm Assumptions

**SVM makes the following assumptions:**

1. **Linear Separability**: Data classes can be separated by hyperplane (soft margin allows violations)
2. **Feature Independence**: No specific independence requirements
3. **Scaling Sensitivity**: Features should be on similar scales
4. **Convex Optimization**: Problem has unique global optimum
5. **Sufficient Data**: Adequate samples for reliable margin estimation

## 🧪 Experimental Setup

### Data Preprocessing
- **Missing Values**: None in the dataset
- **Feature Scaling**: StandardScaler normalization (μ=0, σ=1)
- **Encoding**: Binary labels (M=1, B=-1)
- **Train/Test Split**: 80/20 with fixed random seed

### Training Configuration
- **Optimization**: Mini-batch gradient descent
- **Convergence**: Fixed iterations (1000 epochs)
- **Batch Processing**: 64 samples per batch
- **Regularization**: L2 penalty on weights

## 📊 Mathematical Workflow

### 1. **Data Preparation**
```
X ← StandardScaler(features)
y ← {1 for Malignant, -1 for Benign}
```

### 2. **Model Training**
```
For each epoch:
  For each mini-batch:
    1. Compute margins: mᵢ = yᵢ(w^T xᵢ + b)
    2. Identify support vectors: mᵢ < 1
    3. Calculate gradients using hinge loss
    4. Update parameters: w, b ← w - η∇w, b - η∇b
```

### 3. **Prediction**
```
For new sample x:
  1. Compute f(x) = w^T x + b
  2. Predict class = sign(f(x))
```

## 🚧 Future Enhancements

- [ ] **Kernel Implementation**: Add RBF, polynomial kernels to custom SVM
- [ ] **SMO Algorithm**: Implement Sequential Minimal Optimization
- [ ] **Hyperparameter Tuning**: Grid search for optimal parameters
- [ ] **Multi-class Extension**: One-vs-one or one-vs-all strategies
- [ ] **Convergence Analysis**: Early stopping and convergence monitoring
- [ ] **Decision Boundary Visualization**: 2D projections of hyperplane

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Kernel function implementations
- Optimization algorithm enhancements
- Visualization improvements
- Performance analysis tools
- Documentation enhancements

---

**📝 Note**: This implementation is designed for educational purposes to understand SVM fundamentals. For production applications, consider using optimized libraries like scikit-learn which implement advanced algorithms like SMO for superior performance.

**🎯 Key Takeaway**: SVMs excel at finding optimal decision boundaries by maximizing margins, making them particularly effective for high-dimensional data and cases where clear class separation exists.
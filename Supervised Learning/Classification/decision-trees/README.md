# Decision Tree Classification

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive machine learning project implementing the Decision Tree algorithm from scratch for breast cancer diagnosis, featuring custom entropy/Gini calculations, recursive tree building, and detailed performance analysis.

## � Project Overview

This project demonstrates the implementation of a Decision Tree classifier for binary classification of breast cancer tumors as either benign (B) or malignant (M). The implementation focuses on understanding the algorithm mechanics and provides educational value through:

- **Custom Decision Tree Implementation**: Built from scratch using only NumPy and basic Python
- **Multiple Splitting Criteria**: Implementation of both Entropy and Gini impurity measures
- **Recursive Tree Building**: Complete tree construction with stopping criteria
- **Visualization**: Tree structure visualization and data distribution plots
- **Performance Evaluation**: Detailed accuracy metrics and model evaluation
- **Algorithm Analysis**: Comparison of different impurity measures and hyperparameters

## � Dataset

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
1. **Data Cleaning**: Dropped irrelevant columns (`id`)
2. **Target Encoding**: Mapped `M`→1 (malignant), `B`→0 (benign)
3. **Train/Test Split**: Random 80/20 split with fixed seed for reproducibility

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib jupyter scikit-learn
```

### Running the Project

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Supervised Learning/Classification/decision-trees"
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook decision_trees.ipynb
   ```

3. **Run all cells** to see the complete analysis including:
   - Data exploration and preprocessing
   - Custom Decision Tree implementation
   - Model training and evaluation
   - Tree structure visualization
   - Comparison with scikit-learn

## � Implementation Details

### Custom Decision Tree Algorithm

The custom implementation includes a complete `DecisionTreeClassifier` class with the following methods:

#### **Core Components:**

1. **`__init__(self, min_sample_split, max_depth, criterion)`**: Initialize classifier with hyperparameters
2. **`fit(self, X, y)`**: Train the decision tree on training data
3. **`predict(self, X)`**: Make predictions on test samples
4. **`build_tree(self, dataset, curr_depth)`**: Recursively construct tree structure
5. **`get_best_split(self, dataset, num_samples, num_features)`**: Find optimal split criteria
6. **`entropy(self, y)`**: Calculate entropy impurity measure
7. **`gini(self, y)`**: Calculate Gini impurity measure
8. **`information_gain(self, parent, left_child, right_child, criterion)`**: Measure split quality
9. **`print_tree(self, tree, depth)`**: Visualize tree structure

#### **Algorithm Steps:**

1. **Training Phase**:
   - Start with entire training dataset at root
   - For each node, find best feature and threshold to split
   - Recursively build left and right subtrees
   - Stop when reaching minimum samples or maximum depth
   - Create leaf nodes with majority class

2. **Prediction Phase**:
   - For each test sample, traverse tree from root to leaf
   - At each node, compare feature value with threshold
   - Follow left (≤ threshold) or right (> threshold) branch
   - Return leaf node prediction

### Mathematical Foundation

**Entropy Formula:**
```
Entropy(S) = -Σ(pi × log₂(pi))
```

**Gini Impurity Formula:**
```
Gini(S) = 1 - Σ(pi²)
```

**Information Gain Formula:**
```
IG(S,A) = Entropy(S) - Σ(|Sv|/|S| × Entropy(Sv))
```

Where:
- `S` is the dataset
- `pi` is the proportion of samples belonging to class i
- `A` is the attribute/feature
- `Sv` are the subsets after splitting on attribute A

## 📈 Results

### Performance Metrics

Both custom and scikit-learn implementations achieve excellent accuracy on the breast cancer dataset:

| Implementation | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| **Custom Decision Tree** | 94.74% | 94.2% | 94.7% | 94.4% |
| **Scikit-learn DecisionTreeClassifier** | 92.98% | 92.5% | 93.0% | 92.7% |

### Performance Analysis

The Decision Tree implementation achieves excellent accuracy on the breast cancer dataset:

| Metric | Custom Implementation | Scikit-learn |
|--------|----------------------|--------------|
| **Best Accuracy** | 94.74% | 92.98% |
| **Optimal Depth** | 3 | 3 |
| **Training Time** | O(n²×d×log n) | O(n²×d×log n) |
| **Prediction Time** | O(log n) per sample | O(log n) per sample |

### Hyperparameter Analysis

**Effect of Tree Depth:**
- **Depth=1**: High bias, potential underfitting
- **Depth=3**: Optimal balance between bias and variance
- **Depth>5**: Increased variance, potential overfitting

**Effect of Minimum Sample Split:**
- **min_samples=2**: Risk of overfitting on small subsets
- **min_samples=3-5**: Good balance for this dataset size
- **min_samples>10**: May prevent useful splits

### Key Insights

- **Superior Performance**: Custom implementation achieves 94.74% vs scikit-learn's 92.98%
- **Entropy vs Gini**: Both criteria perform similarly on this dataset
- **Tree Structure**: Relatively shallow trees (depth=3) work well
- **Feature Importance**: Some features consistently chosen for early splits
- **Educational Value**: Custom implementation demonstrates algorithm mechanics while achieving better performance

## 📁 Project Structure

```
decision-trees/
├── decision_trees.ipynb       # Main implementation notebook
├── README.md                  # This documentation
└── Datasets/
    └── data.csv              # Breast cancer dataset
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and preprocessing
- **Matplotlib**: Data visualization and plotting
- **Scikit-learn**: Performance comparison and metrics
- **Jupyter Notebook**: Interactive development environment

## 🎓 Learning Objectives

This project teaches:

1. **Tree-Based Learning**: Understanding recursive tree construction algorithms
2. **Impurity Measures**: Implementation of entropy and Gini impurity
3. **Information Gain**: How to measure and use information gain for splits
4. **Hyperparameter Tuning**: Effect of depth and minimum samples on performance
5. **Overfitting Prevention**: Understanding stopping criteria and tree pruning
6. **Decision Boundaries**: How trees create piecewise-constant decision regions

## � Algorithm Characteristics

**Decision Tree Algorithm Properties:**

1. **Interpretable**: Easy to understand and visualize decision process
2. **Non-Parametric**: Makes no assumptions about data distribution
3. **Hierarchical**: Creates nested if-then-else conditions
4. **Greedy**: Makes locally optimal splits at each node
5. **Prone to Overfitting**: Can memorize training data without proper constraints

## 🧪 Experimental Setup

### Data Preprocessing
- **Missing Values**: None in the dataset
- **Feature Scaling**: Not required for decision trees
- **Train/Test Split**: 80/20 with random_state=42

### Hyperparameters
- **Maximum Depth**: Tested values (1, 3, 5, unlimited)
- **Minimum Sample Split**: Tested values (2, 3, 5, 10)
- **Splitting Criterion**: Entropy vs Gini impurity
- **Threshold Selection**: All unique feature values tested

## � Future Enhancements

- [ ] **Tree Pruning**: Pre-pruning and post-pruning techniques
- [ ] **Multi-class Support**: Extension to multi-class classification
- [ ] **Regression Trees**: Implementation for continuous target variables
- [ ] **Random Forest**: Ensemble method using multiple decision trees
- [ ] **Feature Importance**: Calculate and visualize feature importance scores
- [ ] **Cross-Validation**: More robust hyperparameter selection
- [ ] **Optimized Splitting**: Improved threshold selection strategies

## 📚 Algorithm Complexity

### Time Complexity
- **Training**: O(n² × d × log n) where n=samples, d=features
- **Prediction**: O(log n) average case, O(n) worst case
- **Space**: O(n) for storing the tree structure

### Space Complexity
- **Memory Usage**: O(n) for tree nodes in worst case
- **Scalability**: Efficient for moderate-sized datasets

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Tree pruning algorithms implementation
- Multi-class classification support
- Performance optimizations
- Advanced visualization techniques
- Cross-validation framework

## 📚 References

This implementation was developed following the educational tutorial:
- **YouTube Tutorial**: [Decision Trees in Machine Learning](https://www.youtube.com/watch?v=sgQAhG5Q7iY&t=18s) - Comprehensive guide to building decision trees from scratch

**📝 Note**: This implementation is for educational purposes to understand the mechanics of decision tree classification. For production use with large datasets, consider optimized libraries like scikit-learn with advanced pruning and optimization techniques.
# Random Forest Classification

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive machine learning project implementing the Random Forest algorithm from scratch for breast cancer diagnosis, featuring custom ensemble methods, bootstrap sampling, majority voting, and detailed performance analysis comparing with scikit-learn.

## 🎯 Project Overview

This project demonstrates the implementation of a Random Forest classifier for binary classification of breast cancer tumors as either benign (B) or malignant (M). The implementation focuses on understanding ensemble learning mechanics and provides educational value through:

- **Custom Random Forest Implementation**: Built from scratch using only NumPy and basic Python
- **Ensemble Learning**: Implementation of bagging with bootstrap sampling
- **Feature Randomness**: Random feature selection at each split for diversity
- **Majority Voting**: Democratic decision-making across multiple trees
- **Bootstrap Sampling**: Creating diverse training sets for each tree
- **Performance Evaluation**: Detailed accuracy metrics and comparison with scikit-learn
- **Educational Value**: Step-by-step algorithm explanation and visualization

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
   cd "Supervised Learning/Classification/random-forest"
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook random-forest.ipynb
   ```

3. **Run all cells** to see the complete analysis including:
   - Data exploration and preprocessing
   - Custom Random Forest implementation
   - Model training and evaluation
   - Bootstrap sampling demonstration
   - Comparison with scikit-learn

## ⚙️ Implementation Details

### Custom Random Forest Algorithm

The custom implementation includes a complete `RandomForest` class with the following methods:

#### **Core Components:**

1. **`RandomForest` Class:**
   - **`__init__(self, n_estimators, max_depth, min_sample_split, max_features, criterion, random_state)`**: Initialize ensemble with hyperparameters
   - **`fit(self, X, y)`**: Train multiple decision trees on bootstrap samples
   - **`predict(self, X)`**: Make predictions using majority voting
   - **`predict_proba(self, X)`**: Get class probabilities from vote proportions

2. **`DecisionTreeForRF` Class:**
   - **`fit(self, X, y)`**: Train individual tree with feature randomness
   - **`predict(self, X)`**: Make predictions from single tree
   - **`get_best_split(self, dataset, num_samples, num_features)`**: Find optimal split using random features
   - **`build_tree(self, dataset, curr_depth)`**: Recursively construct tree structure

3. **`bootstrap_sample(X, y)` Function:**
   - Creates random samples with replacement for tree diversity

#### **Algorithm Steps:**

1. **Training Phase**:
   - For each tree in the forest:
     - Create bootstrap sample (random sampling with replacement)
     - Train decision tree on bootstrap sample with random feature selection
     - Store trained tree in the ensemble
   - Repeat for n_estimators trees

2. **Prediction Phase**:
   - For each test sample:
     - Get prediction from every tree in the forest
     - Use majority voting to determine final prediction
     - Calculate class probabilities based on vote proportions

### Mathematical Foundation

**Bootstrap Sampling:**
```
Bootstrap Sample = Random selection of n samples WITH replacement
Out-of-Bag (OOB) ≈ 37% of original samples not selected
```

**Feature Randomness:**
```
Features per split = √(total_features) for classification
Randomly select subset at each node split
```

**Majority Voting:**
```
Final Prediction = argmax(Σ tree_predictions)
Class Probability = (votes for class) / (total trees)
```

**Ensemble Variance Reduction:**
```
Var(ensemble) = Var(individual) / n_trees (for independent trees)
```

## 📈 Results

### Performance Metrics

Both custom and scikit-learn implementations achieve excellent accuracy on the breast cancer dataset:

| Implementation | Accuracy | Precision (Avg) | Recall (Avg) | F1-Score (Avg) |
|----------------|----------|-----------------|--------------|----------------|
| **Custom Random Forest** | 96.49% | 97.0% | 96.0% | 96.0% |
| **Scikit-learn RandomForestClassifier** | 96.49% | 97.0% | 96.0% | 96.0% |

### Performance Analysis

The Random Forest implementation achieves excellent accuracy on the breast cancer dataset:

| Metric | Custom Implementation | Scikit-learn |
|--------|----------------------|--------------|
| **Best Accuracy** | 96.49% | 96.49% |
| **Number of Trees** | 100 | 100 |
| **Max Depth** | 10 | 10 |
| **Training Time** | O(n_trees × n_samples × log n) | O(n_trees × n_samples × log n) |
| **Prediction Time** | O(n_trees × log n) per sample | O(n_trees × log n) per sample |

### Hyperparameter Analysis

**Effect of Number of Trees (n_estimators):**
- **10 trees**: Good baseline, potential for improvement
- **50 trees**: Significant improvement, diminishing returns begin
- **100 trees**: Optimal balance between performance and computation
- **500+ trees**: Marginal gains, increased computation time

**Effect of Max Features:**
- **√(n_features)**: Default choice, good balance of diversity and performance
- **log₂(n_features)**: More diversity, potentially lower individual tree performance
- **All features**: Less diversity, risk of overfitting

**Effect of Bootstrap Sampling:**
- **With replacement**: Creates diversity, some samples appear multiple times
- **Out-of-bag samples**: ~37% of data not used per tree, natural validation set
- **Sample size**: Same as original dataset maintains statistical properties

### Key Insights

- **Performance Comparison**: Both implementations achieve identical 96.49% accuracy
- **Ensemble Effect**: Multiple trees significantly outperform single decision tree
- **Feature Randomness**: Random feature selection creates meaningful diversity
- **Bootstrap Sampling**: Sampling with replacement prevents overfitting
- **Majority Voting**: Democratic approach provides robust predictions
- **Educational Value**: Custom implementation demonstrates ensemble mechanics while achieving identical performance to scikit-learn

## 📁 Project Structure

```
random-forest/
├── random-forest.ipynb       # Main implementation notebook
├── README.md                 # This documentation
└── Datasets/
    └── data.csv             # Breast cancer dataset
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and preprocessing
- **Matplotlib**: Data visualization and plotting
- **Scikit-learn**: Performance comparison and metrics
- **Jupyter Notebook**: Interactive development environment
- **Collections**: Counter for majority voting implementation

## 🎓 Learning Objectives

This project teaches:

1. **Ensemble Learning**: Understanding how multiple models improve performance
2. **Bootstrap Sampling**: Implementation of bagging for model diversity
3. **Feature Randomness**: How random feature selection creates tree diversity
4. **Majority Voting**: Democratic decision-making in ensemble methods
5. **Bias-Variance Tradeoff**: How ensembles reduce variance while maintaining low bias
6. **Out-of-Bag Evaluation**: Using unsampled data for validation

## ⚡ Algorithm Characteristics

**Random Forest Algorithm Properties:**

1. **Robust**: Less prone to overfitting than individual decision trees
2. **Parallel**: Trees can be trained independently (parallelizable)
3. **Feature Importance**: Can calculate importance scores for features
4. **Out-of-Bag Error**: Built-in validation without separate test set
5. **Handles Missing Values**: Can work with incomplete data
6. **Non-Parametric**: Makes no assumptions about data distribution

## 🧪 Experimental Setup

### Data Preprocessing
- **Missing Values**: None in the dataset
- **Feature Scaling**: Not required for tree-based methods
- **Train/Test Split**: 80/20 with random_state=42

### Hyperparameters
- **Number of Trees (n_estimators)**: 100
- **Maximum Depth**: 10 (prevents overfitting)
- **Minimum Samples Split**: 2
- **Max Features**: √(30) ≈ 5 features per split
- **Splitting Criterion**: Entropy for information gain
- **Bootstrap**: True (sampling with replacement)

### Bootstrap Sampling Analysis
- **Sample Size**: Same as original dataset (455 samples)
- **Replacement**: True (allows duplicate samples)
- **Out-of-Bag**: ~37% samples not selected per tree
- **Diversity**: Each tree sees different data combinations

## 🔮 Future Enhancements

- [ ] **Feature Importance**: Calculate and visualize feature importance scores
- [ ] **Out-of-Bag Error**: Implement OOB error estimation
- [ ] **Parallel Training**: Multi-threaded tree training
- [ ] **Random Subspaces**: Additional randomness through feature subsampling
- [ ] **Regression Support**: Extension to Random Forest Regressor
- [ ] **Cross-Validation**: More robust hyperparameter selection
- [ ] **Tree Visualization**: Visualize individual trees in the forest
- [ ] **Memory Optimization**: Efficient storage for large forests

## 📚 Algorithm Complexity

### Time Complexity
- **Training**: O(n_trees × n_samples × n_features × log n_samples)
- **Prediction**: O(n_trees × log n_samples) per sample
- **Bootstrap Sampling**: O(n_samples) per tree
- **Space**: O(n_trees × tree_size) for storing the forest

### Space Complexity
- **Memory Usage**: O(n_trees × average_tree_size)
- **Scalability**: Linear scaling with number of trees
- **Storage**: Each tree stored independently

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Out-of-bag error implementation
- Feature importance calculation
- Parallel training optimization
- Memory efficiency improvements
- Advanced ensemble techniques

## 📚 References

This implementation was developed following ensemble learning principles and Random Forest methodology:

- **Breiman, L. (2001)**: "Random Forests" - Original Random Forest paper
- **Ensemble Learning Theory**: Bootstrap aggregating (bagging) methodology
- **Decision Tree Foundation**: Built upon decision tree classification principles

**Educational Resources:**
- Random Forest algorithm explanation and implementation
- Bootstrap sampling and ensemble learning concepts
- Majority voting and prediction aggregation techniques

**📝 Note**: This implementation is for educational purposes to understand the mechanics of Random Forest classification and ensemble learning. For production use with large datasets, consider optimized libraries like scikit-learn with advanced optimizations, parallel processing, and memory management techniques.

## 🏆 Key Achievements

- **High Accuracy**: 96.49% accuracy on breast cancer classification
- **Ensemble Mastery**: Successfully implemented bootstrap sampling and majority voting
- **Educational Value**: Clear demonstration of Random Forest principles
- **Identical Performance**: Matches scikit-learn's optimized implementation exactly
- **Complete Pipeline**: End-to-end implementation from data loading to evaluation
- **Robust Design**: Handles various hyperparameters and edge cases

This Random Forest implementation demonstrates the power of ensemble learning and provides a solid foundation for understanding more advanced ensemble techniques!
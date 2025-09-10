# Naive Bayes Classification Model

A machine learning project implementing Gaussian Naive Bayes classifier for breast cancer diagnosis using both custom implementation and scikit-learn comparison.

## 📋 Project Overview

This project demonstrates the implementation of a Naive Bayes classifier for binary classification of breast cancer tumors as either benign (B) or malignant (M). The project includes:

- **Custom Naive Bayes Implementation**: Built from scratch using NumPy
- **Scikit-learn Comparison**: Using GaussianNB for performance comparison
- **Data Visualization**: Scatter plots showing data distribution
- **Performance Evaluation**: Accuracy metrics for both implementations

## 📊 Dataset

The dataset contains breast cancer diagnostic features with the following characteristics:

- **Total Samples**: 569 cases
- **Features**: 30 numerical features (mean, standard error, and worst values for 10 real-valued features)
- **Target Classes**: +
  - `B` - Benign (non-cancerous)
  - `M` - Malignant (cancerous)
- **Train/Test Split**: 80% training, 20% testing

### Key Features Include:
- Radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Concave points, symmetry, fractal dimension

3. Run all cells to see the complete analysis

## 🔧 Implementation Details

### Custom Naive Bayes Algorithm

The custom implementation follows these key steps:

1. **Training Phase**:
   - Calculate prior probabilities for each class
   - Compute mean and standard deviation for each feature per class
   
2. **Prediction Phase**:
   - Apply Gaussian probability density function
   - Use Bayes' theorem to calculate posterior probabilities
   - Predict class with highest posterior probability

### Mathematical Foundation

The classifier uses Bayes' theorem:

```
P(class|features) = P(features|class) × P(class) / P(features)
```

Where:
- `P(class|features)` is the posterior probability
- `P(features|class)` is the likelihood (Gaussian PDF)
- `P(class)` is the prior probability

## 📈 Results

Both implementations achieve comparable accuracy on the breast cancer dataset:

- **Custom Implementation**: 96.4912% accuracy
- **Scikit-learn GaussianNB**: 97.3684% accuracy

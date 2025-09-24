# K-Means Clustering

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive machine learning project implementing the K-Means clustering algorithm from scratch for FIFA 24 player segmentation, featuring custom centroid initialization, iterative optimization, and real-time visualization with performance comparison to scikit-learn.

## 🎯 Project Overview

This project demonstrates K-Means clustering implementation for unsupervised learning on FIFA 24 players dataset to segment players based on performance statistics. Key features include:

- **Custom K-Means Implementation**: Built from scratch using only NumPy and basic Python
- **Lloyd's Algorithm**: Complete implementation with convergence detection
- **Real-time Visualization**: PCA-based 2D clustering visualization during training
- **Performance Comparison**: Benchmarking against scikit-learn implementation
- **Player Segmentation**: Meaningful clustering of players by skill profiles

## 📊 Dataset

**FIFA 24 Male Players Dataset:**
- **Total Players**: ~17,000+ professional football players
- **Features**: 7 performance metrics (age, overall, PAC, SHO, DRI, DEF, PHY)
- **Preprocessing**: Z-score normalization and missing value removal

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn jupyter scikit-learn
```

### Running the Project

1. **Navigate to the project directory:**
   ```bash
   cd "Unsupervised Learning/k-means-clustering"
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook k-means-clustering.ipynb
   ```

3. **Run all cells** to see the complete analysis and real-time clustering visualization.

## ⚙️ Implementation Details

### Custom K-Means Algorithm

The implementation includes a complete `KMeansClustering` class with core methods:

- **`initialize_centroids()`**: Random centroid initialization within data bounds
- **`calculate_distances()`**: Euclidean distance calculation from points to centroids
- **`assign_labels()`**: Assign each point to nearest centroid
- **`update_centroids()`**: Update centroids as cluster means
- **`fit()`**: Main training method with real-time PCA visualization

### Mathematical Foundation

**Objective Function:**
```
J = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
```

**Algorithm Steps:**
1. Initialize k centroids randomly
2. Assign points to nearest centroids
3. Update centroids as cluster means
4. Repeat until convergence (centroid movement < 1e-4)

## 📈 Results

**Performance Comparison:**

| Implementation | Convergence | Final Inertia | Player Clusters |
|----------------|-------------|---------------|-----------------|
| **Custom K-Means** | ~20 iterations | Optimized | 4 meaningful groups |
| **Scikit-learn** | Similar | Matching | Identical results |

**Player Segmentation:**
- **Elite Attackers**: High shooting and pace
- **Defensive Specialists**: Strong defensive ratings
- **Young Prospects**: Developing skills
- **Balanced Veterans**: Well-rounded attributes

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Scikit-learn**: Comparison and PCA
- **Jupyter Notebook**: Interactive development

## 🎓 Learning Objectives

- Understanding unsupervised learning concepts
- Lloyd's algorithm implementation
- Centroid optimization and convergence
- Real-time visualization techniques
- Player segmentation analysis

## 🏆 Key Achievements

- **Complete Implementation**: K-Means algorithm built from scratch
- **Real-time Visualization**: PCA-based clustering animation
- **Performance Parity**: Results match scikit-learn exactly
- **Practical Application**: Meaningful FIFA player segmentation
- **Educational Value**: Clear demonstration of clustering mechanics
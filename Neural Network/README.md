# Neural Network from Scratch

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![CuPy](https://img.shields.io/badge/CuPy-GPU-green.svg)](https://cupy.dev/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive implementation of a feedforward neural network built entirely from scratch using NumPy/CuPy, featuring custom backpropagation, multiple activation functions, GPU acceleration, and achieving **97.69% accuracy** on MNIST digit classification.

## Project Overview

This project demonstrates a complete neural network implementation for multi-class classification, built without using high-level ML frameworks like TensorFlow or PyTorch. The implementation focuses on understanding the mathematical foundations and provides:

- **Full Neural Network Architecture**: Custom implementation with multiple hidden layers
- **Backpropagation Algorithm**: Complete gradient computation and weight updates
- **Multiple Activation Functions**: Support for ReLU, Sigmoid, and derivatives
- **GPU Acceleration**: Optional CuPy support for faster training
- **Mini-Batch Training**: Efficient stochastic gradient descent with data shuffling
- **Flexible Architecture**: Configurable layer sizes and activation methods
- **Multiple Datasets**: Tested on Iris (3-class) and MNIST (10-class) datasets
- **Smart Initialization**: He initialization for ReLU, Xavier for Sigmoid

## Datasets

### 1. Iris Dataset
- **Total Samples**: 150 flowers
- **Features**: 4 numerical features (sepal length, sepal width, petal length, petal width)
- **Target Classes**: 3 species (Setosa, Versicolor, Virginica)
- **Train/Test Split**: 80% training (120), 20% testing (30)
- **Best Accuracy**: ~96-98%

### 2. MNIST Dataset (Primary)
- **Total Samples**: 70,000 handwritten digit images
- **Features**: 784 pixels (28×28 grayscale images, normalized to 0-1)
- **Target Classes**: 10 digits (0-9)
- **Train/Test Split**: 80% training (56,000), 20% testing (14,000)
- **Best Accuracy**: **97.69%**

## Quick Start

### Prerequisites
```bash
# Basic requirements
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Optional: GPU acceleration (requires CUDA-compatible GPU)
pip install cupy-cuda11x  # For CUDA 11.x
# OR
pip install cupy-cuda12x  # For CUDA 12.x
```

### Running the Project

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Neural Network"
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook neural-network.ipynb
   ```

3. **Run the cells sequentially** to:
   - Define the `NeuralNetwork` class
   - Load and preprocess data
   - Train the model
   - Evaluate performance
   - Visualize results

## Architecture

### Network Structure
```
Input Layer (784)  →  Hidden Layer 1 (256)  →  Hidden Layer 2 (128)  →  Hidden Layer 3 (64)  →  Output Layer (10)
```

### Key Components

| Component | Implementation |
|-----------|---------------|
| **Initialization** | He initialization for ReLU, Xavier for Sigmoid |
| **Forward Pass** | Matrix multiplication with activation functions |
| **Activation Functions** | ReLU (hidden layers), Sigmoid (output layer) |
| **Loss Function** | Mean Squared Error (MSE) |
| **Optimizer** | Mini-batch Stochastic Gradient Descent (SGD) |
| **Backpropagation** | Custom gradient computation using chain rule |
| **Data Handling** | Shuffling and batching for stable training |

## Training Results

### Experiment Comparison

| Experiment | Architecture | LR | Epochs | Batch Size | Accuracy |
|------------|--------------|----|----|------------|----------|
| **Baseline** | 784→128→64→10 | 0.005 | 100 | 128 | 96.79% |
| **Optimized LR** | 784→128→64→10 | 0.01 | 150 | 128 | 97.57% |
| **Deep Network** | 784→256→128→64→10 | 0.01 | 150 | 128 | **97.69%** |

### Performance Highlights
- ✅ **Best Accuracy**: 97.69% on MNIST test set
- ✅ **Training Time**: ~2-3 minutes on GPU (CuPy)
- ✅ **Convergence**: Stable loss decrease with mini-batch training
- ✅ **Per-Digit Accuracy**: 96-99% across all digits (0-9)

### Key Findings
1. **Learning Rate Impact**: Increasing LR from 0.005 to 0.01 improved accuracy by 0.79%
2. **Network Depth**: Adding layers (256→128→64) improved accuracy by 0.12%
3. **Activation Functions**: ReLU significantly outperforms Sigmoid for deep networks
4. **Batch Size**: 128 provides good balance between speed and stability
5. **GPU Acceleration**: CuPy provides 3-5x speedup over NumPy

## Implementation Details

### Class Structure
```python
class NeuralNetwork:
    def __init__(layer_sizes, use_gpu, activation_method)
    def sigmoid(x) / relu(x)
    def sigmoid_derivative(x) / relu_derivative(x)
    def forward_with_activation(X, activation_method)
    def backward_with_activation(activations, delta, lr, activation_method)
    def train(X, y, epochs, lr, batch_size, activation_method)
    def predict(X, activation_method)
```

### Training Process
```python
# Create network
nn = NeuralNetwork(layer_sizes=[784, 256, 128, 64, 10], use_gpu=True, activation_method='relu')

# Train with mini-batch SGD
losses = nn.train(
    X_train, y_train,
    epochs=150,
    batch_size=128,
    lr=0.01,
    activation_method='relu'
)

# Evaluate
accuracy = evaluate(nn, X_test, y_test)
```

## Visualizations

The notebook includes:
- **Training Loss Curves**: Monitor convergence over epochs
- **Confusion Matrix**: Visualize classification errors
- **Per-Digit Accuracy**: Breakdown of performance per class
- **Sample Predictions**: Visual inspection of correct/incorrect predictions
- **Architecture Diagrams**: Network structure visualization

## Advanced Features

### GPU Acceleration
```python
# Automatically uses CuPy if available
nn = NeuralNetwork(layer_sizes=[784, 128, 10], use_gpu=True, activation_method='relu')
```

### Flexible Activation Functions
```python
# Use ReLU (recommended for deep networks)
nn.train(X, y, activation_method='relu')

# Use Sigmoid (better for shallow networks)
nn.train(X, y, activation_method='sigmoid')
```

### Mini-Batch Training
```python
# Full batch
nn.train(X, y, batch_size=None)

# Mini-batch (faster, more stable)
nn.train(X, y, batch_size=128)
```

## Mathematical Foundation

### Forward Propagation
```
z^(l) = a^(l-1) · W^(l) + b^(l)
a^(l) = σ(z^(l))
```

### Backpropagation
```
δ^(L) = a^(L) - y
δ^(l) = (δ^(l+1) · (W^(l+1))^T) ⊙ σ'(a^(l))
∂L/∂W^(l) = (a^(l-1))^T · δ^(l) / m
W^(l) = W^(l) - α · ∂L/∂W^(l)
```

### Activation Functions
```
ReLU: f(x) = max(0, x)
ReLU': f'(x) = 1 if x > 0, else 0

Sigmoid: f(x) = 1 / (1 + e^(-x))
Sigmoid': f'(x) = f(x) · (1 - f(x))
```

## Performance Tips

### To Improve Accuracy Further:
1. **Increase Network Capacity**: Use `[784, 512, 256, 128, 10]`
2. **Tune Learning Rate**: Try values between 0.005-0.02
3. **Adjust Batch Size**: Experiment with 64, 128, or 256
4. **Train Longer**: Use 200+ epochs for better convergence
5. **Use ReLU**: Always use ReLU for deep networks (avoids vanishing gradients)

### Features to Add for 98%+ Accuracy:
- **Momentum/Adam Optimizer**: Faster convergence
- **Learning Rate Decay**: Gradually reduce LR during training
- **Dropout**: Prevent overfitting
- **Batch Normalization**: Stabilize training
- **Cross-Entropy Loss**: Better for classification than MSE

### Common Issues:
- **Loss becomes NaN**: Learning rate too high → reduce by 10x (try 0.001)
- **Accuracy stuck at ~10%**: Network not learning → check initialization or LR
- **Slow training**: Enable GPU with `use_gpu=True` and install CuPy
- **Low accuracy with sigmoid**: Use ReLU for hidden layers instead

## Project Structure

```
Neural Network/
├── neural-network.ipynb    # Main notebook with implementation
├── README.md              # This file
└── .ipynb_checkpoints/    # Auto-generated (ignored by git)
```

## Experiments to Try

1. **Different Architectures**: Compare shallow `[784, 512, 10]` vs deep `[784, 128, 128, 128, 10]`
2. **Activation Functions**: Compare ReLU vs Sigmoid on same architecture
3. **Batch Sizes**: Test 32, 64, 128, 256 to find optimal speed/accuracy tradeoff
4. **Learning Rates**: Grid search over [0.001, 0.005, 0.01, 0.02, 0.05]
5. **Advanced Optimizers**: Implement momentum, Adam, or RMSprop
6. **Regularization**: Add L2 penalty or dropout layers
7. **Loss Functions**: Implement Cross-Entropy loss for better classification
8. **Other Datasets**: Try Fashion-MNIST, CIFAR-10 (requires CNN)

## Learning Outcomes

By studying this implementation, you will understand:
- ✅ How neural networks learn through backpropagation
- ✅ The role of activation functions in non-linear learning
- ✅ Why gradient descent converges to optimal weights
- ✅ How matrix operations enable efficient computation
- ✅ The impact of hyperparameters on model performance
- ✅ GPU acceleration principles in deep learning

## Future Enhancements

**Potential Features to Add:**
- ⏳ **Momentum Optimizer**: Accelerate convergence with velocity-based updates
- ⏳ **Adam Optimizer**: Adaptive learning rates per parameter
- ⏳ **Learning Rate Decay**: Gradually reduce learning rate during training
- ⏳ **Dropout Regularization**: Randomly deactivate neurons to prevent overfitting
- ⏳ **Batch Normalization**: Normalize layer inputs for stable training
- ⏳ **Cross-Entropy Loss**: Better gradient flow for classification
- ⏳ **L2 Regularization**: Weight penalty to reduce overfitting
- ⏳ **Early Stopping**: Stop training when validation loss stops improving

**Contributions Welcome:**
Feel free to fork and add these features to improve the implementation!

## Notes

- **Pure NumPy Implementation**: No TensorFlow, PyTorch, or Keras dependencies
- **Educational Focus**: Code prioritizes clarity over optimization
- **GPU Optional**: Works perfectly on CPU, faster with CuPy/GPU
- **Reproducible**: Fixed random seeds for consistent results

## Achievements

- **97.69% accuracy** on MNIST (top 2% behind state-of-the-art CNNs at ~99.7%)
- **GPU-accelerated** training with CuPy
- **Multiple datasets** supported (Iris, MNIST)
- **Production-ready** code with proper error handling

## References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Deep Learning Book](https://www.deeplearningbook.org/) by Ian Goodfellow
- [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) by 3Blue1Brown

## License

This project is part of the Machine-Learning-From-Scratch repository.

---

**Built with love for learning and understanding neural networks from first principles**

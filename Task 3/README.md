# Deep Learning Mechanics and Architecture

## Overview

This project implements and compares:

- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)

Both models are trained for multi-class classification using:

- Softmax activation function
- Cross-Entropy Loss
- Gradient Descent optimization

# Loss Function

## Softmax Activation

The output layer uses the Softmax function to convert logits into probabilities.

For logits:

z_i = w_i^T x + b_i

Softmax is defined as:

ŷ_i = exp(z_i) / Σ exp(z_j)

Where:
- K = number of classes
- ŷ_i = predicted probability for class i

Softmax ensures:

Σ ŷ_i = 1

Thus, outputs can be interpreted as probabilities.

---

## Cross-Entropy Loss

For multi-class classification, we use Categorical Cross-Entropy:

L = - Σ y_i log(ŷ_i)

Where:
- y_i = true label (this is one-hot encoded)
- ŷ_i = predicted probability

Since only one y_i = 1, this simplifies to:

L = - log(ŷ_true)

Interpretation:
- If predicted probability is high → loss is small
- If predicted probability is low → loss is large

Cross-Entropy strongly penalizes incorrect high-confidence predictions.

---

# Gradient Descent Update Rule

We use (Stochastic) Gradient Descent for optimization.

General update rule:

w = w - η ∂L/∂w  
b = b - η ∂L/∂b  

Where:
- η = learning rate
- ∂L/∂w = gradient of loss with respect to weights

With Softmax + Cross-Entropy, the gradient simplifies to:

∂L/∂z = ŷ - y

Thus weight update becomes:

w = w - η (ŷ - y)x

This simplification makes training computationally efficient and numerically stable.

---


# ANN vs CNN Comparison

| Model | Architecture | Parameters | Test Accuracy | Observations |
|--------|-------------|------------|---------------|--------------|
| ANN | Fully Connected Layers | (Add value) | (Add %) | Treats pixels independently |
| CNN | Convolution + Pooling Layers | (Add value) | (Add %) | Learns spatial features |

---

## Key Differences

### ANN
- Flattens image input
- Large number of parameters
- Does not preserve spatial relationships
- Typically lower accuracy for image tasks

### CNN
- Uses convolutional filters
- Preserves spatial structure
- Weight sharing reduces parameters
- Better performance on image datasets

---

#  Conclusion

- Softmax converts logits into class probabilities.
- Cross-Entropy measures prediction error.
- Gradient Descent updates parameters to minimize loss.
- CNN generally outperforms ANN in image classification tasks due to spatial feature extraction.
- More in-depth explainations have been provided in the markdown sections of the .ipynb file which is provided. 
---

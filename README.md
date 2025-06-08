# 🧠 Neural Network Digit Recognizer (NumPy Only)

This project is a handcrafted neural network for digit recognition, implemented entirely with **NumPy**, without any high-level machine learning libraries like TensorFlow or PyTorch.

---

## 🔍 About the Project

The goal of this project was to deeply understand the core mechanics behind neural networks by manually building and training a model to classify digits from the MNIST dataset.

All core components—including initialization, activation, forward and backward propagation, and optimization—were implemented from scratch using low-level NumPy operations.

---

## 📌 Key Features

- **Weight Initialization**: Xavier and He methods  
- **Activation Function**: Leaky ReLU (used instead of ReLU to prevent dying neuron issues)  
- **Backpropagation**: Fully vectorized using multivariable calculus  
- **Optimization Techniques**:  
  - L2 Regularization  
  - Learning Rate Decay  
  - Early Stopping based on validation cost variance  
- **Loss Tracking**: Training loss, validation loss, variance, and test accuracy plotted after training

---

## 📊 Performance

- **Dataset**: MNIST (loaded via `torchvision.datasets`)
- **Current Accuracy**: ~84% on test set
- **Limitation**: Currently investigating a memory efficiency issue, likely related to buffer reallocation during training

---

## 📁 Repository Contents

- `digit_recognizer.py`: Full implementation of the `NN` class and training/testing loops  
- `README.md`: Project overview (this file)  
- No external dependencies beyond NumPy, Matplotlib, and Torchvision (used only for dataset loading)

---

## ▶️ How to Run

```bash
git clone https://github.com/aprotsenko24/digit-recognizer-numpy.git
cd digit-recognizer-numpy
pip install numpy matplotlib torchvision
python digit_recognizer.py

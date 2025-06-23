# 🧠 Neural Network Digit Recognizer (NumPy Only)

This project implements a neural network for digit classification using the MNIST dataset. The entire training pipeline — from architecture and forward propagation to backpropagation and optimization — is developed from scratch using only the NumPy library.

---

## 📌 Overview

- **Dataset:** MNIST (loaded via `torchvision.datasets`)
- **Implementation:** Python, NumPy (no high-level ML frameworks)
- **Accuracy:** ~82–91% on 75 unseen test samples
- **Architecture:** 784–20–10 fully connected layers

---

## ⚙️ Features

- **Manual Feedforward & Backpropagation**
  - Custom matrix-based forward propagation
  - Hand-coded backward pass using gradient descent with Adam optimizer

- **Optimizations**
  - **L2 regularization**
  - **Exponential learning rate decay**
  - **Early stopping** based on validation cost variance
  - **Weight & bias checkpointing** for best-performing epoch

- **Initialization**
  - **Xavier initialization** for hidden layers
  - **He initialization** for output layer

- **Activation & Output**
  - **Leaky ReLU** for hidden layers
  - **Softmax** for final classification

- **Visualization**
  - Cost curves for training, validation, and test sets
  - Tracking of variance to identify overfitting
  - Printout of final accuracies per dataset

---

## 📊 Performance

- Training Accuracy: Varies across epochs; up to ~90%
- Validation Accuracy: Tracked to prevent overfitting
- Test Accuracy: ~82–91% on 75 random MNIST digits

---

## 📁 Project Structure

digit-recognizer-numpy/
├── digit_recognizer.py # Full training + testing pipeline with hardcoded neural network
├── README.md # Project documentation
└── /data # Automatically created folder by torchvision for MNIST dataset

---

## ▶️ How to Run

```bash
git clone https://github.com/aprotsenko24/digit-recognizer-numpy.git
cd digit-recognizer-numpy
pip install numpy matplotlib torchvision
python digit_recognizer.py

---

Let me know if you'd like to include example plots (like loss curves) or add a section on limitations and future improvements!

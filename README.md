# 🧠 Neural Network Digit Recognizer (NumPy Only)

This project implements a digit recognizer neural network entirely from scratch using **NumPy**. No high-level machine learning frameworks like TensorFlow or PyTorch are used—only pure NumPy and low-level matrix operations.

---

## 🔍 About the Project

This project was built to gain a deep, hands-on understanding of the internal workings of neural networks. Every core feature—including forward and backward propagation, softmax classification, gradient descent, regularization, and initialization—is manually implemented.

---

## 📌 Key Features

- **Weight Initialization**:
  - **He Initialization** (for output layer)
  - **Xavier Initialization** (for hidden layers)
- **Activation**:
  - **Leaky ReLU** to avoid dead neurons
- **Output**:
  - **Softmax** layer for classification
- **Loss Function**:
  - Cross-entropy with **L2 Regularization**
- **Optimization Techniques**:
  - Learning Rate **Decay** (Exponential)
  - Manual **Gradient Descent**
- **Training Controls**:
  - Modular feedforward/backpropagation logic
  - Custom cost tracking and accuracy evaluation

---

## 📊 Performance

- **Dataset**: MNIST (loaded via `torchvision.datasets`)
- **Training Accuracy**: ~88–90%  
- **Regularization Impact**: L2 Regularization penalizes weight overgrowth, improving generalization
- **Test Performance**: Final evaluation pending

---

## 📁 Project Structure

- `digit_recognizer.py` – Entire training pipeline including:
  - CNN model construction
  - Feedforward/backpropagation logic
  - Cost computation
  - Visualization of cost decay
- `README.md` – Project overview and setup instructions

---

## ▶️ How to Run

```bash
git clone https://github.com/aprotsenko24/digit-recognizer-numpy.git
cd digit-recognizer-numpy
pip install numpy matplotlib torchvision
python digit_recognizer.py

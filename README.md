# 🚀 Logistic Regression from Scratch (NumPy)

<p align="center">
  <img src="https://media.giphy.com/media/3o7TKtnuHOHHUjR38Y/giphy.gif" width="250"/>
</p>

<p align="center">
  <b>Pure NumPy implementation of Logistic Regression</b><br>
  Built to deeply understand ML math, optimization, and learning mechanics
</p>

---


<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![ML](https://img.shields.io/badge/Machine%20Learning-From%20Scratch-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</p>

---

## 📌 Project Preview

| Sigmoid Curve | Training Loss |
|--------------|--------------|
| ![Sigmoid Curve](Sigmoid%20Curve.png) | ![Training Loss](Training%20Loss.png) |

---

## 🧠 Project Goal

This project implements **Logistic Regression from scratch** to understand:

- How models learn from data
- How gradient descent optimizes parameters
- How probabilities are computed
- How decision boundaries are formed

---

## 🏗️ Architecture Overview

### 🔹 Model Pipeline

```

Input Features (X)
↓
Linear Transformation (Xw + b)
↓
Sigmoid Activation
↓
Predicted Probability (ŷ)
↓
Cross-Entropy Loss
↓
Gradient Descent Updates
↓
Optimized Weights (w, b)

````

---

## 🧠 Mathematical Flow

### 🔹 Linear Model

$$
z = Xw + b
$$

---

### 🔹 Sigmoid Function

$$
\hat{y} = \frac{1}{1 + e^{-z}}
$$

---

### 🔹 Loss Function (Cross Entropy)

$$
L = -\frac{1}{m} \sum \left[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})\right]
$$

---

### 🔹 Gradient Descent Updates

$$
w = w - \alpha \frac{1}{m} X^T(\hat{y} - y)
$$

$$
b = b - \alpha \frac{1}{m} \sum (\hat{y} - y)
$$

---

## 📁 Project Structure

```bash
Logistic-Regression-From-Scratch/
├── main.py
├── README.md
├── Sigmoid Curve.png
└── Training Loss.png
````

---

## 🚀 Features

* Logistic Regression from scratch
* Multi-feature input support
* Sigmoid activation function
* Cross-entropy loss
* Gradient descent optimization
* Probability predictions
* Class predictions (0/1)
* Training loss visualization
* Decision boundary visualization

---

## 📊 Dataset Example

```python
X = [
    [2, 7],
    [3, 6],
    [4, 6],
    [5, 5],
    [6, 4],
    [7, 3]
]

y = [0, 0, 0, 1, 1, 1]
```

---

## ▶️ Installation

```bash
git clone https://github.com/yourusername/logistic-regression-from-scratch.git
cd logistic-regression-from-scratch
pip install numpy matplotlib
```

---

## ▶️ Run Project

```bash
python main.py
```

---

## ▶️ Usage Example

```python
from main import LogisticRegression

model = LogisticRegression(learning_rate=0.1, epochs=2000)

model.fit(X, y)

print(model.predict([[4, 6]]))
print(model.predict_proba([[4, 6]]))
```

---

## 📈 Output Example

```
Epoch 0   | Loss: 0.6931
Epoch 100 | Loss: 0.4212
Epoch 200 | Loss: 0.3124
Epoch 300 | Loss: 0.2451
```

---

## 📊 Visualizations

### 🔹 Sigmoid Decision Curve

![Sigmoid Curve](Sigmoid%20Curve.png)

Shows:

* Data distribution
* Learned decision boundary
* Probability curve

---

### 🔹 Training Loss Curve

![Training Loss](Training%20Loss.png)

Shows:

* Loss decreasing over time
* Gradient descent convergence
* Learning stability

---

## ⚙️ Core Methods

| Method            | Description                        |
| ----------------- | ---------------------------------- |
| `fit()`           | Train model using gradient descent |
| `predict()`       | Predict class labels (0/1)         |
| `predict_proba()` | Predict probabilities              |
| `compute_loss()`  | Cross-entropy loss                 |
| `plot_loss()`     | Visualize training loss            |

---

## 🧠 Key Learnings

* Logistic regression intuition
* Gradient descent mechanics
* Role of sigmoid function
* Loss minimization process
* Vectorized NumPy operations

---

## 🔥 Future Improvements

* Train/test split
* Feature scaling (standardization)
* Confusion matrix
* Precision / Recall / F1-score
* L2 regularization
* Softmax regression (multi-class)
* Mini-batch gradient descent

---

## 🧰 Tech Stack

* Python
* NumPy
* Matplotlib

---

## 👨‍💻 Author

Built from scratch to master machine learning fundamentals.

---

## ⭐ Support

If this helped you:

⭐ Star the repo
🍴 Fork it
📚 Learn from it
🚀 Build your own version

---

## 📜 License

MIT License - Free to use and modify


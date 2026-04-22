# Logistic Regression from Scratch (NumPy + Matplotlib)

A complete **from-scratch implementation of Logistic Regression** using only **NumPy** and **Matplotlib**, without any machine learning libraries like scikit-learn.

This project is built to help understand the **mathematics, intuition, and training process** behind logistic regression.

---

## 📌 Project Preview

### 📈 Sigmoid Curve

![Sigmoid Curve](Sigmoid%20Curve.png)

### 📉 Training Loss

![Training Loss](Training%20Loss.png)

---

## 📁 Project Structure

```bash id="x7k2lq"
Logistic-Regression-Form-Scratch/
┣ main.py
┣ README.md
┣ Sigmoid Curve.png
┗ Training Loss.png
```

---

## 🚀 Features

* Logistic Regression from scratch
* Binary classification (0 / 1)
* Sigmoid activation function
* Cross-entropy loss function
* Gradient descent optimization
* Probability predictions
* Class predictions
* Visualization of decision curve
* Training loss tracking

---

## 🧠 What is Logistic Regression?

Logistic Regression is a supervised learning algorithm used for **binary classification problems**.

Examples:

* Pass / Fail
* Spam / Not Spam
* Disease / No Disease
* Fraud / Not Fraud

It predicts probabilities between 0 and 1 using a sigmoid function.

---

## 📘 Mathematical Foundation

### 🔹 Hypothesis (Sigmoid Function)

\hat y = \frac{1}{1 + e^{-(wx + b)}}

Where:

* w = weight
* b = bias
* x = input feature

---

### 🔹 Cost Function (Cross Entropy Loss)

L = -\frac{1}{m}\sum \left[y\log(\hat y) + (1-y)\log(1-\hat y)\right]

---

### 🔹 Gradient Descent Updates

```markdown id="g7p2kq"
w = w - α (∂L / ∂w)
b = b - α (∂L / ∂b)
```

Rendered version:

w = w - \alpha \frac{\partial L}{\partial w}

b = b - \alpha \frac{\partial L}{\partial b}

---

## 📊 Dataset Example

```python id="k2m8qv"
X = [1,2,3,4,5,6,7]
y = [0,0,0,0,1,1,1]
```

| Study Hours | Output   |
| ----------- | -------- |
| 1           | 0 (Fail) |
| 2           | 0 (Fail) |
| 3           | 0 (Fail) |
| 4           | 0 (Fail) |
| 5           | 1 (Pass) |
| 6           | 1 (Pass) |
| 7           | 1 (Pass) |

---

## ⚙️ Installation

Clone the repository:

```bash id="p9x0ab"
git clone https://github.com/maroofiums/Logistic-Regression-Form-Scratch.git
cd Logistic-Regression-Form-Scratch
```

Install dependencies:

```bash id="z1v8mq"
pip install numpy matplotlib
```

---

## ▶️ Run the Project

```bash id="t4n6wp"
python main.py
```

---

## ▶️ Usage Example

```python id="v2m8xq"
from main import LogisticRegression

X = [1,2,3,4,5,6,7]
y = [0,0,0,0,1,1,1]

model = LogisticRegression(learning_rate=0.1, epochs=2000)

model.fit(X, y)

print(model.predict([2]))  # [0]
print(model.predict([6]))  # [1]
print(model.predict_proba([4]))
```

---

## 📈 Visualizations

### 🔹 Sigmoid Curve

![Sigmoid Curve](Sigmoid%20Curve.png)

Shows:

* Training data points
* Learned S-shaped probability curve
* Decision threshold at 0.5

---

### 🔹 Training Loss

![Training Loss](Training%20Loss.png)

Shows:

* Loss decreasing over epochs
* Convergence of gradient descent
* Model learning progress

---

## 📌 Model Methods

### `fit(X, y)`

Trains the model using gradient descent.

### `predict(X)`

Returns binary predictions (0 or 1).

### `predict_proba(X)`

Returns probability values between 0 and 1.

### `plot_sigmoid(X, y)`

Plots decision curve.

### `plot_loss()`

Plots training loss curve.

---

## 🎯 Why Build This From Scratch?

Building ML models from scratch helps you understand:

* How learning actually happens
* Why loss decreases
* What gradients really do
* How models make decisions
* The math behind predictions

---

## 🔥 Possible Improvements

* Multi-feature logistic regression
* Accuracy score function
* Confusion matrix
* Precision / recall metrics
* L1 / L2 regularization
* Softmax regression (multi-class)
* Mini-batch gradient descent

---

## 🧰 Tech Stack

* Python
* NumPy
* Matplotlib

---

## 👨‍💻 Author

Built from scratch with curiosity and consistency.

---

## ⭐ Support

If you like this project:

* ⭐ Star the repo
* 🍴 Fork it
* 🧠 Learn from it
* 🔁 Share it

---

## 📜 License

This project is open-source and free to use.

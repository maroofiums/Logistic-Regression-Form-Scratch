import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        X = np.array(X)
        z = self.w * X + self.b
        return self.sigmoid(z)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        eps = 1e-9
        return -(1/m) * np.sum(
            y_true * np.log(y_pred + eps) +
            (1-y_true) * np.log(1-y_pred + eps)
        )

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.w = 0.0
        self.b = 0.0

        m = len(y)

        for epoch in range(self.epochs):

            y_pred = self.predict_proba(X)

            dw = (1/m) * np.sum((y_pred - y) * X)
            db = (1/m) * np.sum(y_pred - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss={loss:.4f}")

    def plot_sigmoid(self, X, y):
        X = np.array(X)
        y = np.array(y)

        x_vals = np.linspace(min(X)-1, max(X)+1, 200)
        probs = self.predict_proba(x_vals)

        plt.figure(figsize=(8,5))
        plt.scatter(X, y, label="Data")
        plt.plot(x_vals, probs, label="Sigmoid Curve")
        plt.axhline(0.5, linestyle="--", alpha=0.6)
        plt.xlabel("Study Hours")
        plt.ylabel("Probability")
        plt.title("Logistic Regression")
        plt.legend()
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(8,5))
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()


X = [1,2,3,4,5,6,7]
y = [0,0,0,0,1,1,1]

model = LogisticRegression(learning_rate=0.1, epochs=2000)
model.fit(X,y)

model.plot_sigmoid(X,y)
model.plot_loss()
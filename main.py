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
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def compute_loss(self, y_true, y_pred):
        eps = 1e-9
        return -(1 / len(y_true)) * np.sum(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        for epoch in range(self.epochs):

            y_pred = self.predict_proba(X)

            # Gradients
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Loss tracking
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history)
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

# Example Usage: 

X = [
    [2, 7],
    [3, 6],
    [4, 6],
    [5, 5],
    [6, 4],
    [7, 3],
    [8, 3]
]

y = [0, 0, 0, 1, 1, 1, 1]

model = LogisticRegression(learning_rate=0.1, epochs=2000)
model.fit(X, y)

test_data = [
    [3, 7],
    [6, 5],
    [8, 2]
]

preds = model.predict(test_data)
probs = model.predict_proba(test_data)

print("\nPredictions:")
for i in range(len(test_data)):
    print(f"Input: {test_data[i]} -> Class: {preds[i]}, Prob: {probs[i]:.4f}")

train_preds = model.predict(X)
acc = model.accuracy(np.array(y), train_preds)

print(f"\nTraining Accuracy: {acc:.4f}")


model.plot_loss()
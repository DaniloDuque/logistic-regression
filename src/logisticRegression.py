import torch
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, w):
        self.w = w

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def forward(self, X):
        return self.sigmoid(X @ self.w)         # Salida en (0, 1)

    def predict(self, X):
        return (self.forward(X) >= 0.5).float() # Umbral en 0.5

    def train(self, X, t, steps=10, alpha=0.1):
        N = X.shape[0]
        for _ in range(steps):
            y = self.forward(X)                # Predicciones sigmoidales
            grad = (X.t() @ (t - y)) / N       # Gradiente de la log-verosimilitud
            self.w = self.w + alpha * grad     # Ascenso por gradiente
        return self

    def accuracy(self, X, t):
        return accuracy_score(t.cpu(), self.predict(X).cpu())

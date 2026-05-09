import torch
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, w):
        self.w = w

    def forward(self, X):
        return torch.sign(X @ self.w)

    def train(self, X, t, steps=10, alpha=0.1):
        for _ in range(steps):
            criteria = X @ self.w * t
            misclassified = (criteria < 0).view(-1)
            grad = alpha * (X * t)[misclassified].sum(0)
            self.w = self.w + grad.view(-1, 1)
        return self

    def accuracy(self, X, t):
        return accuracy_score(t.cpu(), self.forward(X).cpu())

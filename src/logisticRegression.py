import torch
import unittest
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

class TestLogisticRegression(unittest.TestCase):

    # ------------------------------------------------------------------ #
    #  PRUEBA 1: forward() — salida sigmoidal en (0, 1)
    # ------------------------------------------------------------------ #
    def test_forward_output_range(self):
        """
        Objetivo:
            Verificar que forward() devuelve valores en el intervalo
            abierto (0, 1) para cualquier entrada, ya que usa la función
            sigmoidal en lugar de sign() como el perceptrón.

        Entradas:
            X = [[1, 35, 50],
                 [1, 40, 53],
                 [1, 25, 80],
                 [1, 28, 73]]
            w = [[0.3], [0.2], [0.25]]

        Salidas esperadas:
            Todos los valores y = sigmoid(X @ w) deben cumplir 0 < y < 1.
        """
        X = torch.tensor([[1, 35.0, 50],
                          [1, 40.0, 53],
                          [1, 25.0, 80],
                          [1, 28.0, 73]])
        w = torch.tensor([[0.3], [0.2], [0.25]])
        model = LogisticRegression(w)

        y = model.forward(X)

        self.assertEqual(y.shape, (4, 1),
                         "La salida debe tener forma (N, 1)")
        self.assertTrue((y > 0).all().item(),
                        "Todos los valores deben ser > 0")
        self.assertTrue((y < 1).all().item(),
                        "Todos los valores deben ser < 1")

    # ------------------------------------------------------------------ #
    #  PRUEBA 2: forward() — valores extremos conocidos
    # ------------------------------------------------------------------ #
    def test_forward_known_values(self):
        """
        Objetivo:
            Verificar que forward() calcula correctamente sigmoid(z)
            para valores conocidos, a diferencia del perceptrón que
            devuelve únicamente -1 o +1.

        Entradas:
            X = [[1]]   (muestra con una sola característica)
            w = [[0]]   → z = 0  → sigmoid(0) = 0.5  (esperado)
            w = [[100]] → z = 100 → sigmoid(100) ≈ 1  (esperado)
            w = [[-100]] → z = -100 → sigmoid(-100) ≈ 0 (esperado)

        Salidas esperadas:
            sigmoid(0)    ≈ 0.5
            sigmoid(100)  ≈ 1.0
            sigmoid(-100) ≈ 0.0
        """
        X = torch.tensor([[1.0]])

        # sigmoid(0) = 0.5
        model = LogisticRegression(torch.tensor([[0.0]]))
        self.assertAlmostEqual(model.forward(X).item(), 0.5, places=5,
                               msg="sigmoid(0) debe ser 0.5")

        # sigmoid(100) ≈ 1.0
        model = LogisticRegression(torch.tensor([[100.0]]))
        self.assertAlmostEqual(model.forward(X).item(), 1.0, places=4,
                               msg="sigmoid(100) debe aproximarse a 1.0")

        # sigmoid(-100) ≈ 0.0
        model = LogisticRegression(torch.tensor([[-100.0]]))
        self.assertAlmostEqual(model.forward(X).item(), 0.0, places=4,
                               msg="sigmoid(-100) debe aproximarse a 0.0")

    # ------------------------------------------------------------------ #
    #  PRUEBA 3: train() — los pesos cambian tras el entrenamiento
    # ------------------------------------------------------------------ #
    def test_train_weights_update(self):
        """
        Objetivo:
            Verificar que train() actualiza los pesos w en cada paso,
            usando el gradiente sobre TODAS las muestras (no solo las
            mal clasificadas como en el perceptrón).

        Entradas:
            X = [[1, 35, 50], [1, 40, 53], [1, 25, 80], [1, 28, 73]]
            t = [[1], [1], [0], [0]]
            w_inicial = [[0.3], [0.2], [0.25]]
            steps = 10, alpha = 0.1

        Salidas esperadas:
            w_final != w_inicial  (los pesos deben haber cambiado)
        """
        X = torch.tensor([[1, 35.0, 50],
                          [1, 40.0, 53],
                          [1, 25.0, 80],
                          [1, 28.0, 73]])
        t = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
        w_inicial = torch.tensor([[0.3], [0.2], [0.25]])

        model = LogisticRegression(w_inicial.clone())
        model.train(X, t, steps=10, alpha=0.1)

        self.assertFalse(torch.allclose(model.w, w_inicial),
                         "Los pesos deben cambiar tras el entrenamiento")

    # ------------------------------------------------------------------ #
    #  PRUEBA 4: train() — el modelo mejora con el entrenamiento
    # ------------------------------------------------------------------ #
    def test_train_improves_accuracy(self):
        """
        Objetivo:
            Verificar que train() mejora (o mantiene) la precisión del
            modelo tras el entrenamiento, lo que valida que el descenso
            de gradiente sobre la log-verosimilitud funciona correctamente.

        Entradas:
            X = [[1, 35, 50], [1, 40, 53], [1, 25, 80], [1, 28, 73]]
            t = [[1], [1], [0], [0]]
            w = [[0.0], [0.0], [0.0]]  (pesos neutros al inicio)
            steps = 100, alpha = 0.1

        Salidas esperadas:
            accuracy_final >= accuracy_inicial
        """
        X = torch.tensor([[1, 35.0, 50],
                          [1, 40.0, 53],
                          [1, 25.0, 80],
                          [1, 28.0, 73]])
        t = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
        w = torch.tensor([[0.0], [0.0], [0.0]])

        model = LogisticRegression(w.clone())
        acc_inicial = model.accuracy(X, t)
        model.train(X, t, steps=100, alpha=0.1)
        acc_final = model.accuracy(X, t)

        self.assertGreaterEqual(acc_final, acc_inicial,
                                "La precisión debe mejorar o mantenerse")


if __name__ == "__main__":
    unittest.main(verbosity=2)

import torch

def train_with_history(model, X_train, y_train, X_test, y_test, steps: int = 1000, alpha: float = 0.1) -> tuple:
    """
    Entrena el modelo paso a paso y registra el MAE en cada iteración
    para los conjuntos de entrenamiento y prueba.
    Devuelve una tupla (train_errors, test_errors) para graficar la curva de aprendizaje.
    """
    train_errors, test_errors = [], []
    for _ in range(steps):
        model.train(X_train, y_train, steps=1, alpha=alpha)
        train_errors.append(compute_mae(y_train, model.forward(X_train)))
        test_errors.append(compute_mae(y_test,  model.forward(X_test)))
    return train_errors, test_errors

def compute_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calcula el Error Absoluto Medio:
        eMAE = (1/N) * sum|t_i - t̂_i|
    """
    return torch.mean(torch.abs(y_true - y_pred)).item()
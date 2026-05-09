import torch

def train_with_history(model, X_train, y_train, steps: int = 1000, alpha: float = 0.1) -> list:
    """
    Trains the model step by step and records the MAE at each iteration.
    Returns a list of errors to plot the training curve.
    """
    errors = []
    for _ in range(steps):
        model.train(X_train, y_train, steps=1, alpha=alpha)
        y_pred = model.forward(X_train)
        errors.append(compute_mae(y_train, y_pred))
    return errors

def compute_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the Mean Absolute Error:
        eMAE = (1/N) * sum|t_i - t̂_i|
    """
    return torch.mean(torch.abs(y_true - y_pred)).item()
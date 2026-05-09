import torch
import numpy as np
from data_generator import generate_data
from logistic_regression import LogisticRegression
from trainer import compute_mae

def run_experiment(separable: bool, steps: int = 1000, alpha: float = 0.1) -> list:
    """
    Runs 10 independent trials with randomly generated data each time.
    Returns a list of MAE values on the test set.
    """
    mae_scores = []
    for seed in range(10):
        X_train, X_test, y_train, y_test = generate_data(
            separable=separable, n_samples=500, random_state=seed
        )
        w = torch.zeros(X_train.shape[1], 1)
        model = LogisticRegression(w)
        model.train(X_train, y_train, steps=steps, alpha=alpha)
        mae_scores.append(compute_mae(y_test, model.forward(X_test)))
    return mae_scores

def print_mae_table(label: str, mae_train: float, mae_test: float):
    """
    Prints a formatted single-row MAE result table.
    """
    print("=" * 55)
    print(f"{'Caso':<30} {'MAE Entren.':>10} {'MAE Prueba':>10}")
    print("-" * 55)
    print(f"{label:<30} {mae_train:>10.4f} {mae_test:>10.4f}")
    print("=" * 55)

def print_runs_table(mae_separable: list, mae_nonseparable: list):
    """
    Prints a formatted table with mean and std for 10 runs.
    """
    print("=" * 55)
    print(f"{'Caso':<30} {'MAE Medio':>10} {'Desv. Típ.':>10}")
    print("-" * 55)
    print(f"{'Linealmente separable':<30} {np.mean(mae_separable):>10.4f} {np.std(mae_separable):>10.4f}")
    print(f"{'No linealmente separable':<30} {np.mean(mae_nonseparable):>10.4f} {np.std(mae_nonseparable):>10.4f}")
    print("=" * 55)
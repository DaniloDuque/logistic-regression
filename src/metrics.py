import torch
import numpy as np
from data_generator import generate_data
from logistic_regression import LogisticRegression
from trainer import compute_mae
from sklearn.metrics import accuracy_score

def compute_accuracy(model, X: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calcula la precisión de clasificación usando model.predict().
    """
    return accuracy_score(y.cpu(), model.predict(X).cpu())

def run_experiment(separable: bool, steps: int = 1000, alpha: float = 0.1,
                   device=None) -> dict:
    """
    Ejecuta 10 ensayos independientes con datos generados aleatoriamente cada vez.
    Devuelve un dict con listas de valores de MAE y precisión sobre el conjunto de prueba.
    """
    mae_scores, acc_scores = [], []
    for seed in range(10):
        X_train, X_test, y_train, y_test = generate_data(
            separable=separable, n_samples=500, random_state=seed, device=device
        )
        w = torch.zeros(X_train.shape[1], 1, device=device)
        model = LogisticRegression(w)
        model.train(X_train, y_train, steps=steps, alpha=alpha)
        mae_scores.append(compute_mae(y_test, model.forward(X_test)))
        acc_scores.append(compute_accuracy(model, X_test, y_test))
    return {"mae": mae_scores, "acc": acc_scores}

def print_single_result(label: str, mae_train: float, mae_test: float,
                        acc_train: float, acc_test: float):
    """
    Imprime una tabla de resultados formateada para una sola ejecución.
    """
    print("=" * 65)
    print(f"{'Caso':<30} {'MAE Entren.':>10} {'MAE Prueba':>10} {'Acc Prueba':>10}")
    print("-" * 65)
    print(f"{label:<30} {mae_train:>10.4f} {mae_test:>10.4f} {acc_test:>10.4f}")
    print("=" * 65)

def print_runs_table(results_sep: dict, results_ns: dict):
    """
    Imprime una tabla resumen formateada para 10 ejecuciones con
    media, desviación típica, mínimo y máximo para MAE y precisión.
    """
    def row(label, scores):
        return (f"{label:<26} "
                f"{np.mean(scores):>8.4f} "
                f"{np.std(scores):>8.4f} "
                f"{np.min(scores):>8.4f} "
                f"{np.max(scores):>8.4f}")

    print("=" * 70)
    print(f"{'Caso':<26} {'Media':>8} {'Desv.Típ':>8} {'Mín':>8} {'Máx':>8}")
    print("-" * 70)
    print("MAE (prueba)")
    print(row("  Linealmente separable",    results_sep["mae"]))
    print(row("  No linealmente separable", results_ns["mae"]))
    print("-" * 70)
    print("Precisión (prueba)")
    print(row("  Linealmente separable",    results_sep["acc"]))
    print(row("  No linealmente separable", results_ns["acc"]))
    print("=" * 70)

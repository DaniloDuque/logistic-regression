import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import friedmanchisquare

from logistic_regression import LogisticRegression
from llm_classifier import classify_batch, load_classifiers, MODELS as ZERO_MODELS
from few_shot_classifier import classify_few_shot_batch, load_few_shot_classifiers

TREATMENTS = [
    "TF-IDF + LogReg",
    "BERT + LogReg",
    "RoBERTa + LogReg",
    "Selectra (zero-shot)",
    "BART (zero-shot)",
    "Selectra 1-shot",
    "Selectra 3-shot",
    "Selectra 5-shot",
]

_FEWSHOT_CONFIGS = [
    ("1shot_newline", "Selectra 1-shot"),
    ("3shot_newline", "Selectra 3-shot"),
    ("5shot_newline", "Selectra 5-shot"),
]


def _train_logreg(X_train, y_train, steps):
    X = torch.tensor(X_train, dtype=torch.float64)
    t = torch.tensor(y_train, dtype=torch.float64).unsqueeze(1)
    model = LogisticRegression(torch.zeros((X.shape[1], 1), dtype=torch.float64))
    model.train(X, t, steps=steps)
    return model


def run_30_corridas(all_texts, all_labels, X_tfidf, emb_bert, emb_roberta,
                    n_runs=30, test_size=0.2, steps_lr=1000):
    """
    Ejecuta n_runs corridas con partición aleatoria 80/20.
    Retorna dict {treatment: list[float]} con las accuracies por corrida.
    """
    print("Cargando clasificadores LLM...")
    zero_clfs    = load_classifiers()
    fewshot_clfs = load_few_shot_classifiers()
    print("Modelos cargados.\n")

    y_all   = np.array(all_labels)
    results = {t: [] for t in TREATMENTS}
    idx     = np.arange(len(all_texts))

    for run in range(n_runs):
        print(f"── Corrida {run + 1}/{n_runs} ──")
        idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=run)
        y_train, y_test     = y_all[idx_train], y_all[idx_test]
        texts_test          = [all_texts[i] for i in idx_test]

        # TF-IDF + LogReg
        m = _train_logreg(X_tfidf[idx_train], y_train, steps_lr)
        results["TF-IDF + LogReg"].append(float(m.accuracy(
            torch.tensor(X_tfidf[idx_test], dtype=torch.float64),
            torch.tensor(y_test, dtype=torch.float64).unsqueeze(1)
        )))

        # BERT + LogReg
        m = _train_logreg(emb_bert[idx_train], y_train, steps_lr)
        results["BERT + LogReg"].append(float(m.accuracy(
            torch.tensor(emb_bert[idx_test], dtype=torch.float64),
            torch.tensor(y_test, dtype=torch.float64).unsqueeze(1)
        )))

        # RoBERTa + LogReg
        m = _train_logreg(emb_roberta[idx_train], y_train, steps_lr)
        results["RoBERTa + LogReg"].append(float(m.accuracy(
            torch.tensor(emb_roberta[idx_test], dtype=torch.float64),
            torch.tensor(y_test, dtype=torch.float64).unsqueeze(1)
        )))

        # Zero-shot LLMs
        for model_key, treatment in [("selectra", "Selectra (zero-shot)"),
                                      ("bart",     "BART (zero-shot)")]:
            cfg   = ZERO_MODELS[model_key]
            preds = classify_batch(zero_clfs[model_key], texts_test, cfg["labels"], cfg["template"])
            results[treatment].append(float((preds == y_test).mean()))

        # Few-shot (Selectra)
        for config_key, treatment in _FEWSHOT_CONFIGS:
            preds = classify_few_shot_batch(fewshot_clfs["selectra"], texts_test,
                                            model_key="selectra", config_key=config_key)
            results[treatment].append(float((preds == y_test).mean()))

    print("\n✓ 30 corridas completadas.")
    return results


def print_30_results(results):
    """Imprime accuracy por corrida y tabla resumen de media y desviación estándar."""
    n_runs = max(len(v) for v in results.values())
    treatments = list(results.keys())
    col = 10

    # Encabezado de corridas
    header = f"{'Corrida':>8s}  " + "  ".join(f"{t[:col]:>{col}s}" for t in treatments)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for i in range(n_runs):
        row = f"{i + 1:>8d}  " + "  ".join(f"{results[t][i]:>{col}.4f}" for t in treatments)
        print(row)
    print("=" * len(header))


def print_summary(results):
    """Imprime tabla resumen de media y desviación estándar por tratamiento."""
    print("\n" + "=" * 62)
    print(f"{'Tratamiento':<28s}  {'Media Acc':>9s}  {'Desv. Típ.':>10s}")
    print("=" * 62)
    for treatment, accs in results.items():
        print(f"{treatment:<28s}  {np.mean(accs):>9.4f}  {np.std(accs):>10.4f}")
    print("=" * 62)


def plot_30_results(results, output_path="figures/sec5_30runs_accuracy.png"):
    """Gráfico de barras con barras de error para las 30 corridas."""
    labels = list(results.keys())
    means  = [np.mean(results[t]) for t in labels]
    stds   = [np.std(results[t])  for t in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(labels)), means, yerr=stds, capsize=5,
           color='steelblue', alpha=0.8, ecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy media (30 corridas, 80/20) — todos los tratamientos")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Línea base aleatoria')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def friedman_nemenyi(results):
    """
    Ejecuta el test de Friedman y el post-hoc de Nemenyi.
    Imprime resultados y retorna la tabla de p-valores.
    """
    import scikit_posthocs as sp

    groups = [np.array(results[t]) for t in TREATMENTS]
    stat, p_value = friedmanchisquare(*groups)
    print(f"Friedman χ² = {stat:.4f},  p = {p_value:.6f}")

    if p_value < 0.05:
        print("→ Se rechaza H0: hay diferencias significativas entre modelos (p < 0.05)\n")
    else:
        print("→ No hay evidencia de diferencias significativas\n")

    data_matrix = np.array([results[t] for t in TREATMENTS]).T  # (30, 8)
    nemenyi = sp.posthoc_nemenyi_friedman(data_matrix)
    nemenyi.columns = TREATMENTS
    nemenyi.index   = TREATMENTS

    print("Tabla de p-valores (Nemenyi):")
    print(nemenyi.round(4).to_string())
    return nemenyi

from transformers import pipeline
import numpy as np
from transformers.pipelines.pt_utils import KeyDataset
from text_dataset import TextDataset
    

MODELS = {
    "selectra": {
        "model_id":  "Recognai/zeroshot_selectra_medium",
        "labels":    ["simple", "complejo"],
        "template":  "Este texto es {}.",
    },
    "bart": {
        "model_id":  "facebook/bart-large-mnli",
        "labels":    ["simple", "complex"],
        "template":  "This example is {}.",
    },
}


def load_classifiers(model_keys=None, device=None):
    """
    Carga y retorna un dict {key: pipeline} para los modelos indicados.
    Si model_keys es None, carga todos los definidos en MODELS.
    """
    keys = model_keys or list(MODELS.keys())
    classifiers = {}
    for key in keys:
        cfg = MODELS[key]
        print(f"Cargando {key} ({cfg['model_id']})...")
        classifiers[key] = pipeline(
            "zero-shot-classification",
            model=cfg["model_id"],
            device=device,
        )
    return classifiers


def classify_batch(classifier, texts, labels, template, batch_size=32):
    """
    Clasifica una lista de textos usando un Dataset para eficiencia en GPU.
    Retorna array de enteros: 0 = primer label, 1 = segundo label.
    """
    dataset = TextDataset(texts)
    preds = []

    for i, result in enumerate(classifier(
        dataset,
        candidate_labels=labels,
        hypothesis_template=template,
        batch_size=batch_size
    )):
        preds.append(0 if result["labels"][0] == labels[0] else 1)
        if i % 5000 == 0:
            print(f"  {i}/{len(texts)} procesados...")

    return np.array(preds)


def classify_all(classifiers, texts):
    """
    Corre classify_batch para cada modelo cargado.
    Retorna dict {key: np.array de predicciones}.
    """
    return {
        key: classify_batch(
            clf,
            texts,
            MODELS[key]["labels"],
            MODELS[key]["template"]
        )
        for key, clf in classifiers.items()
    }


def show_examples(classifiers, texts):
    """Imprime tabla comparativa para una lista corta de textos."""
    headers = list(classifiers.keys())
    print("=" * 76)
    print(f"{'Texto':<45s}" + "".join(f"  {h:>12s}" for h in headers))
    print("=" * 76)
    for texto in texts:
        row = texto[:42] + "..." if len(texto) > 45 else texto
        line = f"{row:<45s}"
        for key, clf in classifiers.items():
            cfg = MODELS[key]
            res = clf(texto,
                      candidate_labels=cfg["labels"],
                      hypothesis_template=cfg["template"])
            line += f"  {res['labels'][0]:>6s} ({res['scores'][0]:.2f})"
        print(line)
    print("=" * 76)


def compute_llm_metrics(predictions, y_true):
    """
    Dado un dict {key: preds} y el array de etiquetas reales,
    retorna un dict {key: {"accuracy": ..., "mae": ...}}.
    """
    y = np.array(y_true)
    return {
        key: {
            "accuracy": (preds == y).mean(),
            "mae":      np.abs(preds - y).mean(),
        }
        for key, preds in predictions.items()
    }


def print_metrics_table(metrics):
    """Imprime tabla de accuracy y MAE por modelo."""
    print("\n" + "=" * 50)
    print(f"{'Modelo':<30s}  {'Accuracy':>8s}  {'MAE':>6s}")
    print("=" * 50)
    for key, m in metrics.items():
        print(f"{key:<30s}  {m['accuracy']:>8.4f}  {m['mae']:>6.4f}")
    print("=" * 50)

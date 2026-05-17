# Basado en: Chamieh et al. (2024) "LLMs in Short Answer Scoring:
# Limitations and Promise of Zero-Shot and Few-Shot Approaches"

from transformers import pipeline
import numpy as np
from text_dataset import TextDataset


# ── Ejemplos few-shot por clase (extraídos del dataset FEINA) ─────
# Deben elegirse manualmente: ejemplos claros y representativos
FEW_SHOT_EXAMPLES = [
    # (texto, etiqueta)
    ("El niño fue al colegio.", "simple"),
    ("La desinhibición conductual asociada a lesiones prefrontales "
     "compromete la regulación ejecutiva del comportamiento.", "complejo"),
    ("El médico le recetó una pastilla para el dolor.", "simple"),
    ("La farmacocinética del principio activo determina su biodisponibilidad "
     "y el perfil de absorción gastrointestinal.", "complejo"),
    ("El perro ladró toda la noche.", "simple"),
    ("La modulación alostérica de los receptores ionotrópicos altera "
     "la cinética de activación del canal.", "complejo"),
]

# ── Configuraciones few-shot (del artículo: Tabla 2/3/4) ──────────
CONFIGS = {
    "1shot": {"n_shots": 1},
    "3shot": {"n_shots": 3},
    "5shot": {"n_shots": 5},
}

MODELS = {
    "selectra": {
        "model_id": "Recognai/zeroshot_selectra_medium",
        "labels":   ["simple", "complejo"],
        "lang":     "es",
    },
    "bart": {
        "model_id": "facebook/bart-large-mnli",
        "labels":   ["simple", "complex"],
        "lang":     "en",
    },
}


def _examples_for_label(label, n_shots):
    """Retorna hasta n_shots ejemplos del label indicado."""
    return [text for text, lbl in FEW_SHOT_EXAMPLES if lbl == label][:n_shots]


def build_hypothesis(label, n_shots, lang):
    """
    Construye una hipótesis enriquecida con ejemplos few-shot para el label dado.
    El modelo NLI puntúa: ¿el texto de entrada implica esta hipótesis?
    """
    examples = _examples_for_label(label, n_shots)
    example_str = "; ".join(f'"{e}"' for e in examples)
    if lang == "es":
        return f"Este texto es {label}, similar a: {example_str}."
    return f"This text is {label}, similar to: {example_str}."


def load_few_shot_classifiers(model_keys=None, device=None):
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


def classify_few_shot_batch(classifier, texts, model_key, config_key,
                             batch_size=32):
    """
    Clasifica usando hipótesis enriquecidas con ejemplos few-shot.
    El texto es la premisa; la hipótesis incluye ejemplos representativos
    del label, aprovechando correctamente la cabeza NLI del modelo.
    """
    cfg_model = MODELS[model_key]
    n_shots   = CONFIGS[config_key]["n_shots"]
    labels    = cfg_model["labels"]
    lang      = cfg_model["lang"]

    # Una hipótesis enriquecida por label (fija para todo el batch)
    hypotheses = [build_hypothesis(lbl, n_shots, lang) for lbl in labels]

    dataset = TextDataset(texts)
    preds   = []

    for i, result in enumerate(classifier(
        dataset,
        candidate_labels=hypotheses,
        batch_size=batch_size,
    )):
        # Mapear la hipótesis ganadora de vuelta al índice del label
        winning_hypothesis = result["labels"][0]
        preds.append(hypotheses.index(winning_hypothesis))
        if i % 500 == 0:
            print(f"  [{config_key}] {i}/{len(texts)} procesados...")

    return np.array(preds)


def classify_all_configs(classifiers, texts):
    """
    Corre todas las configuraciones para todos los modelos.
    Retorna dict {model_key: {config_key: np.array}}.
    """
    results = {}
    for model_key, clf in classifiers.items():
        results[model_key] = {}
        for config_key in CONFIGS:
            print(f"\n── {model_key} / {config_key} ──")
            results[model_key][config_key] = classify_few_shot_batch(
                clf, texts, model_key, config_key
            )
    return results


def show_few_shot_examples(classifiers, texts):
    """Muestra ejemplos cualitativos para todas las configs y modelos."""
    for model_key, clf in classifiers.items():
        cfg_model = MODELS[model_key]
        print(f"\n{'='*70}")
        print(f"Modelo: {model_key}")
        print(f"{'Texto':<40s}  {'Config':<20s}  {'Pred':>10s}")
        print(f"{'='*70}")
        for config_key, cfg_shot in CONFIGS.items():
            n_shots    = cfg_shot["n_shots"]
            hypotheses = [build_hypothesis(lbl, n_shots, cfg_model["lang"])
                          for lbl in cfg_model["labels"]]
            dataset = TextDataset(texts)
            for text, result in zip(texts, clf(dataset, candidate_labels=hypotheses)):
                winning = result["labels"][0]
                label   = cfg_model["labels"][hypotheses.index(winning)]
                row     = text[:37] + "..." if len(text) > 40 else text
                print(f"{row:<40s}  {config_key:<20s}  {label:>10s}")
        print(f"{'='*70}")


def compute_few_shot_metrics(results, y_true):
    """
    Retorna dict {model_key: {config_key: {"accuracy": ..., "mae": ...}}}.
    """
    y = np.array(y_true)
    metrics = {}
    for model_key, configs in results.items():
        metrics[model_key] = {}
        for config_key, preds in configs.items():
            metrics[model_key][config_key] = {
                "accuracy": (preds == y).mean(),
                "mae":      np.abs(preds - y).mean(),
            }
    return metrics


def print_few_shot_metrics(metrics):
    """Imprime tabla comparativa de todas las configs y modelos."""
    print("\n" + "=" * 60)
    print(f"{'Modelo':<20s}  {'Config':<20s}  {'Acc':>8s}  {'MAE':>6s}")
    print("=" * 60)
    for model_key, configs in metrics.items():
        for config_key, m in configs.items():
            print(f"{model_key:<20s}  {config_key:<20s}  "
                  f"{m['accuracy']:>8.4f}  {m['mae']:>6.4f}")
    print("=" * 60)

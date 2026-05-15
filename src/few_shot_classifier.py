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
    "1shot_newline": {"n_shots": 1,  "delimiter": "\n"},
    "3shot_newline": {"n_shots": 3,  "delimiter": "\n"},
    "5shot_newline": {"n_shots": 5,  "delimiter": "\n"},
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


def build_few_shot_prompt(text, n_shots, delimiter, lang="es"):
    """
    Construye el prompt few-shot siguiendo el formato del artículo
    (Appendix B). El texto a clasificar va al final sin etiqueta.
    """
    if lang == "es":
        intro = ("Clasifica el siguiente texto como 'simple' o 'complejo'. "
                 "Responde solo con la etiqueta.")
        answer_prefix = "Texto:"
        label_prefix  = "Etiqueta:"
    else:
        intro = ("Classify the following text as 'simple' or 'complex'. "
                 "Return only the label.")
        answer_prefix = "Text:"
        label_prefix  = "Label:"

    # Tomar n_shots ejemplos (alternando clases para balance)
    shots = FEW_SHOT_EXAMPLES[:n_shots * 2]  # 2 por shot (1 simple, 1 complejo)
    shots = shots[:n_shots]

    lines = [intro]
    for ejemplo, etiqueta in shots:
        lines.append(f"{answer_prefix} {ejemplo} -> {label_prefix} {etiqueta}")

    # Texto a clasificar (sin etiqueta — el modelo debe completar)
    lines.append(f"{answer_prefix} {text} -> {label_prefix}")

    return delimiter.join(lines)


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
    Clasifica usando el prompt few-shot como hipótesis contextualizada.
    Sigue la estrategia del artículo: incluir ejemplos en el prompt.
    """
    cfg_model  = MODELS[model_key]
    cfg_shot   = CONFIGS[config_key]
    labels     = cfg_model["labels"]
    lang       = cfg_model["lang"]
    n_shots    = cfg_shot["n_shots"]
    delimiter  = cfg_shot["delimiter"]

    # Construir prompts few-shot para cada texto
    prompts = [
        build_few_shot_prompt(t, n_shots, delimiter, lang)
        for t in texts
    ]

    dataset = TextDataset(prompts)
    preds   = []

    template = "{}"  # El prompt ya está completo, la hipótesis ES el prompt
    for i, result in enumerate(classifier(
        dataset,
        candidate_labels=labels,
        hypothesis_template="{}",
        batch_size=batch_size,
    )):
        preds.append(0 if result["labels"][0] == labels[0] else 1)
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
            prompts = [
                build_few_shot_prompt(t, cfg_shot["n_shots"], cfg_shot["delimiter"], cfg_model["lang"])
                for t in texts
            ]
            dataset = TextDataset(prompts)
            for text, result in zip(texts, clf(dataset, candidate_labels=cfg_model["labels"], hypothesis_template="{}")):
                row = text[:37] + "..." if len(text) > 40 else text
                print(f"{row:<40s}  {config_key:<20s}  {result['labels'][0]:>10s}")
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

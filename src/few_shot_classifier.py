# few_shot_classifier.py
# Basado en: Chamieh et al. (2024) "LLMs in Short Answer Scoring:
# Limitations and Promise of Zero-Shot and Few-Shot Approaches"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np


# ── Ejemplos few-shot por clase (extraídos del dataset FEINA) ─────
FEW_SHOT_EXAMPLES = [
    ("El niño fue al colegio.", "simple"),
    ("La desinhibición conductual asociada a lesiones prefrontales "
     "compromete la regulación ejecutiva del comportamiento.", "complejo"),
    ("El médico le recetó una pastilla para el dolor.", "simple"),
    ("La farmacocinética del principio activo determina su biodisponibilidad "
     "y el perfil de absorción gastrointestinal.", "complejo"),
    ("El perro ladró toda la noche.", "simple"),
    ("La modulación alostérica de los receptores ionotrópicos altera "
     "la cinética de activación del canal.", "complejo"),
    ("El agua hierve a 100 grados.", "simple"),
    ("La homeostasis glucémica depende de la regulación pancreática "
     "de insulina y glucagón.", "complejo"),
    ("Compró pan en la tienda.", "simple"),
    ("La volatilidad implícita refleja las expectativas del mercado "
     "sobre la dispersión futura del subyacente.", "complejo"),
]

# ── Configuraciones: número de ejemplos por clase ─────────────────
# n_shots = ejemplos POR CLASE → 1-shot = 2 ejemplos totales (1 simple + 1 complejo)
CONFIGS = {
    "1shot": {"n_shots": 1},
    "3shot": {"n_shots": 3},
    "5shot": {"n_shots": 5},
}

MODELS = {
    "gemma-2b-it": {
        "model_id": "google/gemma-2b-it",
        "causal":   True,
    },
    "qwen2.5-1.5b": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "causal":   True,
    },
}

LABELS = ["simple", "complejo"]


# ── Construcción de prompts ───────────────────────────────────────

def _get_examples(n_shots):
    """
    Retorna n_shots ejemplos por clase alternados (simple, complejo, simple, ...),
    siguiendo la recomendación del artículo de mantener balance.
    """
    simples   = [(t, l) for t, l in FEW_SHOT_EXAMPLES if l == "simple"][:n_shots]
    complejos = [(t, l) for t, l in FEW_SHOT_EXAMPLES if l == "complejo"][:n_shots]
    # Intercalar para que el modelo no aprenda sesgo posicional
    examples = []
    for s, c in zip(simples, complejos):
        examples.append(s)
        examples.append(c)
    return examples


def build_prompt_gemma(text, n_shots):
    """
    Prompt para Gemma-2b-it usando el formato de instrucción de Gemma.
    Gemma-it usa: <start_of_turn>user ... <end_of_turn><start_of_turn>model
    """
    examples = _get_examples(n_shots)
    example_lines = "\n".join(
        f"Texto: {t}\nEtiqueta: {l}" for t, l in examples
    )
    return (
        "<start_of_turn>user\n"
        "Clasifica el siguiente texto como 'simple' o 'complejo'. "
        "Responde únicamente con la etiqueta.\n\n"
        f"{example_lines}\n\n"
        f"Texto: {text}\n"
        "Etiqueta:<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def build_prompt_qwen(text, n_shots):
    """
    Prompt para Qwen2.5-Instruct usando el formato de chat Qwen.
    """
    examples = _get_examples(n_shots)
    example_lines = "\n".join(
        f"- Texto: \"{t}\" → {l}" for t, l in examples
    )
    return (
        "<|im_start|>system\n"
        "Eres un clasificador de complejidad textual en español. "
        "Responde únicamente con 'simple' o 'complejo', sin explicación.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Ejemplos de clasificación:\n{example_lines}\n\n"
        f"Ahora clasifica este texto: {text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


_PROMPT_BUILDERS = {
    "gemma-2b-it":  build_prompt_gemma,
    "qwen2.5-1.5b": build_prompt_qwen,
}


# ── Carga de modelos ──────────────────────────────────────────────

def load_few_shot_classifiers(model_keys=None, device=None):
    keys   = model_keys or list(MODELS.keys())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clfs   = {}
    for key in keys:
        model_id = MODELS[key]["model_id"]
        print(f"Cargando {key} ({model_id})...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"           # ← mismo fix
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        model.eval()
        clfs[key] = (tokenizer, model, device)
    return clfs


# ── Generación y decodificación ───────────────────────────────────

def _generate_batch(tokenizer, model, device, prompts, batch_size=16):
    """Genera respuestas para una lista de prompts. Retorna lista de strings."""
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch  = prompts[i: i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=768
        ).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,          # greedy — determinista
            )
        # Decodificar solo los tokens nuevos
        input_len = inputs["input_ids"].shape[1]
        new_ids   = ids[:, input_len:]
        outputs.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        if i % 500 == 0:
            print(f"  {i}/{len(prompts)} procesados...")
    return outputs


def _parse_label(generated_text):
    """Extrae 'simple' o 'complejo' del texto generado. Default: 'simple'."""
    text = generated_text.strip().lower()
    if "complejo" in text:
        return 1
    if "simple" in text:
        return 0
    return 0   # fallback


# ── Clasificación ─────────────────────────────────────────────────

def classify_few_shot_batch(classifier, texts, model_key, config_key,
                             batch_size=16):
    tokenizer, model, device = classifier
    n_shots       = CONFIGS[config_key]["n_shots"]
    prompt_builder = _PROMPT_BUILDERS[model_key]
    prompts        = [prompt_builder(t, n_shots) for t in texts]
    generated      = _generate_batch(tokenizer, model, device, prompts, batch_size)
    return np.array([_parse_label(g) for g in generated])


def classify_all_configs(classifiers, texts):
    """Retorna dict {model_key: {config_key: np.array}}."""
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
    """Muestra tabla comparativa de ejemplos para todas las configs y modelos."""
    for model_key, clf in classifiers.items():
        tokenizer, model, device = clf
        print(f"\n{'='*72}")
        print(f"Modelo: {model_key}")
        print(f"{'Texto':<38s}  {'Config':<10s}  {'Predicción':>10s}")
        print(f"{'='*72}")
        for config_key in CONFIGS:
            n_shots        = CONFIGS[config_key]["n_shots"]
            prompt_builder = _PROMPT_BUILDERS[model_key]
            for text in texts:
                prompt    = prompt_builder(text, n_shots)
                generated = _generate_batch(tokenizer, model, device, [prompt])[0]
                label     = "complejo" if _parse_label(generated) == 1 else "simple"
                row       = text[:35] + "..." if len(text) > 38 else text
                print(f"{row:<38s}  {config_key:<10s}  {label:>10s}")
        print(f"{'='*72}")


def compute_few_shot_metrics(results, y_true):
    """Retorna dict {model_key: {config_key: {'accuracy': ..., 'mae': ...}}}."""
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
    """Imprime tabla de resultados."""
    print("\n" + "=" * 60)
    print(f"{'Modelo':<20s}  {'Config':<10s}  {'Accuracy':>8s}  {'MAE':>6s}")
    print("=" * 60)
    for model_key, configs in metrics.items():
        for config_key, m in configs.items():
            print(f"{model_key:<20s}  {config_key:<10s}  "
                  f"{m['accuracy']:>8.4f}  {m['mae']:>6.4f}")
    print("=" * 60)

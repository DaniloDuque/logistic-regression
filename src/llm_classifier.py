# llm_classifier.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np


MODELS = {
    "gemma-2b-it": {
        "model_id": "google/gemma-2b-it",
    },
    "qwen2.5-1.5b": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    },
}

LABELS = ["simple", "complejo"]

_ZERO_SHOT_PROMPTS = {
    "gemma-2b-it": (
        "<start_of_turn>user\n"
        "Clasifica el siguiente texto como 'simple' o 'complejo'. "
        "Responde únicamente con la etiqueta, sin explicación.\n\n"
        "Texto: {text}\n"
        "Etiqueta:<end_of_turn>\n"
        "<start_of_turn>model\n"
    ),
    "qwen2.5-1.5b": (
        "<|im_start|>system\n"
        "Eres un clasificador de complejidad textual en español. "
        "Responde únicamente con 'simple' o 'complejo', sin explicación.<|im_end|>\n"
        "<|im_start|>user\n"
        "Clasifica este texto: {text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}


# ── Carga ─────────────────────────────────────────────────────────

def load_classifiers(model_keys=None, device=None):
    """Carga y retorna dict {key: (tokenizer, model, device)}."""
    keys   = model_keys or list(MODELS.keys())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clfs   = {}
    for key in keys:
        model_id = MODELS[key]["model_id"]
        print(f"Cargando {key} ({model_id})...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        clfs[key] = (tokenizer, model, device)
    return clfs


# ── Generación ────────────────────────────────────────────────────

def _generate_batch(tokenizer, model, device, prompts, batch_size=16):
    """Genera respuestas para una lista de prompts. Retorna lista de strings."""
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch  = prompts[i: i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        new_ids   = ids[:, input_len:]
        outputs.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))
        if i % 500 == 0:
            print(f"  {i}/{len(prompts)} procesados...")
    return outputs


def _parse_label(generated_text):
    """Extrae 'simple' (0) o 'complejo' (1) del texto generado."""
    text = generated_text.strip().lower()
    if "complejo" in text:
        return 1
    if "simple" in text:
        return 0
    return 0  # fallback


# ── Clasificación ─────────────────────────────────────────────────

def classify_batch(classifier, texts, model_key, batch_size=16):
    """Clasifica una lista de textos. Retorna array de enteros (0 o 1)."""
    tokenizer, model, device = classifier
    prompts   = [_ZERO_SHOT_PROMPTS[model_key].format(text=t) for t in texts]
    generated = _generate_batch(tokenizer, model, device, prompts, batch_size)
    return np.array([_parse_label(g) for g in generated])


def classify_all(classifiers, texts):
    """Retorna dict {key: np.array de predicciones}."""
    return {
        key: classify_batch(clf, texts, model_key=key)
        for key, clf in classifiers.items()
    }


def show_examples(classifiers, texts):
    """Imprime tabla comparativa para una lista corta de textos."""
    headers = list(classifiers.keys())
    print("=" * 76)
    print(f"{'Texto':<45s}" + "".join(f"  {h:>12s}" for h in headers))
    print("=" * 76)
    for text in texts:
        row  = text[:42] + "..." if len(text) > 45 else text
        line = f"{row:<45s}"
        for key, clf in classifiers.items():
            tokenizer, model, device = clf
            prompt    = _ZERO_SHOT_PROMPTS[key].format(text=text)
            generated = _generate_batch(tokenizer, model, device, [prompt])[0]
            label     = "complejo" if _parse_label(generated) == 1 else "simple"
            line     += f"  {label:>12s}"
        print(line)
    print("=" * 76)


# ── Métricas ──────────────────────────────────────────────────────

def compute_llm_metrics(predictions, y_true):
    """Retorna dict {key: {'accuracy': ..., 'mae': ...}}."""
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

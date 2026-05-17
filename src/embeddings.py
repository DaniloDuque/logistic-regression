import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

BERT_MODELS = {
    "bert":    "dccuchile/bert-base-spanish-wwm-cased",
    "roberta": "bertin-project/bertin-roberta-base-spanish",
}


def load_bert_models(model_keys=None):
    """Carga tokenizadores y modelos. Retorna dict {key: (tokenizer, model)}."""
    keys = model_keys or list(BERT_MODELS.keys())
    models = {}
    for key in keys:
        model_id = BERT_MODELS[key]
        print(f"Cargando {key} ({model_id})...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        model.eval()
        models[key] = (tokenizer, model)
    return models


def get_embeddings_batch(texts, model, tokenizer, batch_size=32, device="cpu"):
    """
    Calcula embeddings del token CLS para una lista de textos.
    Retorna array numpy de forma (N, 768).
    """
    model = model.to(device)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.base_model(**inputs)

        # Token CLS → forma (batch_size, 768)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())

    return torch.cat(all_embeddings, dim=0).numpy()  # (N, 768)

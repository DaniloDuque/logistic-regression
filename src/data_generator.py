import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID   = "saul1917/FEINA"
FILENAME  = "FEINA_1.xlsx"

COL_SIMPLE  = "simple_text"
COL_COMPLEX = "complex_text"

def load_feina(verbose=True):
    """
    Descarga y carga el dataset FEINA desde Hugging Face.

    Retorna
    -------
    df         : pd.DataFrame  — dataframe completo
    texts      : list[str]     — textos simples + complejos concatenados
    labels     : list[int]     — 0 = simple, 1 = complejo
    """
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset"
    )

    df = pd.read_excel(file_path)

    if verbose:
        print(f"Shape:   {df.shape}")
        print(f"Columnas: {df.columns.tolist()}")
        print(df.head(3))

    # Validar que las columnas esperadas existan
    for col in (COL_SIMPLE, COL_COMPLEX):
        if col not in df.columns:
            raise ValueError(
                f"Columna '{col}' no encontrada. "
                f"Columnas disponibles: {df.columns.tolist()}"
            )

    texts  = df[COL_SIMPLE].tolist() + df[COL_COMPLEX].tolist()
    labels = [0] * len(df) + [1] * len(df)

    return df, texts, labels

def generate_data(separable=True, n_samples=500, random_state=42):
    std = 1.0 if separable else 4.0
    X, y = make_blobs(n_samples=n_samples, centers=2, 
                      cluster_std=std, random_state=random_state)
    
    # Agregar columna de bias (unos)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    
    # Convertir a tensores
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)
    return X_train, X_test, y_train, y_test
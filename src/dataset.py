import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID  = "saul1917/FEINA"
FILENAME = "FEINA_1.xlsx"

COL_COMPLEX = "Segment"    # texto original (complejo)
COL_SIMPLE  = "Proposal"   # texto simplificado
COL_LABEL   = "lex"        # etiqueta de complejidad léxica (0/1)


def load_feina(verbose=True):
    """
    Descarga y carga el dataset FEINA desde Hugging Face.

    Retorna
    -------
    df         : pd.DataFrame  — dataframe completo
    texts      : list[str]     — textos (Segment + Proposal concatenados)
    labels     : list[int]     — 1 = complejo (Segment), 0 = simple (Proposal)
    """
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset"
    )

    df = pd.read_excel(file_path)
    df = df.dropna(subset=[COL_COMPLEX, COL_SIMPLE])

    if verbose:
        print(f"Shape:    {df.shape}")
        print(f"Columnas: {df.columns.tolist()}")
        print(df[[COL_COMPLEX, COL_SIMPLE, COL_LABEL]].head(3))

    texts  = df[COL_COMPLEX].tolist() + df[COL_SIMPLE].tolist()
    labels = [1] * len(df) + [0] * len(df)

    return df, texts, labels
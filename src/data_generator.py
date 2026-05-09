import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

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
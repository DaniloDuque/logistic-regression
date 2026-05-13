import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.svg_export import save_pdf


def plot_results(model, X: torch.Tensor, y: torch.Tensor,
                 train_errors: list, test_errors: list, title: str,
                 output_path: str = None, device=None):
    """
    Genera una figura con 2 subgráficas:
      - Izquierda: curvas de error de entrenamiento Y prueba (MAE por iteración)
      - Derecha:   diagrama de dispersión + frontera de decisión

    Pasos para graficar la frontera de decisión:
      1. Crear una malla que cubra el espacio de características
      2. Agregar columna de bias (unos) para coincidir con el formato de entrada del modelo
      3. Ejecutar model.predict() sobre cada punto de la malla
      4. Reformar las predicciones a la forma de la malla
      5. Usar contourf para sombrear cada región y contour para dibujar la línea frontera

    Si se indica output_path, guarda la figura como PDF en esa ruta.
    """
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # --- Curvas de error de entrenamiento y prueba ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(train_errors, color='steelblue', linewidth=1.5, label='MAE Entrenamiento')
    ax1.plot(test_errors,  color='tomato',    linewidth=1.5, label='MAE Prueba', linestyle='--')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('MAE')
    ax1.set_title('Curva de error: Entrenamiento vs Prueba')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Frontera de decisión ---
    ax2 = fig.add_subplot(gs[1])

    # X[:, 0] es el bias, X[:, 1] y X[:, 2] son las características
    x1_vals = X[:, 1].cpu().numpy()
    x2_vals = X[:, 2].cpu().numpy()
    labels  = y.squeeze().cpu().numpy()

    # Paso 1: crear malla
    margin = 1.0
    xx, yy = np.meshgrid(
        np.linspace(x1_vals.min() - margin, x1_vals.max() + margin, 300),
        np.linspace(x2_vals.min() - margin, x2_vals.max() + margin, 300)
    )

    # Paso 2: agregar columna de bias
    grid = np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)

    # Pasos 3 y 4: predecir y reformar
    Z = model.predict(grid_tensor).cpu().numpy().reshape(xx.shape)

    # Paso 5: graficar
    ax2.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu')
    ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)
    scatter = ax2.scatter(x1_vals, x2_vals, c=labels,
                          cmap='RdBu', edgecolors='k', linewidths=0.5, s=40)
    ax2.set_xlabel('Característica 1')
    ax2.set_ylabel('Característica 2')
    ax2.set_title('Datos + Frontera de decisión')
    plt.colorbar(scatter, ax=ax2, label='Clase')

    plt.tight_layout()

    if output_path:
        save_pdf(fig, output_path)

    plt.show()

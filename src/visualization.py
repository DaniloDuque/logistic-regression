import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_results(model, X: torch.Tensor, y: torch.Tensor, errors: list, title: str):
    """
    Generates a figure with 2 subplots:
      - Left:  training error curve (MAE per iteration)
      - Right: scatter plot + decision boundary

    Steps to plot the decision boundary:
      1. Create a mesh grid covering the feature space
      2. Add a bias column (ones) to match the model's input format
      3. Run model.predict() over every point in the grid
      4. Reshape predictions to the grid shape
      5. Use contourf to shade each region and contour to draw the boundary line
    """
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # --- Training error curve ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(errors, color='steelblue', linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MAE')
    ax1.set_title('Training error curve')
    ax1.grid(True, alpha=0.3)

    # --- Decision boundary ---
    ax2 = fig.add_subplot(gs[1])

    # X[:, 0] is bias, X[:, 1] and X[:, 2] are the features
    x1_vals = X[:, 1].numpy()
    x2_vals = X[:, 2].numpy()
    labels  = y.squeeze().numpy()

    # Step 1: create mesh grid
    margin = 1.0
    xx, yy = np.meshgrid(
        np.linspace(x1_vals.min() - margin, x1_vals.max() + margin, 300),
        np.linspace(x2_vals.min() - margin, x2_vals.max() + margin, 300)
    )

    # Step 2: add bias column
    grid = np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    # Step 3 & 4: predict and reshape
    Z = model.predict(grid_tensor).numpy().reshape(xx.shape)

    # Step 5: plot
    ax2.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu')
    ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5)
    scatter = ax2.scatter(x1_vals, x2_vals, c=labels,
                          cmap='RdBu', edgecolors='k', linewidths=0.5, s=40)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Data + Decision boundary')
    plt.colorbar(scatter, ax=ax2, label='Class')

    plt.tight_layout()
    plt.show()
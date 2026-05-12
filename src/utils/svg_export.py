import io
import matplotlib.pyplot as plt

def fig_to_svg(fig: plt.Figure) -> bytes:
    """Convierte una figura de matplotlib a bytes SVG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    return buf.read()

def save_svg(fig: plt.Figure, path: str) -> None:
    """Guarda una figura de matplotlib como archivo SVG en disco."""
    fig.savefig(path, format='svg', bbox_inches='tight')


def fig_to_svg_base64(fig: plt.Figure) -> str:
    """Convierte una figura a SVG codificado en base64 (útil para notebooks/HTML)."""
    import base64
    return base64.b64encode(fig_to_svg(fig)).decode('utf-8')


def fig_to_pdf(fig: plt.Figure) -> bytes:
    """Convierte una figura de matplotlib a bytes PDF."""
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    return buf.read()


def save_pdf(fig: plt.Figure, path: str) -> None:
    """Guarda una figura de matplotlib como archivo PDF en disco."""
    fig.savefig(path, format='pdf', bbox_inches='tight')

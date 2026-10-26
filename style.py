"""Shared look for every figure: dark theme, one palette, glow helper."""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

BG = "#0d1117"
PANEL = "#161b22"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"

MEAS = "#7d8fa9"    # raw sensor data
TRUE = "#e6edf3"    # ground truth
EST = "#ffb454"     # Kalman estimate (the hero colour)
ACC = "#39d0d8"     # gain K
BAD = "#ff6b81"     # errors / innovation
GOOD = "#56d364"    # baseline / success


def apply():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.5,
        "grid.linewidth": 0.6,
        "text.color": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Inter", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": FG,
        "legend.framealpha": 0.95,
    })


def glow(color, lw=2.4):
    """Path effects that give a line a soft neon glow."""
    return [pe.Stroke(linewidth=lw * 3, foreground=color, alpha=0.18),
            pe.Stroke(linewidth=lw * 1.8, foreground=color, alpha=0.25),
            pe.Normal()]

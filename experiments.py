"""Run the experiment suite. Writes results/ (CSV, markdown table, figures).

Usage:  python experiments.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from kalman import kalman_1d, rmse, simulate

# ---------- settings ----------
N = 100
Q_SIM = 1e-4               # true process noise variance used to generate data
R_SIM = 0.5 ** 2           # true measurement noise variance
SEEDS = range(20)          # each config is repeated over these seeds
WARMUP = 20                # steps skipped for the "steady-state" RMSE
SHOWCASE_SEED = 42         # seed used for the single-run figures

OUT = Path("results")
OUT.mkdir(exist_ok=True)
style.apply()

# Filter configs. Q / R here are what the FILTER believes.
CONFIGS = [
    dict(name="Baseline",    note="filter Q, R match the data",
         Q=1e-4, R=R_SIM, signal="random_walk"),
    dict(name="Q too high",  note="filter Q = 1e-2  (100x too large)",
         Q=1e-2, R=R_SIM, signal="random_walk"),
    dict(name="Q too low",   note="filter Q = 1e-6  (100x too small)",
         Q=1e-6, R=R_SIM, signal="random_walk"),
    dict(name="R too small", note="filter R = R/4  (over-trusts sensor)",
         Q=1e-4, R=R_SIM / 4, signal="random_walk"),
    dict(name="R too large", note="filter R = 4R  (under-trusts sensor)",
         Q=1e-4, R=R_SIM * 4, signal="random_walk"),
    dict(name="Sine truth",  note="state is a sine, model assumes constant",
         Q=1e-4, R=R_SIM, signal="sine"),
]


def run_one(cfg, seed):
    x_true, z = simulate(N, Q_SIM, R_SIM, seed=seed, signal=cfg["signal"])
    x_hat, P, K = kalman_1d(z, cfg["Q"], cfg["R"])
    return x_true, z, x_hat, P, K


def draw_signal(ax, x_true, z, x_hat, P, labels=False):
    """Measurements + truth + glowing estimate + filter's own ±2σ band."""
    band = 2 * np.sqrt(P)
    ax.fill_between(range(N), x_hat - band, x_hat + band,
                    color=style.EST, alpha=0.13, lw=0,
                    label="±2σ (filter's own uncertainty)" if labels else None)
    ax.scatter(range(N), z, s=14, color=style.MEAS, alpha=0.6, lw=0, zorder=2,
               label="noisy measurements" if labels else None)
    ax.plot(x_true, color=style.TRUE, ls=(0, (4, 3)), lw=1.3, zorder=3,
            label="true state" if labels else None)
    ax.plot(x_hat, color=style.EST, lw=2.4, zorder=4,
            path_effects=style.glow(style.EST),
            label="Kalman estimate" if labels else None)


# ---------- 1. multi-seed experiment table ----------
rows = []
for cfg in CONFIGS:
    for seed in SEEDS:
        x_true, z, x_hat, P, K = run_one(cfg, seed)
        rows.append(dict(
            experiment=cfg["name"], seed=seed,
            Q_filt=cfg["Q"], R_filt=cfg["R"],
            rmse_meas=rmse(z, x_true),
            rmse_filt=rmse(x_hat, x_true),
            rmse_filt_steady=rmse(x_hat[WARMUP:], x_true[WARMUP:]),
            final_K=K[-1],
        ))

raw = pd.DataFrame(rows)
raw.to_csv(OUT / "results_raw.csv", index=False)

summary = (
    raw.groupby("experiment", sort=False)
    .agg(Q_filt=("Q_filt", "first"), R_filt=("R_filt", "first"),
         rmse_meas=("rmse_meas", "mean"),
         rmse_filt=("rmse_filt", "mean"), rmse_filt_std=("rmse_filt", "std"),
         rmse_steady=("rmse_filt_steady", "mean"),
         final_K=("final_K", "mean"))
    .reset_index()
)
# round metrics only: rounding Q_filt/R_filt turned 1e-6 into 0
metric_cols = ["rmse_meas", "rmse_filt", "rmse_filt_std", "rmse_steady", "final_K"]
summary[metric_cols] = summary[metric_cols].round(4)
summary.to_csv(OUT / "results_summary.csv", index=False)
(OUT / "results_summary.md").write_text(summary.to_markdown(index=False))
print(summary.to_string(index=False))

# ---------- 2. hero image (LinkedIn-sized, 1200x627) ----------
x_true, z, x_hat, P, K = run_one(CONFIGS[0], SHOWCASE_SEED)
r_meas, r_filt = rmse(z, x_true), rmse(x_hat, x_true)

fig, ax = plt.subplots(figsize=(12, 6.27))
fig.subplots_adjust(left=0.07, right=0.97, top=0.70, bottom=0.12)
draw_signal(ax, x_true, z, x_hat, P, labels=True)
ax.set_ylim(3.7, 6.6)
ax.set_xlim(0, N - 1)
ax.set_xlabel("time step")
ax.legend(loc="upper right", fontsize=9, ncol=2)
fig.text(0.07, 0.90, "1D Kalman Filter", fontsize=28, fontweight="bold")
fig.text(0.07, 0.83, "Noisy sensor in, clean estimate out.",
         fontsize=14, color=style.MUTED)
fig.text(0.97, 0.90, f"{r_meas:.2f}  →  {r_filt:.2f}", fontsize=28,
         fontweight="bold", color=style.EST, ha="right")
fig.text(0.97, 0.83, f"RMSE, ~{r_meas / r_filt:.0f}x lower error",
         fontsize=14, color=style.MUTED, ha="right")
fig.savefig(OUT / "fig_hero.png", dpi=100)
plt.close(fig)

# ---------- 3. 2x3 comparison figure (single seed) ----------
base_rmse = None
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.985, top=0.83, bottom=0.07,
                    hspace=0.40, wspace=0.07)
for i, (ax, cfg) in enumerate(zip(axes.ravel(), CONFIGS)):
    x_true, z, x_hat, P, K = run_one(cfg, SHOWCASE_SEED)
    r = rmse(x_hat, x_true)
    base_rmse = r if i == 0 else base_rmse
    draw_signal(ax, x_true, z, x_hat, P, labels=(i == 0))
    ax.set_ylim(3.7, 6.6)
    ax.set_xlim(0, N - 1)
    ax.set_title(cfg["name"], loc="left", fontsize=14, fontweight="bold",
                 pad=24)
    ax.text(0, 1.03, cfg["note"], transform=ax.transAxes, fontsize=9.5,
            color=style.MUTED)
    badge_color = style.GOOD if r <= base_rmse * 1.15 else style.BAD
    ax.text(0.97, 0.06, f"RMSE {r:.3f}", transform=ax.transAxes, ha="right",
            fontsize=11, fontweight="bold", color=badge_color,
            bbox=dict(boxstyle="round,pad=0.35", fc=style.PANEL,
                      ec=badge_color, lw=1))
    if i >= 3:
        ax.set_xlabel("time step")
axes[0, 0].legend(loc="upper right", fontsize=8)
fig.text(0.05, 0.955, "What happens when the filter is wrong",
         fontsize=22, fontweight="bold")
fig.text(0.05, 0.912, "Same data, same seed. Only the filter's beliefs change.",
         fontsize=12, color=style.MUTED)
fig.savefig(OUT / "fig_comparison.png", dpi=170)
plt.close(fig)

# ---------- 4. RMSE bar chart (mean ± std over seeds) ----------
fig, ax = plt.subplots(figsize=(10, 5.6))
fig.subplots_adjust(left=0.17, right=0.95, top=0.80, bottom=0.13)
names = summary["experiment"].tolist()
colors = [style.GOOD if n == "Baseline" else style.BAD if "Sine" in n
          else style.EST for n in names]
ypos = np.arange(len(names))
ax.barh(ypos, summary["rmse_filt"], xerr=summary["rmse_filt_std"],
        color=colors, height=0.6, alpha=0.9,
        error_kw=dict(ecolor=style.FG, capsize=3, lw=1, alpha=0.7))
for y, v in zip(ypos, summary["rmse_filt"]):
    ax.text(v + 0.045, y, f"{v:.3f}", va="center", fontsize=11,
            fontweight="bold")
meas = summary["rmse_meas"].mean()
ax.axvline(meas, color=style.MEAS, ls=(0, (4, 3)), lw=1.5)
ax.text(meas - 0.01, len(names) - 0.45, f"raw sensor  {meas:.2f}",
        color=style.MEAS, ha="right", va="bottom", fontsize=10)
ax.set_yticks(ypos)
ax.set_yticklabels(names, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel("RMSE vs true state (lower is better)")
ax.grid(axis="y", visible=False)
fig.text(0.04, 0.92, "How much does a wrong guess cost?", fontsize=20,
         fontweight="bold")
fig.text(0.04, 0.86, f"Mean ± std over {len(SEEDS)} seeds",
         fontsize=12, color=style.MUTED)
fig.savefig(OUT / "fig_rmse.png", dpi=170)
plt.close(fig)

# ---------- 5. Kalman gain vs Q ----------
fig, ax = plt.subplots(figsize=(10, 5.6))
fig.subplots_adjust(left=0.09, right=0.83, top=0.80, bottom=0.13)
for Q, c in zip((1e-6, 1e-4, 1e-2), (style.ACC, style.EST, style.BAD)):
    cfg = dict(Q=Q, R=R_SIM, signal="random_walk")
    _, _, _, _, K = run_one(cfg, SHOWCASE_SEED)
    ax.plot(K, color=c, lw=2.4, path_effects=style.glow(c))
    ax.text(N - 1 + 2, K[-1], f"Q = {Q:g}\nK ≈ {K[-1]:.3f}", color=c,
            va="center", fontsize=11, fontweight="bold")
ax.set_yscale("log")
ax.set_xlim(0, N - 1)
ax.set_xlabel("time step")
ax.set_ylabel("Kalman gain K  (log scale)")
fig.text(0.04, 0.92, "Bigger Q, bigger gain", fontsize=20, fontweight="bold")
fig.text(0.04, 0.86,
         "If the state may drift, the filter trusts the sensor more.",
         fontsize=12, color=style.MUTED)
fig.savefig(OUT / "fig_gain_vs_Q.png", dpi=170)
plt.close(fig)

print(f"\nSaved results and figures to {OUT.resolve()}")

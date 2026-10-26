"""Animated GIF of the filter processing measurements one at a time.

Top panel:    measurements, estimate, ±2σ band, and the innovation
              (gap between prediction and the new measurement).
Bottom panel: Kalman gain K decaying as the filter gains confidence.

Usage:  python animate.py      ->  results/kalman.gif
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

import style
from kalman import kalman_1d, simulate

N, Q, R = 100, 1e-4, 0.5 ** 2
HOLD = 25                                   # extra frames to pause on the end
x_true, z = simulate(N, Q, R, seed=42)
x_hat, P, K = kalman_1d(z, Q, R)
band = 2 * np.sqrt(P)
x_pred = np.concatenate(([np.nan], x_hat[:-1]))   # prediction = last estimate

Path("results").mkdir(exist_ok=True)
style.apply()

fig = plt.figure(figsize=(10, 6.4))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.05], hspace=0.22,
                      left=0.08, right=0.97, top=0.82, bottom=0.10)
ax = fig.add_subplot(gs[0])
axk = fig.add_subplot(gs[1], sharex=ax)

fig.text(0.08, 0.93, "Kalman filter, one measurement at a time",
         fontsize=20, fontweight="bold")
fig.text(0.08, 0.875, "Watch the gain K fall as the filter gains confidence.",
         fontsize=11.5, color=style.MUTED)

# --- top panel
ax.set_xlim(0, N - 1)
ax.set_ylim(3.7, 6.6)
ax.tick_params(labelbottom=False)
ax.plot(x_true, color=style.TRUE, ls=(0, (4, 3)), lw=1.2, alpha=0.55)
meas_line, = ax.plot([], [], "o", color=style.MEAS, ms=4.5, alpha=0.65, mew=0)
est_line, = ax.plot([], [], color=style.EST, lw=2.6,
                    path_effects=style.glow(style.EST))
innov, = ax.plot([], [], color=style.BAD, lw=2.4, solid_capstyle="round")
cur_meas, = ax.plot([], [], "o", mfc="none", mec=style.BAD, ms=11, mew=2)
cur_est, = ax.plot([], [], "o", color=style.EST, ms=8, mec=style.BG, mew=1.5,
                   zorder=6)
band_artist = [None]

handles = [
    Line2D([], [], color=style.TRUE, ls=(0, (4, 3)), lw=1.2, label="true state"),
    Line2D([], [], marker="o", color=style.MEAS, lw=0, ms=5,
           label="measurements"),
    Line2D([], [], color=style.EST, lw=2.6, label="Kalman estimate"),
    Line2D([], [], color=style.BAD, lw=2.4,
           label="innovation (measurement − prediction)"),
]
ax.legend(handles=handles, loc="upper right", fontsize=8.5, ncol=2)

# --- bottom panel
axk.set_ylim(0, 1.05)
axk.set_ylabel("gain K")
axk.set_xlabel("time step")
axk.plot(K, color=style.GRID, lw=1.5)
k_line, = axk.plot([], [], color=style.ACC, lw=2.4,
                   path_effects=style.glow(style.ACC))
k_dot, = axk.plot([], [], "o", color=style.ACC, ms=8, mec=style.BG, mew=1.5)
k_txt = axk.text(0.98, 0.78, "", transform=axk.transAxes, ha="right",
                 fontsize=12, fontweight="bold", color=style.ACC)


def update(frame):
    k = min(frame, N - 1)
    xs = np.arange(k + 1)

    meas_line.set_data(xs[:-1], z[:k])           # past measurements
    est_line.set_data(xs, x_hat[: k + 1])
    cur_meas.set_data([k], [z[k]])
    cur_est.set_data([k], [x_hat[k]])

    if k >= 1:
        innov.set_data([k, k], [x_pred[k], z[k]])
    else:
        innov.set_data([], [])

    if band_artist[0] is not None:
        band_artist[0].remove()
    band_artist[0] = ax.fill_between(
        xs, x_hat[: k + 1] - band[: k + 1], x_hat[: k + 1] + band[: k + 1],
        color=style.EST, alpha=0.14, lw=0)

    k_line.set_data(xs, K[: k + 1])
    k_dot.set_data([k], [K[k]])
    k_txt.set_text(f"step {k:3d}    K = {K[k]:.3f}")
    return meas_line, est_line, innov, cur_meas, cur_est, k_line, k_dot, k_txt


anim = FuncAnimation(fig, update, frames=N + HOLD, interval=50, blit=False)
anim.save("results/kalman.gif", writer=PillowWriter(fps=20), dpi=100)
print("Saved results/kalman.gif")

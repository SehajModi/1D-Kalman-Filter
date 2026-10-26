"""Core pieces: data simulation, the 1D Kalman filter, and RMSE."""
import numpy as np


def simulate(n, Q_sim, R_sim, seed=42, x0=5.0, signal="random_walk"):
    """Generate a true state and noisy measurements.

    Q_sim, R_sim are VARIANCES; rng.normal wants standard deviations, hence sqrt.
    signal: "random_walk" (state drifts with variance Q_sim) or "sine".
    """
    rng = np.random.default_rng(seed)

    if signal == "random_walk":
        steps = rng.normal(0, np.sqrt(Q_sim), n - 1)
        x_true = x0 + np.concatenate(([0.0], np.cumsum(steps)))
    elif signal == "sine":
        x_true = x0 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 50)
    else:
        raise ValueError(f"unknown signal: {signal}")

    z = x_true + rng.normal(0, np.sqrt(R_sim), n)
    return x_true, z


def kalman_1d(z, Q, R, x0=0.0, P0=10.0):
    """Scalar Kalman filter for the model x_k = x_{k-1} + w, z_k = x_k + v.

    Q, R are the variances the FILTER believes (they can differ from the
    simulation's). Returns estimates, variance history, gain history.
    """
    n = len(z)
    x_hat = np.empty(n)
    P_log = np.empty(n)
    K_log = np.empty(n)

    x_est, P_est = x0, P0
    for k in range(n):
        # predict
        x_pred = x_est
        P_pred = P_est + Q

        # gain
        K = P_pred / (P_pred + R)

        # update
        innovation = z[k] - x_pred
        x_est = x_pred + K * innovation
        P_est = (1 - K) * P_pred

        x_hat[k], P_log[k], K_log[k] = x_est, P_est, K

    return x_hat, P_log, K_log


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

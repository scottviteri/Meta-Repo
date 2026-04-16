"""Blahut-Arimoto algorithm for channel capacity."""

import numpy as np


def blahut_arimoto(
    channel: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> float:
    """Compute channel capacity in nats using Blahut-Arimoto.

    Args:
        channel: (num_inputs, num_outputs) matrix where channel[x, y] = P(y|x).
                 Rows must sum to 1.
        max_iter: maximum BA iterations.
        tol: convergence threshold on capacity bounds gap.

    Returns:
        Channel capacity in nats.
    """
    n_in, n_out = channel.shape
    if n_in == 0 or n_out == 0:
        return 0.0

    row_sums = channel.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"Channel rows must sum to 1, got sums: {row_sums}")

    # Remove zero-probability outputs
    used_outputs = channel.sum(axis=0) > 0
    if not used_outputs.any():
        return 0.0
    ch = channel[:, used_outputs]
    n_in, n_out = ch.shape

    if n_in == 1:
        return 0.0

    q = np.ones(n_in) / n_in

    for _ in range(max_iter):
        r = q @ ch  # marginal output distribution
        r = np.maximum(r, 1e-300)

        # c[x, y] = P(y|x) * log(P(y|x) / r(y))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(ch) - np.log(r[np.newaxis, :])
            log_ratio = np.where(ch > 0, log_ratio, 0.0)

        # phi[x] = exp(sum_y P(y|x) log(P(y|x)/r(y)))
        phi = np.exp(np.sum(ch * log_ratio, axis=1))

        c_lower = np.log(q @ phi)
        c_upper = np.log(np.max(phi))

        if c_upper - c_lower < tol:
            return float(c_lower)

        q = q * phi
        q /= q.sum()

    return float(np.log(q @ phi))

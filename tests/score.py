import numpy as np


def score_param_similarity(
    fitted: np.ndarray,
    true: np.ndarray,
    scale: np.ndarray | float | None = None,
    fitted_sigma: np.ndarray | float | None = None,
    n_boot: int = 2000,
    seed: int = 42,
):
    fitted = np.asarray(fitted, dtype=float)
    true = np.asarray(true, dtype=float)
    if true.ndim == 0:
        true = np.full_like(fitted, float(true))

    if scale is None:
        scale = np.maximum(np.abs(true), 1e-16)
    scale = np.asarray(scale, dtype=float)
    if scale.ndim == 0:
        scale = np.full_like(fitted, float(scale))

    if fitted_sigma is not None:
        fitted_sigma = np.asarray(fitted_sigma, dtype=float)
        if fitted_sigma.ndim == 0:
            fitted_sigma = np.full_like(fitted, float(fitted_sigma))
        denom = np.sqrt(scale**2 + fitted_sigma**2)
    else:
        denom = scale

    r = (fitted - true) / denom
    s = float(np.sqrt(np.mean(r**2)))
    rng = np.random.default_rng(seed)
    m = r.size
    bs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        rb = r[idx]
        bs[b] = np.sqrt(np.mean(rb**2))
    s_se = float(bs.std(ddof=1))

    return {"sigma": r, "score": s, "score_se": s_se}

import numpy as np


def score_param_similarity(
    fitted: np.ndarray, true: np.ndarray, scale: np.ndarray = None
):
    if scale is None:
        scale = np.maximum(np.abs(true), 1e-16)  # in case of 0
    diff = (fitted - true) / scale
    per_param = diff
    total_score = np.sqrt(np.mean(diff**2))
    return {
        "sigma": per_param,
        "score": total_score,
    }

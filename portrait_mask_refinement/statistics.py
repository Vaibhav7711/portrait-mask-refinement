"""Paired uncertainty summaries, with foregrounds as the resampling unit."""
import numpy as np


def paired_interval(values, seed=17):
    """Input must contain one paired delta per foreground, not per corruption."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError("Cannot bootstrap an empty comparison")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (5000, len(values)), replace=True).mean(1)
    return {"mean": float(values.mean()), "foregrounds": len(values),
            "bootstrap_95ci": np.quantile(means, [.025, .975]).tolist(),
            "improved_foregrounds": int((values < 0).sum())}

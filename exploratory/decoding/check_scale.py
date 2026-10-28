"""Check for dominant ("massive"/outlier) activation dimensions and whether standardizing matters (DRAFT).

For one token position, and every saved layer, this reports
  1. how concentrated the across-sample variance is in a few dimensions
     (share of total variance in the top 5 dimensions, and max/median std), and
  2. cross-validated ridge R^2 for one belief label with RAW features
     (centered only) versus STANDARDIZED features (z-scored per dimension on
     the training fold), pooled over all checkpoints.

Needs only numpy/pandas.  Ridge is solved in dual form (n << d) and its
penalty is chosen from a grid scaled to each layer's feature scale.  CV folds
group whole conversations together.  This is a scale diagnostic, not the final
analysis: it does not include the baselines in NOTES.md.

Usage:
  python exploratory/decoding/check_scale.py --run runs/NAME --position prompt_end --target belief_core
Needs <run>/activations/*.pt (extract_activations.py) and <run>/labels.csv (make_labels.py).
"""

import numpy as np
import torch
import pandas as pd
import argparse
from pathlib import Path

MULTIPLIERS = 10.0 ** np.arange(-4, 3.5, 0.5)


def load_samples(run: Path, position: str) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    """Return activations [n, n_layers, d] at one position, with phase/conversation keys."""

    arrays, rows, layers = [], [], None
    for path in sorted((run / "activations").glob("*.pt")):
        saved = torch.load(path)
        layers = saved["layers"]
        index = saved["positions"].index(position)
        for c, checkpoint in enumerate(saved["checkpoints"]):
            arrays.append(saved["activations"][c, index].float().numpy())
            rows.append({"cell_id": saved["cell_id"], "replicate_index": saved["replicate_index"], "phase": checkpoint["phase"]})
    return np.stack(arrays), pd.DataFrame(rows), layers


def variance_summary(x: np.ndarray, top_k: int = 5) -> dict[str, object]:
    """Describe how much a few dimensions dominate across-sample variance."""

    variance = x.var(axis=0)
    std = np.sqrt(variance)
    order = np.argsort(variance)[::-1]
    return {
        "top_dims": order[:top_k].tolist(),
        "top_share": float(variance[order[:top_k]].sum() / variance.sum()),
        "max_over_median_std": float(std.max() / max(np.median(std), 1e-12)),
    }


def grouped_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray, standardize: bool, n_folds: int = 5, seed: int = 0) -> tuple[float, float]:
    """Cross-validated R^2 with whole conversations held out together.

    Returns (best R^2, penalty multiplier).  The penalty is alpha = multiplier x
    mean eigenvalue of the training kernel, so the grid is comparable between
    raw and standardized features.  The best multiplier is picked on the same
    held-out folds, so R^2 is slightly optimistic (one tuned hyperparameter).
    """

    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    fold_of = {group: i % n_folds for i, group in enumerate(unique)}
    folds = np.array([fold_of[group] for group in groups])
    predictions = np.zeros((len(MULTIPLIERS), len(y)))
    for fold in range(n_folds):
        train, test = folds != fold, folds == fold
        x_train, x_test = x[train], x[test]
        mean = x_train.mean(axis=0)
        if standardize:
            std = x_train.std(axis=0) + 1e-6
            x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
        else:
            x_train, x_test = x_train - mean, x_test - mean
        y_mean = y[train].mean()
        values, vectors = np.linalg.eigh(x_train @ x_train.T)
        values = np.clip(values, 0.0, None)
        projected = vectors.T @ (y[train] - y_mean)
        cross = (x_test @ x_train.T) @ vectors
        for i, multiplier in enumerate(MULTIPLIERS):
            predictions[i, test] = cross @ (projected / (values + multiplier * values.mean())) + y_mean
    total = np.sum((y - y.mean()) ** 2)
    scores = 1.0 - np.sum((y - predictions) ** 2, axis=1) / total
    best = int(np.argmax(scores))
    return float(scores[best]), float(MULTIPLIERS[best])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--position", default="prompt_end")
    parser.add_argument("--target", default="belief_core")
    args = parser.parse_args()

    x, keys, layers = load_samples(args.run, args.position)
    labels = pd.read_csv(args.run / "labels.csv")[["cell_id", "replicate_index", "phase", args.target]]
    merged = keys.merge(labels, on=["cell_id", "replicate_index", "phase"], how="left")
    keep = merged[args.target].notna().to_numpy()
    x, merged = x[keep], merged[keep].reset_index(drop=True)
    y = merged[args.target].to_numpy(dtype=float)
    groups = (merged["cell_id"] + "__" + merged["replicate_index"].astype(str)).to_numpy()
    print(f"{len(y)} samples from {len(np.unique(groups))} conversations, position={args.position}, target={args.target}, label std={y.std():.2f}")

    table = []
    for i, layer in enumerate(layers):
        summary = variance_summary(x[:, i])
        raw = grouped_r2(x[:, i], y, groups, standardize=False)
        standardized = grouped_r2(x[:, i], y, groups, standardize=True)
        table.append(
            {
                "layer": layer,
                "top5_var_share": round(summary["top_share"], 3),
                "max/median_std": round(summary["max_over_median_std"], 1),
                "top_dims": summary["top_dims"],
                "R2_raw": round(raw[0], 3),
                "mult_raw": raw[1],
                "R2_standardized": round(standardized[0], 3),
                "mult_std": standardized[1],
            }
        )
    result = pd.DataFrame(table)
    print(result.to_string(index=False))
    result.to_csv(args.run / f"scale_check_{args.position}_{args.target}.csv", index=False)


if __name__ == "__main__":
    main()

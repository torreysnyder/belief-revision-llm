"""Check for dominant ("massive"/outlier) activation dimensions and whether standardizing matters (DRAFT).

For one token position, and every saved layer, this reports
  1. how concentrated the across-sample variance is in a few dimensions
     (share of total variance in the top 5 dimensions, and max/median std), and
  2. cross-validated ridge R^2 for one belief label with RAW features
     (centered only) versus STANDARDIZED features (z-scored per dimension on
     the training fold), pooled over all checkpoints.

Needs only numpy/pandas.  Ridge is solved in dual form (n << d) and its
penalty is picked per training fold by closed-form leave-one-out.  CV folds
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

ALPHAS = 10.0 ** np.arange(-2, 8, 1.0)


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


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Dual-form ridge with the penalty chosen by closed-form leave-one-out."""

    mean, y_mean = x_train.mean(axis=0), y_train.mean()
    xc, yc = x_train - mean, y_train - y_mean
    values, vectors = np.linalg.eigh(xc @ xc.T)
    values = np.clip(values, 0.0, None)
    projected = vectors.T @ yc
    best_alpha, best_error = ALPHAS[0], np.inf
    for alpha in ALPHAS:
        shrink = values / (values + alpha)
        leverage = (vectors**2) @ shrink
        residual = (yc - vectors @ (shrink * projected)) / np.clip(1.0 - leverage, 1e-8, None)
        error = float(np.mean(residual**2))
        if error < best_error:
            best_alpha, best_error = alpha, error
    dual = vectors @ (projected / (values + best_alpha))
    return (x_test - mean) @ (xc.T @ dual) + y_mean


def grouped_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray, standardize: bool, n_folds: int = 5, seed: int = 0) -> float:
    """Cross-validated R^2 with whole conversations held out together."""

    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    fold_of = {group: i % n_folds for i, group in enumerate(unique)}
    folds = np.array([fold_of[group] for group in groups])
    predictions = np.full(len(y), np.nan)
    for fold in range(n_folds):
        train, test = folds != fold, folds == fold
        x_train, x_test = x[train], x[test]
        if standardize:
            mean, std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
            x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
        predictions[test] = ridge_predict(x_train, y[train], x_test)
    return float(1.0 - np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2))


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
        table.append(
            {
                "layer": layer,
                "top5_var_share": round(summary["top_share"], 3),
                "max/median_std": round(summary["max_over_median_std"], 1),
                "top_dims": summary["top_dims"],
                "R2_raw": round(grouped_r2(x[:, i], y, groups, standardize=False), 3),
                "R2_standardized": round(grouped_r2(x[:, i], y, groups, standardize=True), 3),
            }
        )
    result = pd.DataFrame(table)
    print(result.to_string(index=False))
    result.to_csv(args.run / f"scale_check_{args.position}_{args.target}.csv", index=False)


if __name__ == "__main__":
    main()

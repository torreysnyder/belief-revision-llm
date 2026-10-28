"""Linear decoding of belief from residual-stream activations and a text baseline (DRAFT).

Targets, for each of core / A1 / A2 (auxiliaries are canonical, see make_labels.py):
  score    the evaluator's 0-100 belief score          (regression, R^2)
  absdev   |score - 50|, how far from neutral          (regression, R^2)
  endorse  1 if score > 50, 0 if < 50, dropped at 50   (AUC of the ridge output)

With --labels endorsement the targets come instead from endorsement_probe.py
(<run>/endorsement/scores.csv), per claim (core / A1 / A2), all regression:
  dE        endorsement change from the claim-only baseline, AB format, mean of both A/B mappings
  dEA, dEB  the same for the A_endorse and B_endorse mappings separately
  dU        withhold-vs-commit change, ABC format, mean of both mappings
  dqW       change in the probability of "Withhold judgment", ABC format, mean of both mappings
Within one vignette the baseline is a constant per claim, so a change score
decodes exactly like the raw score; the baseline matters once vignettes are pooled.

Feature sets: the four positions from extract_activations.py, plus
"probe:<claim>:<mapping>" = the hidden state at the probe's "Answer:" token
(AB format) from endorsement_probe.py.  That one is a positive control: it is
read just before the model answers the probe itself.

Analyses, for every feature set (each layer of each token position, plus text):
  pooled_model  one ridge trained on all checkpoints, scored overall and per test checkpoint
  within        train and test on the same checkpoint (held-out conversations)
  transfer      train on one checkpoint, test on another (the same conversation is never in both)
  baseline      phase-mean and cell-mean predictors (no activations)
Each result also reports the largest score over PERMUTATIONS label shuffles
(labels shuffled among samples of the same checkpoint): a chance ceiling.

Ridge is solved in dual form.  The penalty is MULTIPLIER x the mean kernel
eigenvalue; the best multiplier is chosen on the held-out folds, so scores are
slightly optimistic (the shuffles get the same choice, which keeps the
comparison fair).  Folds hold out whole conversations (or whole cells with
--group-by cell).  Needs only numpy/pandas.

Usage:
  python exploratory/decoding/decode.py --run runs/NAME --text-only
  python exploratory/decoding/decode.py --run runs/NAME --positions prompt_end,reply_last
  python exploratory/decoding/decode.py --run runs/NAME --positions prompt_end --layers 14 --out-name test
  python exploratory/decoding/decode.py --run runs/NAME --labels endorsement --positions prompt_end,reply_eot,probe:core:A_endorse --out-name endorsement
Writes <run>/decoding/<out-name>.csv (default name: results).
"""

import re
import json
import numpy as np
import torch
import pandas as pd
import argparse
from check_scale import load_samples
from extract_activations import checkpoint_turns
from pathlib import Path

PHASES = ["intro", "core_challenge_1", "core_challenge_2", "closing"]
BELIEFS = {"core": "belief_core", "A1": "belief_A1_canonical", "A2": "belief_A2_canonical"}
MULTIPLIERS = 10.0 ** np.arange(-3, 2.5, 0.5)
PERMUTATIONS = 10
N_FOLDS = 5


def build_targets(labels: pd.DataFrame) -> pd.DataFrame:
    """Add score / absdev / endorse columns for each belief."""

    out = {}
    for name, column in BELIEFS.items():
        score = labels[column].astype(float)
        out[f"{name}_score"] = score
        out[f"{name}_absdev"] = (score - 50).abs()
        out[f"{name}_endorse"] = np.where(score > 50, 1.0, np.where(score < 50, 0.0, np.nan))
    return pd.DataFrame(out, index=labels.index)


ENDORSEMENT_TARGETS = {"dE": ("AB", None, "dE"), "dEA": ("AB", "A_endorse", "dE"), "dEB": ("AB", "B_endorse", "dE"), "dU": ("ABC", None, "dU"), "dqW": ("ABC", None, "dq_withhold")}


def endorsement_labels(run: Path) -> pd.DataFrame:
    """One row per (conversation, checkpoint) with <claim>_<target> columns from the endorsement probe."""

    scores = pd.read_csv(run / "endorsement" / "scores.csv")
    keys = ["cell_id", "replicate_index", "phase"]
    columns = []
    for target, (fmt, mapping, value) in ENDORSEMENT_TARGETS.items():
        subset = scores[(scores["format"] == fmt) & ((scores["mapping"] == mapping) if mapping else True)]
        table = subset.groupby([*keys, "claim"])[value].mean().unstack("claim")
        table.columns = [f"{claim}_{target}" for claim in table.columns]
        columns.append(table)
    return pd.concat(columns, axis=1).reset_index()


def target_columns(labels: pd.DataFrame, source: str) -> list[str]:
    """Columns that must be present for a sample to be used."""

    if source == "evaluator":
        return list(BELIEFS.values())
    return [c for c in labels.columns if c.split("_")[0] in BELIEFS]


def align(keys: pd.DataFrame, labels: pd.DataFrame, required: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Merge activation-file keys with labels; returns the merged table and a keep mask."""

    merged = keys.merge(labels, on=["cell_id", "replicate_index", "phase"], how="left")
    keep = merged[required].notna().all(axis=1).to_numpy()
    return merged[keep].reset_index(drop=True), keep


def load_probe_samples(run: Path, claim: str, mapping: str) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    """Hidden state at the probe's "Answer:" token for one claim and mapping: [n, n_layers, d]."""

    claims, mappings = ["core", "A1", "A2"], ["A_endorse", "B_endorse"]
    arrays, rows, layers = [], [], None
    for path in sorted((run / "endorsement").glob("*__rep*.pt")):
        saved = torch.load(path)
        layers = saved["layers"]
        for c, phase in enumerate(saved["phases"]):
            arrays.append(saved["hidden"][c, claims.index(claim), mappings.index(mapping)].float().numpy())
            rows.append({"cell_id": saved["cell_id"], "replicate_index": saved["replicate_index"], "phase": phase})
    return np.stack(arrays), pd.DataFrame(rows), layers


def fold_ids(groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Assign each group to one of N_FOLDS folds, deterministically."""

    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    mapping = {group: i % N_FOLDS for i, group in enumerate(unique)}
    return np.array([mapping[group] for group in groups])


def fit_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, standardize: bool) -> np.ndarray:
    """Dual ridge for every multiplier and target column; returns [n_multipliers, n_test, n_targets]."""

    a, b = x[train], x[test]
    mean = a.mean(axis=0)
    if standardize:
        std = a.std(axis=0) + 1e-6
        a, b = (a - mean) / std, (b - mean) / std
    else:
        a, b = a - mean, b - mean
    y_train = y[train]
    y_mean = np.nanmean(y_train, axis=0)
    y_centered = np.nan_to_num(y_train - y_mean)  # missing labels count as the mean
    values, vectors = np.linalg.eigh(a @ a.T)
    values = np.clip(values, 0.0, None)
    projected = vectors.T @ y_centered
    cross = (b @ a.T) @ vectors
    scale = values.mean()
    return np.stack([cross @ (projected / (values + m * scale)[:, None]) + y_mean for m in MULTIPLIERS])


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based area under the ROC curve; NaN if a class is missing."""

    positive = y == 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics_for_column(y: np.ndarray, p: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (primary metric, Pearson r) for each multiplier; p is [n_multipliers, n]."""

    ok = ~np.isnan(y)
    y, p = y[ok], p[:, ok]
    if kind == "endorse":
        return np.array([auc(y, row) for row in p]), np.full(len(p), np.nan)
    if len(y) < 3:
        return np.full(len(p), np.nan), np.full(len(p), np.nan)
    sse = ((y - p) ** 2).sum(axis=1)
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - sse / sst
    pc, yc = p - p.mean(axis=1, keepdims=True), y - y.mean()
    r = (pc @ yc) / np.sqrt((pc**2).sum(axis=1) * (yc**2).sum() + 1e-12)
    return r2, r


def evaluate(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, names: list[str], context: dict[str, object]) -> list[dict[str, object]]:
    """Score every target (and its shuffles) on the masked rows; pick the best multiplier per target."""

    rows = []
    n_targets = len(names)
    for t, name in enumerate(names):
        kind = name.split("_")[1]
        primary, corr = metrics_for_column(y[mask, t], pred[:, mask, t], kind)
        if np.all(np.isnan(primary)):
            continue
        best = int(np.nanargmax(primary))
        null = []
        for p in range(1, PERMUTATIONS + 1):
            shuffled, _ = metrics_for_column(y[mask, t + p * n_targets], pred[:, mask, t + p * n_targets], kind)
            null.append(np.nanmax(shuffled) if not np.all(np.isnan(shuffled)) else np.nan)
        rows.append(
            {
                **context,
                "target": name,
                "metric": "auc" if kind == "endorse" else "r2",
                "value": primary[best],
                "r": corr[best],
                "multiplier": MULTIPLIERS[best],
                "null_max": np.nanmax(null) if null else np.nan,
                "n_test": int(mask.sum()),
                "n_pos": int(np.nansum(y[mask, t])) if kind == "endorse" else np.nan,
            }
        )
    return rows


def analyze(feature_set: str, x: np.ndarray, y: np.ndarray, phase: np.ndarray, folds: np.ndarray, names: list[str], standardize: bool, extra: dict[str, object]) -> list[dict[str, object]]:
    """Run pooled, within-checkpoint and cross-checkpoint decoding for one feature set."""

    n, t = y.shape
    pooled = np.zeros((len(MULTIPLIERS), n, t))
    transfer = [np.zeros((len(MULTIPLIERS), n, t)) for _ in PHASES]
    for fold in range(N_FOLDS):
        test = folds == fold
        pooled[:, test] = fit_predict(x, y, ~test, test, standardize)
        for i in range(len(PHASES)):
            transfer[i][:, test] = fit_predict(x, y, (~test) & (phase == i), test, standardize)

    rows = []
    base = {"feature_set": feature_set, **extra}
    everyone = np.ones(n, dtype=bool)
    rows += evaluate(pooled, y, everyone, names, {**base, "analysis": "pooled_model", "train": "all", "test": "all"})
    for j, test_name in enumerate(PHASES):
        rows += evaluate(pooled, y, phase == j, names, {**base, "analysis": "pooled_model", "train": "all", "test": test_name})
        for i, train_name in enumerate(PHASES):
            analysis = "within" if i == j else "transfer"
            rows += evaluate(transfer[i], y, phase == j, names, {**base, "analysis": analysis, "train": train_name, "test": test_name})
    return rows


def baseline_rows(y: np.ndarray, phase: np.ndarray, cells: np.ndarray, folds: np.ndarray, names: list[str]) -> list[dict[str, object]]:
    """Predict from training-fold phase means or cell means only (no activations)."""

    rows = []
    for label, groups in (("baseline/phase_mean", phase), ("baseline/cell_mean", cells)):
        pred = np.zeros((1, *y.shape))
        for fold in range(N_FOLDS):
            test = folds == fold
            frame = pd.DataFrame(y[~test])
            frame["g"] = groups[~test]
            means = frame.groupby("g").mean()
            overall = np.nanmean(y[~test], axis=0)
            mapped = means.reindex(groups[test]).to_numpy()
            pred[0, test] = np.where(np.isnan(mapped), overall, mapped)
        rows += evaluate(pred, y, np.ones(len(y), dtype=bool), names, {"feature_set": label, "analysis": "pooled_model", "train": "all", "test": "all", "position": "-", "layer": -1})
    return rows


def tfidf(documents: list[str], min_df: int = 5) -> np.ndarray:
    """Dense sublinear TF-IDF over unigrams and bigrams, L2-normalized."""

    token_lists = []
    for document in documents:
        words = re.findall(r"[a-z']+", document.lower())
        token_lists.append(words + [f"{a} {b}" for a, b in zip(words, words[1:])])
    document_frequency: dict[str, int] = {}
    for tokens in token_lists:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    vocabulary = {token: i for i, token in enumerate(sorted(t for t, c in document_frequency.items() if c >= min_df))}
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float32)
    for row, tokens in enumerate(token_lists):
        for token in tokens:
            column = vocabulary.get(token)
            if column is not None:
                matrix[row, column] += 1
    matrix = np.where(matrix > 0, 1 + np.log(np.maximum(matrix, 1)), 0).astype(np.float32)
    idf = np.log((1 + len(documents)) / (1 + np.array([document_frequency[t] for t in vocabulary]))) + 1
    matrix *= idf.astype(np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def text_variants(run: Path, meta: pd.DataFrame) -> dict[str, list[str]]:
    """Per-sample text up to each checkpoint: whole history, assistant turns only, and the final reply only."""

    histories = {}
    with (run / "dialogues.jsonl").open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            histories[(record["cell_id"], record["replicate_index"])] = record["dialogue_history"]
    turn_of = {c["phase"]: c["global_turn_index"] for c in checkpoint_turns()}
    variants: dict[str, list[str]] = {"all_text": [], "assistant_text": [], "last_reply_text": []}
    for cell, replicate, phase in zip(meta["cell_id"], meta["replicate_index"], meta["phase"]):
        g = turn_of[phase]
        turns = [t for t in histories[(cell, replicate)] if t["global_turn_index"] <= g]
        variants["all_text"].append(" \n ".join(t["text"] for t in turns))
        variants["assistant_text"].append(" \n ".join(t["text"] for t in turns if t["speaker"] == "assistant"))
        variants["last_reply_text"].append(" ".join(t["text"] for t in turns if t["speaker"] == "assistant" and t["global_turn_index"] == g))
    return variants


def build_label_matrix(targets: pd.DataFrame, phase: np.ndarray, seed: int = 1) -> tuple[np.ndarray, list[str]]:
    """Stack the target columns, then PERMUTATIONS shuffled copies (shuffled within checkpoint)."""

    names = list(targets.columns)
    y = targets.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    blocks = [y]
    for _ in range(PERMUTATIONS):
        order = np.arange(len(y))
        for j in range(len(PHASES)):
            idx = np.where(phase == j)[0]
            order[idx] = rng.permutation(idx)
        blocks.append(y[order])
    return np.hstack(blocks), names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--positions", default="prompt_end,reply_last")
    parser.add_argument("--layers", default=None, help="Comma-separated layer numbers (default: all).")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--group-by", choices=("conversation", "cell"), default="conversation")
    parser.add_argument("--labels", choices=("evaluator", "endorsement"), default="evaluator", help="Target source: API evaluator scores (labels.csv) or the endorsement probe (endorsement/scores.csv).")
    parser.add_argument("--out-name", default="results")
    args = parser.parse_args()

    labels = pd.read_csv(args.run / "labels.csv") if args.labels == "evaluator" else endorsement_labels(args.run)
    required = target_columns(labels, args.labels)
    out_dir = args.run / "decoding"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.out_name}.csv"
    rows: list[dict[str, object]] = []

    def prepare(keys: pd.DataFrame):
        merged, keep = align(keys, labels, required)
        phase = merged["phase"].map({p: i for i, p in enumerate(PHASES)}).to_numpy()
        group = merged["cell_id"] if args.group_by == "cell" else merged["cell_id"] + "__" + merged["replicate_index"].astype(str)
        targets = build_targets(merged) if args.labels == "evaluator" else merged[required]
        y, names = build_label_matrix(targets, phase)
        return merged, keep, phase, fold_ids(group.to_numpy()), y, names

    merged = None
    for position in [] if args.text_only else args.positions.split(","):
        if position.startswith("probe:"):
            _, claim, mapping = position.split(":")
            x, keys, layers = load_probe_samples(args.run, claim, mapping)
        else:
            x, keys, layers = load_samples(args.run, position)
        merged, keep, phase, folds, y, names = prepare(keys)
        x = x[keep]
        wanted = [int(v) for v in args.layers.split(",")] if args.layers else layers
        for index, layer in enumerate(layers):
            if layer in wanted:
                rows += analyze(f"{position}/L{layer}", x[:, index], y, phase, folds, names, True, {"position": position, "layer": layer})
                print(f"{position} layer {layer} done", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    if merged is None:  # text-only: use every labelled checkpoint row
        merged, _, phase, folds, y, names = prepare(labels[["cell_id", "replicate_index", "phase"]])
    for variant, documents in text_variants(args.run, merged).items():
        rows += analyze(f"text/{variant}", tfidf(documents), y, phase, folds, names, False, {"position": "text", "layer": -1})
        print(f"{variant} done", flush=True)
    rows += baseline_rows(y, phase, merged["cell_id"].to_numpy(), folds, names)
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    print(f"{len(result)} rows -> {out_path}")


if __name__ == "__main__":
    main()

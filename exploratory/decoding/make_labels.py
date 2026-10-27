"""Build the decoding label table from a run's results.csv (DRAFT).

One row per probed checkpoint.  Auxiliary scores are mapped from the DISPLAY
labels the evaluator saw back to CANONICAL ones (A1 = methodological,
A2 = contextual), because aux_order_condition="reversed" swaps which auxiliary
is shown as A1.  Without this the aux decoders mix two different beliefs.

Usage: python exploratory/decoding/make_labels.py --run runs/NAME
"""

import pandas as pd
import argparse
from pathlib import Path


def build_labels(results: pd.DataFrame) -> pd.DataFrame:
    """Keep probed rows and re-key the auxiliary beliefs by canonical id."""

    probed = results[results["probe_ran_after_turn"]].copy()
    for canonical in ("A1", "A2"):
        display = probed[f"canonical_to_display_{canonical}"]
        for measure in ("belief", "confidence_belief"):
            probed[f"{measure}_{canonical}_canonical"] = [row[f"{measure}_{shown}"] for (_, row), shown in zip(probed.iterrows(), display)]
    return probed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    labels = build_labels(pd.read_csv(args.run / "results.csv"))
    labels.to_csv(args.run / "labels.csv", index=False)
    print(f"{len(labels)} checkpoint rows -> {args.run / 'labels.csv'}")
    print(labels.groupby("phase")[["belief_core", "belief_A1_canonical", "belief_A2_canonical"]].agg(["count", "mean", "std"]).round(2))


if __name__ == "__main__":
    main()

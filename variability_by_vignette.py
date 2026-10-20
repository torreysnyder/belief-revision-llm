"""
Variability of probed belief scores (belief_core, belief_A1, belief_A2) by
vignette and probe phase.

Reuses load_and_prepare from analysis_full_crossed.py so the probe filtering,
phase ordering, and categorical handling stay consistent with the main
analysis.

Two views are reported, since "across replications" is ambiguous once other
design factors are crossed with vignette:

- by-vignette-phase: spread across *every* conversation using that vignette
  at that probe phase (i.e. pooled across all styles/trust/goals/anchors/
  order conditions). This is total variability attributable to the vignette,
  including variability the design deliberately introduces.
- within-cell: spread across just the replicate_index draws inside a single
  fully-fixed primary cell (same vignette + style + trust + goal + anchor
  level). This isolates run-to-run noise (LLM stochasticity plus whatever the
  replicate-index-driven counterbalancing contributes) from the between-
  condition variability above, then averages that up to one
  typical-replicate-noise number per vignette/phase.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_full_crossed import (
    load_and_prepare, INPUT_CSV, PROBE_PHASES,
    set_pub_style, PUB_PALETTE, PUB_LABELS,
)

OUTPUT_DIR = Path("output_full_crossed/paper_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BELIEF_COLS = ["belief_core", "belief_A1", "belief_A2"]
MEASURE_COLOR = {"belief_core": PUB_PALETTE["core"], "belief_A1": PUB_PALETTE["A1"], "belief_A2": PUB_PALETTE["A2"]}
MEASURE_LABEL = {"belief_core": PUB_LABELS["core"], "belief_A1": PUB_LABELS["A1"], "belief_A2": PUB_LABELS["A2"]}


def variability_by_vignette_phase(probe):
    rows = []
    for (vignette_id, phase), g in probe.groupby(["vignette_id", "phase"], observed=True):
        for col in BELIEF_COLS:
            vals = g[col].dropna()
            if len(vals) < 2:
                continue
            rows.append({
                "vignette_id": vignette_id,
                "phase": phase,
                "measure": col,
                "n": len(vals),
                "mean": vals.mean(),
                "sd": vals.std(ddof=1),
                "variance": vals.var(ddof=1),
                "min": vals.min(),
                "max": vals.max(),
                "range": vals.max() - vals.min(),
                "iqr": vals.quantile(0.75) - vals.quantile(0.25),
                "cv": vals.std(ddof=1) / vals.mean() if vals.mean() else np.nan,
            })
    return pd.DataFrame(rows)


def variability_within_cell(probe):
    rows = []
    for (cell_id, vignette_id, phase), g in probe.groupby(
        ["cell_id", "vignette_id", "phase"], observed=True
    ):
        for col in BELIEF_COLS:
            vals = g[col].dropna()
            if len(vals) < 2:
                continue
            rows.append({
                "cell_id": cell_id,
                "vignette_id": vignette_id,
                "phase": phase,
                "measure": col,
                "n_replicates": len(vals),
                "mean": vals.mean(),
                "sd": vals.std(ddof=1),
            })
    cell_level = pd.DataFrame(rows)

    summary = (
        cell_level.groupby(["vignette_id", "phase", "measure"], observed=True)
        .agg(
            n_cells=("sd", "size"),
            mean_within_cell_sd=("sd", "mean"),
            median_within_cell_sd=("sd", "median"),
        )
        .reset_index()
    )
    return cell_level, summary


def plot_variability_by_vignette(by_vignette, fname="fig9_variability_by_vignette"):
    set_pub_style()

    vignette_ids = sorted(by_vignette["vignette_id"].unique())
    phases = [p for p in PROBE_PHASES if p in set(by_vignette["phase"])]
    x = np.arange(len(vignette_ids))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, phase in zip(axes.flat, phases):
        d = by_vignette[by_vignette["phase"] == phase]
        for i, measure in enumerate(BELIEF_COLS):
            sub = d[d["measure"] == measure].set_index("vignette_id").reindex(vignette_ids)
            ax.bar(
                x + (i - 1) * width, sub["sd"].values, width=width,
                color=MEASURE_COLOR[measure], label=MEASURE_LABEL[measure],
            )
        ax.set_title(str(phase), loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(vignette_ids, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("SD of belief (0-100 scale)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        "Variability of probed belief scores by vignette and phase",
        x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.08,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{fname}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)


def main(input_csv=INPUT_CSV):
    _, probe = load_and_prepare(input_csv)

    by_vignette = variability_by_vignette_phase(probe)
    by_vignette.to_csv(OUTPUT_DIR / "variability_by_vignette_phase.csv", index=False)
    print(f"Wrote variability_by_vignette_phase.csv ({len(by_vignette)} rows)")

    cell_level, within_cell_summary = variability_within_cell(probe)
    cell_level.to_csv(OUTPUT_DIR / "variability_within_cell_raw.csv", index=False)
    within_cell_summary.to_csv(OUTPUT_DIR / "variability_within_cell_summary.csv", index=False)
    print(f"Wrote variability_within_cell_summary.csv ({len(within_cell_summary)} rows)")

    plot_variability_by_vignette(by_vignette)
    print("Wrote fig9_variability_by_vignette.png/.pdf")


if __name__ == "__main__":
    main()

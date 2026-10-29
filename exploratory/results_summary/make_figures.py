"""Figures and tables for the work-in-progress results summary (RESULTS_SUMMARY.md).

Reads existing outputs only; nothing is re-run:
  output_full_crossed/paper_v3/            early gpt-4.1-mini behavioral experiment (summary files)
  runs/politics03-llama8b-full/labels.csv  Llama-8B politics_03 evaluator scores
  runs/politics03-llama8b-full/decoding/   evaluator-score decoding and probe self-decoding
  runs/politics03-llama8b-full/endorsement/scores.csv  endorsement probe scores

Usage (from the repo root):
  uv run --no-sync python exploratory/results_summary/make_figures.py
Writes PNGs to exploratory/results_summary/figures/ and prints the markdown tables.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EARLY = ROOT / "output_full_crossed" / "paper_v3"
RUN = ROOT / "runs" / "politics03-llama8b-full"
FIGURES = Path(__file__).resolve().parent / "figures"

PHASES = ["intro", "core_challenge_1", "core_challenge_2", "closing"]
PHASE_LABELS = ["intro", "challenge 1", "challenge 2", "closing"]
CLAIMS = ["core", "A1", "A2"]
COLOR = {"core": "#2a78d6", "A1": "#eb6834", "A2": "#1baf7a"}
GRAY, LIGHT_GRAY, INK, MUTED = "#8a8985", "#d9d8d4", "#0b0b0b", "#52514e"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": LIGHT_GRAY,
            "axes.labelcolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#ecebe8",
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def markdown(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


# ---------------------------------------------------------------- 1. early gpt-4.1-mini experiment


def early_experiment() -> str:
    by_phase = pd.read_csv(EARLY / "variability_by_vignette_phase.csv")
    within = pd.read_csv(EARLY / "variability_within_cell_summary.csv")
    measure = {"belief_core": "core", "belief_A1": "A1", "belief_A2": "A2"}
    by_phase["claim"] = by_phase["measure"].map(measure)
    within["claim"] = within["measure"].map(measure)
    vignettes = sorted(by_phase["vignette_id"].unique())

    # Figure 1: mean belief at intro vs closing per vignette (revision), one panel per claim.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
    for ax, claim in zip(axes, CLAIMS):
        sub = by_phase[by_phase["claim"] == claim].set_index(["vignette_id", "phase"])["mean"]
        for y, vignette in enumerate(vignettes):
            start, end = sub[(vignette, "intro")], sub[(vignette, "closing")]
            ax.plot([start, end], [y, y], color=LIGHT_GRAY, linewidth=2, zorder=1)
            ax.scatter(start, y, s=28, color="white", edgecolor=GRAY, linewidth=1.5, zorder=2)
            ax.scatter(end, y, s=36, color=COLOR[claim], edgecolor="white", linewidth=1, zorder=3)
        ax.axvline(50, color=GRAY, linewidth=0.8, linestyle=":")
        ax.set_xlim(0, 100)
        ax.set_title(claim)
        ax.set_xlabel("mean belief (0-100)")
        ax.set_yticks(range(len(vignettes)), vignettes)
        ax.invert_yaxis()
    handles = [axes[0].scatter([], [], s=28, color="white", edgecolor=GRAY, linewidth=1.5, label="intro"), axes[0].scatter([], [], s=36, color=GRAY, label="closing")]
    fig.suptitle("Early experiment (gpt-4.1-mini): mean belief at intro and closing, by vignette", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=8)
    save(fig, "fig1_early_revision_by_vignette")

    # Figure 2: total SD (all conversations) vs typical within-cell SD (replicates of one cell), averaged over phases.
    total = by_phase.groupby(["vignette_id", "claim"])["sd"].mean()
    noise = within.groupby(["vignette_id", "claim"])["mean_within_cell_sd"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
    for ax, claim in zip(axes, CLAIMS):
        for y, vignette in enumerate(vignettes):
            t, w = total[(vignette, claim)], noise[(vignette, claim)]
            ax.plot([w, t], [y, y], color=LIGHT_GRAY, linewidth=2, zorder=1)
            ax.scatter(w, y, s=28, color="white", edgecolor=GRAY, linewidth=1.5, zorder=2)
            ax.scatter(t, y, s=36, color=COLOR[claim], edgecolor="white", linewidth=1, zorder=3)
        ax.set_xlim(0, 40)
        ax.set_title(claim)
        ax.set_xlabel("SD of belief (0-100 scale)")
        ax.set_yticks(range(len(vignettes)), vignettes)
        ax.invert_yaxis()
    handles = [axes[0].scatter([], [], s=28, color="white", edgecolor=GRAY, linewidth=1.5, label="within one cell (replicates)"), axes[0].scatter([], [], s=36, color=GRAY, label="across all conversations")]
    fig.suptitle("Early experiment: total vs replicate-to-replicate variability (mean over phases)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=2, fontsize=8)
    save(fig, "fig2_early_variability")

    # Table: core belief per vignette.
    core = by_phase[by_phase["claim"] == "core"].pivot(index="vignette_id", columns="phase", values="mean")
    rows = []
    for vignette in vignettes:
        rows.append(
            {
                "vignette": vignette,
                "core intro": f"{core.loc[vignette, 'intro']:.1f}",
                "core closing": f"{core.loc[vignette, 'closing']:.1f}",
                "change": f"{core.loc[vignette, 'closing'] - core.loc[vignette, 'intro']:+.1f}",
                "core SD (all)": f"{total[(vignette, 'core')]:.1f}",
                "core SD (within cell)": f"{noise[(vignette, 'core')]:.1f}",
                "A1 SD (all)": f"{total[(vignette, 'A1')]:.1f}",
                "A2 SD (all)": f"{total[(vignette, 'A2')]:.1f}",
            }
        )
    return markdown(pd.DataFrame(rows))


# ---------------------------------------------------------------- 2. Llama politics_03 behavior


def llama_behavior() -> str:
    labels = pd.read_csv(RUN / "labels.csv")
    columns = {"core": "belief_core", "A1": "belief_A1_canonical", "A2": "belief_A2_canonical"}
    early = pd.read_csv(EARLY / "variability_by_vignette_phase.csv")
    early_core = early[(early["vignette_id"] == "politics_03") & (early["measure"] == "belief_core")].set_index("phase")["mean"].reindex(PHASES)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(PHASES))
    for claim, column in columns.items():
        stats = labels.groupby("phase")[column].agg(["mean", "std", "count"]).reindex(PHASES)
        ci = 1.96 * stats["std"] / np.sqrt(stats["count"])
        left.fill_between(x, stats["mean"] - ci, stats["mean"] + ci, color=COLOR[claim], alpha=0.15, linewidth=0)
        left.plot(x, stats["mean"], color=COLOR[claim], marker="o", markersize=5, label=f"{claim} (Llama-8B)")
    left.plot(x, early_core, color=GRAY, linestyle="--", marker="o", markersize=4, label="core (gpt-4.1-mini, early run)")
    left.axhline(50, color=GRAY, linewidth=0.8, linestyle=":")
    left.set_xticks(x, PHASE_LABELS)
    left.set_ylim(0, 100)
    left.set_ylabel("evaluator belief score (mean, 95% CI)")
    left.set_title("Mean belief by checkpoint")
    left.legend(fontsize=8, loc="upper left", ncol=2)

    wide = labels.pivot_table(index=["cell_id", "replicate_index"], columns="phase", values="belief_core")
    change = (wide["closing"] - wide["intro"]).dropna()
    bins = np.arange(-47.5, 50, 5)
    right.hist(change, bins=bins, color=COLOR["core"], edgecolor="white", linewidth=1)
    right.axvline(0, color=GRAY, linewidth=0.8, linestyle=":")
    right.set_xlim(-47.5, 47.5)
    right.set_xlabel("core belief change, closing minus intro")
    right.set_ylabel("conversations")
    right.set_title(f"Per-conversation core revision (n={len(change)})")
    fig.suptitle("Llama-3.1-8B-Instruct, politics_03 (864 conversations)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_llama_politics03_behavior")

    rows = []
    for claim, column in columns.items():
        stats = labels.groupby("phase")[column].agg(["mean", "std"]).reindex(PHASES)
        rows.append({"claim": claim, **{label: f"{m:.1f} ({s:.1f})" for label, m, s in zip(PHASE_LABELS, stats["mean"], stats["std"])}})
    table = markdown(pd.DataFrame(rows))
    summary = f"Core change closing minus intro: mean {change.mean():+.1f}, SD {change.std():.1f}; {np.mean(change > 0):.0%} up, {np.mean(change < 0):.0%} down, {np.mean(change == 0):.0%} unchanged."
    return table + "\n\n" + summary


# ---------------------------------------------------------------- 3. evaluator-score decoding


def evaluator_decoding() -> str:
    d = pd.read_csv(RUN / "decoding" / "results.csv")
    pooled = d[(d["analysis"] == "pooled_model") & (d["test"] == "all")]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, claim in zip(axes, CLAIMS):
        target = pooled[pooled["target"] == f"{claim}_score"]
        for position, dash in (("prompt_end", "-"), ("reply_last", "--")):
            curve = target[target["position"] == position].sort_values("layer")
            ax.plot(curve["layer"], curve["value"], color=COLOR[claim], linestyle=dash, label=position.replace("_", " "))
        text = target[target["feature_set"] == "text/all_text"]["value"].iloc[0]
        ceiling = target[target["position"] == "prompt_end"]["null_max"].max()
        ax.axhline(text, color=INK, linewidth=1.2, linestyle=":", label="text (TF-IDF)")
        ax.axhline(ceiling, color=GRAY, linewidth=0.8, label="shuffle ceiling")
        ax.set_title(f"{claim} score")
        ax.set_xlabel("layer")
    axes[0].set_ylabel("cross-validated R²")
    handles, names = axes[0].get_legend_handles_labels()
    fig.suptitle("Decoding the evaluator's belief score from Llama activations (politics_03)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(handles, ["prompt end", "reply last", "text (TF-IDF)", "shuffle ceiling"], loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=4, fontsize=8)
    save(fig, "fig4_evaluator_decoding_layers")

    # Figure 5: cross-checkpoint transfer (Pearson r), core score, best activation layer vs text.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, feature, title in ((axes[0], "prompt_end/L14", "activations (prompt end, L14)"), (axes[1], "text/all_text", "text (TF-IDF)")):
        sub = d[(d["feature_set"] == feature) & (d["target"] == "core_score") & d["analysis"].isin(["within", "transfer"])]
        matrix = sub.pivot(index="train", columns="test", values="r").reindex(index=PHASES, columns=PHASES)
        image = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=0.7)
        for i in range(len(PHASES)):
            for j in range(len(PHASES)):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.45 else INK)
        ax.set_xticks(range(4), PHASE_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(4), PHASE_LABELS)
        ax.set_xlabel("test checkpoint")
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_ylabel("train checkpoint")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson r (held-out conversations)")
    fig.suptitle("Core score: does a decoder trained at one checkpoint work at another?", x=0.01, ha="left", fontweight="bold")
    save(fig, "fig5_evaluator_decoding_transfer")

    rows = []
    for claim in CLAIMS:
        for kind, metric in (("score", "R²"), ("absdev", "R²"), ("endorse", "AUC")):
            target = pooled[pooled["target"] == f"{claim}_{kind}"]
            cells = {"target": f"{claim} {kind}", "metric": metric}
            for position in ("prompt_end", "reply_last"):
                best = target[target["position"] == position].sort_values("value").iloc[-1]
                cells[position.replace("_", " ")] = f"{best['value']:.2f} (L{int(best['layer'])})"
            cells["text"] = f"{target[target['feature_set'] == 'text/all_text']['value'].iloc[0]:.2f}"
            cells["phase mean"] = f"{target[target['feature_set'] == 'baseline/phase_mean']['value'].iloc[0]:.2f}"
            cells["shuffle ceiling"] = f"{target[target['position'] == 'prompt_end']['null_max'].max():.2f}"
            rows.append(cells)
    return markdown(pd.DataFrame(rows))


# ---------------------------------------------------------------- 4. endorsement probe (partial)


def endorsement_probe() -> str:
    scores = pd.read_csv(RUN / "endorsement" / "scores.csv")
    n = scores[["cell_id", "replicate_index"]].drop_duplicates().shape[0]
    ab = scores[scores["format"] == "AB"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(PHASES))
    for claim in CLAIMS:
        for mapping, dash in (("A_endorse", "-"), ("B_endorse", "--")):
            mean = ab[(ab["claim"] == claim) & (ab["mapping"] == mapping)].groupby("phase")["E"].mean().reindex(PHASES)
            left.plot(x, mean, color=COLOR[claim], linestyle=dash, marker="o", markersize=4, label=f"{claim}, {'A' if mapping == 'A_endorse' else 'B'} = Endorse")
    left.axhline(0, color=GRAY, linewidth=0.8, linestyle=":")
    left.set_xticks(x, PHASE_LABELS)
    left.set_ylabel("mean E = s(Endorse) - s(Reject)")
    left.set_title("Probe endorsement by checkpoint")
    left.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.08))

    d = pd.read_csv(RUN / "decoding" / "probe_selfdecode_partial.csv")
    pooled = d[(d["analysis"] == "pooled_model") & (d["test"] == "all")]
    for offset, claim in zip((-0.8, 0.0, 0.8), CLAIMS):
        target = pooled[pooled["target"] == f"{claim}_dE"]
        probe = target[target["position"].str.startswith("probe:")].copy()
        probe["probe_claim"] = probe["position"].str.split(":").str[1]
        own = probe[probe["probe_claim"] == claim].groupby("layer")["value"].mean()
        other = probe[probe["probe_claim"] != claim].groupby("layer")["value"].mean()
        right.plot(own.index, own.values, color=COLOR[claim], marker="o", markersize=4, label=f"{claim}: own probe state")
        right.plot(other.index, other.values, color=COLOR[claim], linestyle="--", linewidth=1.3, label=f"{claim}: other claims' states")
        text = target[target["feature_set"] == "text/all_text"]["value"].iloc[0]
        right.scatter([33.5 + offset], [text], marker="s", s=30, color=COLOR[claim], edgecolor="white", zorder=3)
    right.annotate("text", (33.5, 0.38), ha="center", fontsize=8, color=MUTED)
    right.set_xlim(-1, 35)
    right.set_ylim(0.3, 1.02)
    right.set_xlabel("layer (squares: text TF-IDF)")
    right.set_ylabel("cross-validated R² (dE)")
    right.set_title("Decoding the probe score from the probe's hidden state")
    right.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Endorsement probe, partial data ({n} of 864 conversations)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig6_endorsement_probe")
    return f"{n} conversations"


def main() -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, section in (("early", early_experiment), ("llama", llama_behavior), ("decoding", evaluator_decoding), ("probe", endorsement_probe)):
        print(f"\n## {name}\n{section()}")


if __name__ == "__main__":
    main()

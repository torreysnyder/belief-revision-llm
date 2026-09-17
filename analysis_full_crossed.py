
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

INPUT_CSV = "belief_revision_results_full_crossed.csv"
OUTPUT_DIR = Path("output_full_crossed/paper_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE_ORDER_FULL = [
    "intro",
    "core_challenge_1",
    "elaboration_1",
    "core_challenge_2",
    "elaboration_2",
    "closing",
]

PROBE_PHASES = ["intro", "core_challenge_1", "core_challenge_2", "closing"]

PUB_PALETTE = {
    "core": "#2E4057",
    "A1": "#B54A3A",
    "A2": "#4B8F8C",
    "epconf": "#6B6B6B",
}

PUB_LABELS = {
    "core": "Core claim (C)",
    "A1": "Auxiliary 1 (A1)",
    "A2": "Auxiliary 2 (A2)",
    "epconf": "Epistemic confidence",
}

# Categorical factor levels, matched to the actual values written by the
# generation script (paste.txt): STYLE_LIBRARY, TRUST_LEVELS,
# CONVERSATION_GOALS, BELIEF_ANCHOR_LEVELS, and the run-plan conditions.
CAT_ORDERS = {
    "phase": PHASE_ORDER_FULL,
    "style_id": ["terse_anxious", "reflective_informal", "skeptical_rambling"],
    "trust_in_institutions": ["low", "mixed"],
    "conversation_goal": ["validation", "explanation", "reassurance", "challenge"],
    "belief_anchor_level": ["low", "medium", "high"],
    "aux_order_condition": ["original", "reversed"],
    "core_order_condition": ["strong_then_moderate", "moderate_then_strong"],
    "elaboration_1_condition": ["ambiguous", "supporting"],
    "elaboration_2_target_condition": ["A1", "A2"],
}

FACTOR_LIST = [
    "style_id",
    "trust_in_institutions",
    "conversation_goal",
    "belief_anchor_level",
    "initial_affect",
    "aux_order_condition",
    "core_order_condition",
    "elaboration_1_condition",
    "elaboration_2_target_condition",
    "domain",
]


# --------- UTILITIES ---------

def set_pub_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def sanitize_filename(text):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(text))


def save_table(df, name):
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_text(text, name):
    path = OUTPUT_DIR / f"{name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# --------- LOAD & PREP ---------

def load_and_prepare(input_csv=INPUT_CSV):
    df = pd.read_csv(input_csv)

    numeric_cols = [
        "replicate_index", "seed",
        "turn_in_phase", "global_turn_index",
        "belief_core", "belief_A1", "belief_A2",
        "assistant_epistemic_confidence",
        "confidence_belief_core", "confidence_belief_A1", "confidence_belief_A2",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    boolish = [
        "probe_ran_after_turn", "inject_evidence", "introduced_new_auxiliary",
        "aux_gate_passed",
    ]
    for c in boolish:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            ).where(df[c].notna(), np.nan)

    for col, order in CAT_ORDERS.items():
        if col in df.columns:
            # Guard against unexpected values not in the predefined order
            # (keeps them, but places them at the end, rather than dropping rows)
            observed = [v for v in df[col].dropna().unique() if v not in order]
            full_order = order + sorted(observed)
            df[col] = pd.Categorical(df[col], categories=full_order, ordered=True)

    required = {"cell_id", "replicate_index", "phase"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns missing from CSV: {missing}")

    df["conversation_id"] = (
        df["cell_id"].astype(str)
        + "__rep_"
        + df["replicate_index"].astype("Int64").astype(str)
    )

    probe = df[df["probe_ran_after_turn"] == True].copy()
    probe["phase"] = pd.Categorical(
        probe["phase"], categories=PROBE_PHASES, ordered=True
    )
    probe = probe.dropna(subset=["phase"])
    probe = probe.sort_values(["conversation_id", "phase"])

    return df, probe


def build_sample_summary(probe):
    rows = []
    for conv_id, g in probe.groupby("conversation_id", dropna=False):
        g = g.sort_values("phase")
        first, last = g.iloc[0], g.iloc[-1]

        row = {
            "conversation_id": conv_id,
            "cell_id": first.get("cell_id"),
            "replicate_index": first.get("replicate_index"),
            "vignette_id": first.get("vignette_id"),
            "domain": first.get("domain"),
            "style_id": first.get("style_id"),
            "trust_in_institutions": first.get("trust_in_institutions"),
            "conversation_goal": first.get("conversation_goal"),
            "belief_anchor_level": first.get("belief_anchor_level"),
            "initial_affect": first.get("initial_affect"),
            "aux_order_condition": first.get("aux_order_condition"),
            "core_order_condition": first.get("core_order_condition"),
            "elaboration_1_condition": first.get("elaboration_1_condition"),
            "elaboration_2_target_condition": first.get("elaboration_2_target_condition"),
            "belief_core_start": first.get("belief_core"),
            "belief_core_end": last.get("belief_core"),
            "belief_A1_start": first.get("belief_A1"),
            "belief_A1_end": last.get("belief_A1"),
            "belief_A2_start": first.get("belief_A2"),
            "belief_A2_end": last.get("belief_A2"),
            "epconf_start": first.get("assistant_epistemic_confidence"),
            "epconf_end": last.get("assistant_epistemic_confidence"),
            "assistant_stance_final": last.get("assistant_stance"),
            "main_change_final": last.get("main_change"),
            "changed_node_final": last.get("changed_node"),
            "introduced_new_auxiliary_final": last.get("introduced_new_auxiliary"),
        }

        row["delta_core"] = (
            row["belief_core_end"] - row["belief_core_start"]
            if pd.notna(row["belief_core_end"]) and pd.notna(row["belief_core_start"])
            else np.nan
        )
        row["delta_A1"] = (
            row["belief_A1_end"] - row["belief_A1_start"]
            if pd.notna(row["belief_A1_end"]) and pd.notna(row["belief_A1_start"])
            else np.nan
        )
        row["delta_A2"] = (
            row["belief_A2_end"] - row["belief_A2_start"]
            if pd.notna(row["belief_A2_end"]) and pd.notna(row["belief_A2_start"])
            else np.nan
        )
        row["delta_epconf"] = (
            row["epconf_end"] - row["epconf_start"]
            if pd.notna(row["epconf_end"]) and pd.notna(row["epconf_start"])
            else np.nan
        )

        rows.append(row)

    sample_df = pd.DataFrame(rows)

    cats = [
        "style_id", "trust_in_institutions", "conversation_goal",
        "belief_anchor_level", "initial_affect", "aux_order_condition",
        "core_order_condition", "elaboration_1_condition",
        "elaboration_2_target_condition", "domain",
    ]
    for c in cats:
        if c in sample_df.columns:
            sample_df[c] = sample_df[c].astype("category")

    return sample_df


# --------- RQ1: CORE VS AUXILIARY CHANGE ---------

def paired_effect_summary(df, start_col, end_col, label):
    sub = df[[start_col, end_col]].dropna()
    n = len(sub)
    if n < 3:
        return {
            "measure": label, "n": n, "mean_start": np.nan, "mean_end": np.nan,
            "mean_change": np.nan, "sd_change": np.nan, "t": np.nan,
            "p": np.nan, "dz": np.nan,
        }

    diff = sub[end_col] - sub[start_col]
    t_stat, p_val = stats.ttest_rel(sub[end_col], sub[start_col], nan_policy="omit")
    sd = diff.std(ddof=1)
    dz = diff.mean() / sd if sd not in (0, np.nan) else np.nan

    return {
        "measure": label, "n": n,
        "mean_start": sub[start_col].mean(), "mean_end": sub[end_col].mean(),
        "mean_change": diff.mean(), "sd_change": sd,
        "t": t_stat, "p": p_val, "dz": dz,
    }


def rq1_core_vs_aux(sample_df):
    d = sample_df[["delta_core", "delta_A1", "delta_A2"]].dropna().copy()
    d["aux_mean_delta"] = d[["delta_A1", "delta_A2"]].mean(axis=1)
    d["aux_mean_abs"] = d[["delta_A1", "delta_A2"]].abs().mean(axis=1)

    t_signed = stats.ttest_rel(d["aux_mean_delta"], d["delta_core"], nan_policy="omit")
    t_abs = stats.ttest_rel(d["aux_mean_abs"], d["delta_core"].abs(), nan_policy="omit")

    return pd.DataFrame([{
        "n": len(d),
        "mean_delta_core": d["delta_core"].mean(),
        "mean_delta_aux_mean": d["aux_mean_delta"].mean(),
        "mean_abs_delta_core": d["delta_core"].abs().mean(),
        "mean_abs_delta_aux_mean": d["aux_mean_abs"].mean(),
        "t_signed": t_signed.statistic, "p_signed": t_signed.pvalue,
        "t_abs": t_abs.statistic, "p_abs": t_abs.pvalue,
    }])


# --------- RQ2: FACTOR EFFECTS ON CHANGE (mixed model with vignette random effect) ---------

def fit_ols_change_model(sample_df, outcome):
    formula = (
        f"{outcome} ~ C(style_id) + C(trust_in_institutions) + "
        f"C(conversation_goal) + C(belief_anchor_level) + "
        f"C(initial_affect) + C(aux_order_condition) + "
        f"C(core_order_condition) + C(elaboration_1_condition) + "
        f"C(elaboration_2_target_condition) + C(domain)"
    )
    d = sample_df.dropna(subset=[outcome]).copy()
    model = smf.ols(formula, data=d).fit(cov_type="HC3")
    return model


def fit_mixedlm_change_model(sample_df, outcome):
    """Vignette-level random intercept, since vignette_id has many levels
    and is a nuisance grouping factor rather than a factor of interest."""
    formula = (
        f"{outcome} ~ C(style_id) + C(trust_in_institutions) + "
        f"C(conversation_goal) + C(belief_anchor_level) + "
        f"C(initial_affect) + C(aux_order_condition) + "
        f"C(core_order_condition) + C(elaboration_1_condition) + "
        f"C(elaboration_2_target_condition) + C(domain)"
    )
    d = sample_df.dropna(subset=[outcome, "vignette_id"]).copy()
    model = smf.mixedlm(formula, data=d, groups=d["vignette_id"]).fit(reml=False)
    return model


def extract_ols_summary(model, outcome_name):
    coefs = model.params
    se = model.bse
    t = model.tvalues
    p = model.pvalues

    out = pd.DataFrame({
        "term": coefs.index, "coef": coefs.values, "se": se.values,
        "t": t.values, "p": p.values,
    })
    out["outcome"] = outcome_name
    if hasattr(model, "rsquared"):
        out["r2"] = model.rsquared
        out["adj_r2"] = model.rsquared_adj
    return out


# --------- RQ3: STANCE & CHANGE LOCATION ---------

def standardized_change_category(sample_df):
    d = sample_df.copy()
    d["main_change_final"] = d["main_change_final"].fillna("none")
    d["changed_node_final"] = d["changed_node_final"].fillna("none")

    def categorize(row):
        if row["main_change_final"] == "none":
            return "no_change"
        if row["main_change_final"] == "core":
            return "core"
        if row["main_change_final"] == "auxiliary":
            if row["changed_node_final"] == "A1":
                return "A1"
            if row["changed_node_final"] == "A2":
                return "A2"
            return "aux_other"
        return "source_other"

    d["change_category"] = d.apply(categorize, axis=1)
    return d


def crosstab_factor(sample_df, factor, outcome):
    d = sample_df[[factor, outcome]].dropna().copy()
    if d.empty:
        return None
    tab = pd.crosstab(d[factor], d[outcome])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    chi2, p, dof, expected = stats.chi2_contingency(tab)
    out = tab.reset_index()
    out["chi2"] = chi2
    out["p"] = p
    out["dof"] = dof
    return out


# --------- GLOBAL FIGURES ---------

def plot_trajectory(probe):
    set_pub_style()
    d = probe.copy()

    agg = (
        d.groupby("phase", observed=True)
        .agg(
            core_m=("belief_core", "mean"),
            core_se=("belief_core", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
            A1_m=("belief_A1", "mean"),
            A1_se=("belief_A1", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
            A2_m=("belief_A2", "mean"),
            A2_se=("belief_A2", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
        )
        .reindex(PROBE_PHASES)
        .reset_index()
    )

    x = np.arange(len(agg))
    phase_labels = {
        "intro": "Intro",
        "core_challenge_1": "Core-\nchallenge 1",
        "core_challenge_2": "Core-\nchallenge 2",
        "closing": "Closing",
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for key, m_col, se_col in [
        ("core", "core_m", "core_se"),
        ("A1", "A1_m", "A1_se"),
        ("A2", "A2_m", "A2_se"),
    ]:
        y = agg[m_col].values
        se = agg[se_col].values
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se,
                         color=PUB_PALETTE[key], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=PUB_PALETTE[key], linewidth=2.4, marker="o", markersize=5.5)
        ax.annotate(PUB_LABELS[key], xy=(x[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=10.5,
                    fontweight="bold", color=PUB_PALETTE[key])

    ax.set_xticks(x)
    ax.set_xticklabels([phase_labels[p] for p in agg["phase"]])
    ax.set_ylabel("Inferred belief (0-100)")
    ax.set_xlabel("Probe phase")
    ax.set_title("Belief trajectories across probe phases", loc="left")
    ax.set_xlim(-0.25, len(x) - 0.25 + 1.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_trajectory.png")
    fig.savefig(OUTPUT_DIR / "fig1_trajectory.pdf")
    plt.close(fig)


def plot_stance_distribution(sample_df):
    set_pub_style()
    d = sample_df.copy()
    d["assistant_stance_final"] = d["assistant_stance_final"].fillna("unclear")
    counts = (
        d["assistant_stance_final"].value_counts(normalize=True)
        .reindex(["reinforces_user", "mixed", "challenges_user", "unclear"])
        .dropna() * 100
    )

    labels = {
        "reinforces_user": "Reinforces\nuser", "mixed": "Mixed",
        "challenges_user": "Challenges\nuser", "unclear": "Unclear",
    }
    colors = ["#B54A3A", "#D9A44A", "#4B8F8C", "#999999"]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    x = np.arange(len(counts))
    bars = ax.bar(x, counts.values, color=colors[: len(counts)])
    for bar, val in zip(bars, counts.values):
        ax.annotate(f"{val:.0f}%", (bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in counts.index])
    ax.set_ylabel("Share of conversations (%)")
    ax.set_ylim(0, max(counts.values) * 1.25)
    ax.set_title("Final assistant stance", loc="left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_stance_distribution.png")
    fig.savefig(OUTPUT_DIR / "fig2_stance_distribution.pdf")
    plt.close(fig)


def plot_change_composition(sample_df):
    set_pub_style()
    d = standardized_change_category(sample_df)
    counts = d["change_category"].value_counts(normalize=True).sort_values(ascending=True) * 100

    label_map = {
        "no_change": "No change", "core": "Core (C)", "A1": "Auxiliary 1",
        "A2": "Auxiliary 2", "aux_other": "Auxiliary (other)", "source_other": "Source / other",
    }
    color_map = {
        "no_change": "#CCCCCC", "core": PUB_PALETTE["core"], "A1": PUB_PALETTE["A1"],
        "A2": PUB_PALETTE["A2"], "aux_other": "#D98E7C", "source_other": "#999999",
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    idx = counts.index
    labels = [label_map[i] for i in idx]
    colors = [color_map[i] for i in idx]
    bars = ax.barh(labels, counts.values, color=colors)
    for bar, val in zip(bars, counts.values):
        ax.annotate(f"{val:.1f}%", (val, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=11, fontweight="bold")
    ax.set_xlabel("Share of conversations (%)")
    ax.set_title("Location of coded belief change", loc="left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_change_composition.png")
    fig.savefig(OUTPUT_DIR / "fig3_change_composition.pdf")
    plt.close(fig)


# --------- FACTOR EFFECT VISUALS (CORE vs AUX) ---------

def plot_core_vs_aux_by_factor(sample_df, factor, fname, title, x_label=None):
    """Mean Delta core, Delta A1, Delta A2 with 95% CI by factor level."""
    if factor not in sample_df.columns:
        return

    d = sample_df[[factor, "delta_core", "delta_A1", "delta_A2"]].dropna().copy()
    if d.empty:
        return

    d_long = d.melt(id_vars=factor, value_vars=["delta_core", "delta_A1", "delta_A2"],
                     var_name="measure", value_name="delta")
    label_map = {"delta_core": "Delta core", "delta_A1": "Delta A1", "delta_A2": "Delta A2"}
    color_map = {"delta_core": PUB_PALETTE["core"], "delta_A1": PUB_PALETTE["A1"], "delta_A2": PUB_PALETTE["A2"]}
    d_long["measure_label"] = d_long["measure"].map(label_map)

    set_pub_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    sns.pointplot(
        data=d_long, x=factor, y="delta", hue="measure_label",
        palette=[color_map[m] for m in ["delta_core", "delta_A1", "delta_A2"]],
        dodge=0.3, errorbar=("ci", 95), linestyle="none", ax=ax,
    )

    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_ylabel("Mean Delta belief (end - start)")
    ax.set_title(title, loc="left")
    if x_label:
        ax.set_xlabel(x_label)
    plt.xticks(rotation=20, ha="right")
    ax.legend(title=None)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{fname}.png")
    fig.savefig(OUTPUT_DIR / f"{fname}.pdf")
    plt.close(fig)


# --------- MAIN ---------

def main(input_csv=INPUT_CSV):
    df, probe = load_and_prepare(input_csv)
    sample_df = build_sample_summary(probe)
    save_table(sample_df, "sample_level_summary")

    paired_table = pd.DataFrame([
        paired_effect_summary(sample_df, "belief_core_start", "belief_core_end", "Core belief"),
        paired_effect_summary(sample_df, "belief_A1_start", "belief_A1_end", "Auxiliary 1 belief"),
        paired_effect_summary(sample_df, "belief_A2_start", "belief_A2_end", "Auxiliary 2 belief"),
        paired_effect_summary(sample_df, "epconf_start", "epconf_end", "Epistemic confidence"),
    ])
    save_table(paired_table, "table1_paired_start_end")

    rq1_table = rq1_core_vs_aux(sample_df)
    save_table(rq1_table, "table_rq1_core_vs_aux")

    ols_outcomes = ["delta_core", "delta_A1", "delta_A2", "delta_epconf"]
    ols_tables = []
    for outcome in ols_outcomes:
        try:
            model = fit_ols_change_model(sample_df, outcome)
            tbl = extract_ols_summary(model, outcome)
            ols_tables.append(tbl)
            save_text(model.summary().as_text(), f"ols_{sanitize_filename(outcome)}")
        except Exception as e:
            save_text(str(e), f"ols_error_{sanitize_filename(outcome)}")

        try:
            mm = fit_mixedlm_change_model(sample_df, outcome)
            save_text(mm.summary().as_text(), f"mixedlm_{sanitize_filename(outcome)}")
        except Exception as e:
            save_text(str(e), f"mixedlm_error_{sanitize_filename(outcome)}")

    if ols_tables:
        save_table(pd.concat(ols_tables, ignore_index=True), "table2_ols_change_models")

    d_cat = standardized_change_category(sample_df)

    crosstab_summaries = []
    for factor in FACTOR_LIST:
        for outcome in ["assistant_stance_final", "change_category"]:
            out = crosstab_factor(
                d_cat if outcome == "change_category" else sample_df,
                factor=factor, outcome=outcome,
            )
            if out is not None:
                fname = f"crosstab_{sanitize_filename(outcome)}__{sanitize_filename(factor)}"
                save_table(out, fname)
                crosstab_summaries.append({
                    "outcome": outcome, "factor": factor,
                    "chi2": out["chi2"].iloc[0], "p": out["p"].iloc[0], "dof": out["dof"].iloc[0],
                })

    if crosstab_summaries:
        save_table(pd.DataFrame(crosstab_summaries), "table3_crosstab_summaries")

    # Global figures
    plot_trajectory(probe)
    plot_stance_distribution(sample_df)
    plot_change_composition(sample_df)

    # Factor-of-interest visuals for RQ2
    plot_core_vs_aux_by_factor(sample_df, "style_id", "fig4_core_vs_aux_by_style",
                                "Delta beliefs by user style profile", "User style")
    plot_core_vs_aux_by_factor(sample_df, "trust_in_institutions", "fig5_core_vs_aux_by_trust",
                                "Delta beliefs by trust in institutions", "Trust in institutions")
    plot_core_vs_aux_by_factor(sample_df, "belief_anchor_level", "fig6_core_vs_aux_by_anchor",
                                "Delta beliefs by initial belief anchor level", "Belief anchor level")
    plot_core_vs_aux_by_factor(sample_df, "conversation_goal", "fig7_core_vs_aux_by_goal",
                                "Delta beliefs by conversation goal", "Conversation goal")
    plot_core_vs_aux_by_factor(sample_df, "elaboration_2_target_condition", "fig8_core_vs_aux_by_elab2_target",
                                "Delta beliefs by elaboration-2 auxiliary target", "Elaboration-2 target")

    rq_summary_lines = ["RQ1: Do auxiliaries move more than the core?"]
    if not rq1_table.empty:
        r = rq1_table.iloc[0]
        rq_summary_lines.append(
            f"Signed change: core M={r['mean_delta_core']:.2f}, aux M={r['mean_delta_aux_mean']:.2f}, "
            f"t={r['t_signed']:.2f}, p={r['p_signed']:.3g}"
        )
        rq_summary_lines.append(
            f"Absolute change: core M={r['mean_abs_delta_core']:.2f}, aux M={r['mean_abs_delta_aux_mean']:.2f}, "
            f"t={r['t_abs']:.2f}, p={r['p_abs']:.3g}"
        )

    rq_summary_lines += [
        "", "RQ2: Which factors predict change?",
        "See table2_ols_change_models.csv, ols_*.txt, mixedlm_*.txt "
        "(vignette random intercept), and figs 4-8 for core vs auxiliary deltas "
        "by style, trust, belief anchor, goal, and elaboration-2 target.",
        "", "RQ3: Where does change land and how often does the assistant challenge?",
        "See fig2_stance_distribution.*, fig3_change_composition.*, "
        "and table3_crosstab_summaries.csv.",
    ]

    save_text("\n".join(rq_summary_lines), "rq_summary")
    print(f"Paper-oriented analysis complete. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

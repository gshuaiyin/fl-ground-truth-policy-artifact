"""Original plotting utilities and frozen-result statistical panels."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'replay/figures'
SOURCES = {
 'rq1': ROOT / 'results/revision_v140/run_20260910_r3/analysis.json',
 'decomposition': ROOT / 'results/full/policy_decomposition.json',
 'rq2': ROOT / 'results/full/counterfactual_multiplicity.json',
 'rq5': ROOT / 'results/rq5_transfer_repair/outputs/transfer_full_identity_multiseed_analysis_v2/analysis.json',
}
BLUE, ORANGE, GRAY = '#0072B2', '#D55E00', '#555555'
STYLE = {
 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
 'xtick.labelsize': 7.5, 'ytick.labelsize': 8,
 'axes.spines.top': False, 'axes.spines.right': False,
 'axes.linewidth': .6, 'axes.edgecolor': GRAY, 'axes.axisbelow': True,
 'lines.linewidth': 1.1, 'pdf.fonttype': 42, 'ps.fonttype': 42,
 'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
}

def read_json(key):
    return json.loads(SOURCES[key].read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(filename, rows):
    with (HERE / filename).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(fig, stem, outputs):
    # Fixed ticks within the declared limits also avoid invisible out-of-range
    # locator labels being mistaken for clipped manuscript text.
    for axis in fig.axes:
        for getter, setter, limits in [(axis.get_xticks, axis.set_xticks, axis.get_xlim()),
                                       (axis.get_yticks, axis.set_yticks, axis.get_ylim())]:
            lo, hi = sorted(limits)
            ticks = getter()
            kept = [tick for tick in ticks if lo <= tick <= hi]
            if len(kept) != len(ticks):
                setter(kept)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    outside = []
    for artist in fig.findobj():
        if isinstance(artist, matplotlib.text.Text) and artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
                outside.append(artist.get_text())
    if outside:
        raise AssertionError(f"Text beyond page in {stem}: {outside}")
    for extension in ["pdf", "svg", "png"]:
        path = HERE / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches=None, transparent=False)
        outputs.append({"file": path.name, "sha256": sha(path)})
    plt.close(fig)


def rq1_candidate(outputs):
    revision = read_json("rq1")
    decomposition = read_json("decomposition")
    formulas = ["ochiai", "dstar", "tarantula", "barinel"]
    names = ["Ochiai", "DStar", "Tarantula", "Barinel"]
    comparisons = ["deleted to direct", "deleted to strict", "deleted to broad", "direct to strict"]
    sample = [234, 345, 345, 250]
    rows = []
    for formula in formulas:
        original = decomposition["contrasts"]["deleted_to_direct"]["by_formula"][formula]
        values = [(original["mrr"]["all_paired_mean"], original["mrr"]["all_paired_bootstrap_95ci"])]
        for key in ["deleted_strict_all", "deleted_broad_all", "direct_strict_all"]:
            item = revision["comparisons"][key][formula]["average"]
            values.append((item["delta_mrr"], item["delta_mrr_ci"]))
        for comparison, n, (mean, ci) in zip(comparisons, sample, values):
            rows.append(dict(formula=formula, comparison=comparison, paired_n=n,
                             delta_mrr=mean, ci_low=ci[0], ci_high=ci[1]))
    assert len(rows) == 16
    fig = plt.figure(figsize=(190/25.4, 133/25.4), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[.9, 1.5], hspace=.09)
    ax = fig.add_subplot(outer[0])
    counts = [348, 250, 374, 376]
    policies = ["deleted", "direct", "strict", "broad"]
    y = np.arange(4)
    ax.barh(y, counts, height=.55, color=["#C9CDD0", "#C9CDD0", BLUE, "#A9CFE5"],
            edgecolor=GRAY, linewidth=.6)
    for i, count in enumerate(counts):
        ax.text(count + 4, i, f"{count}/395", va="center", fontsize=8)
    ax.set(yticks=y, yticklabels=policies, xlim=(0, 430), xlabel="Evaluable faults")
    ax.invert_yaxis()
    ax.set_title("(a) Evaluation coverage; samples can overlap", loc="left")
    ax.grid(axis="x", color=".9", linewidth=.5)
    inner = outer[1].subgridspec(1, 4, wspace=.08)
    low = min(row["ci_low"] for row in rows)
    high = max(row["ci_high"] for row in rows)
    margin = (high-low)*.10
    for i, (formula, name) in enumerate(zip(formulas, names)):
        a = fig.add_subplot(inner[i])
        selected = [row for row in rows if row["formula"] == formula]
        for j, row in enumerate(selected):
            a.errorbar(row["delta_mrr"], j,
                       xerr=[[row["delta_mrr"]-row["ci_low"]], [row["ci_high"]-row["delta_mrr"]]],
                       fmt="o", ms=4, color=BLUE, capsize=2, linewidth=1)
        a.axvline(0, color=GRAY, linestyle="--", linewidth=.7)
        labels = [f"{c}\n(n={n})" for c, n in zip(comparisons, sample)] if i == 0 else [""]*4
        a.set(yticks=np.arange(4), yticklabels=labels, ylim=(3.5, -.5),
              xlim=(low-margin, high+margin), xlabel="MRR difference")
        a.tick_params(axis="y", length=0)
        a.xaxis.set_major_locator(MaxNLocator(nbins=3))
        a.set_title(f"({chr(98+i)}) {name}", loc="left")
        a.grid(axis="x", color=".9", linewidth=.5)
    export(fig, "candidate_rq1_coverage_and_paired_effects", outputs)
    write_csv("rq1_paired_effects.csv", rows)
    return {"comparisons": comparisons, "paired_sample_sizes": sample,
            "uncertainty": "Frozen error-level paired bootstrap 95% CI; no significance stars. Top-k Holm results are separate.",
            "scope": "Full common samples; omission-only subsets and direct-to-broad remain in supplement."}


def rq2_candidate(outputs):
    source = read_json("rq2")
    assert len(source["rows"]) == 112
    metric_specs = [("top10", "Expected Top-10 hit rate", 100., "%", (0, 45)),
                    ("mrr", "Expected MRR", 1., "", (0, .19)),
                    ("exam", "Expected EXAM (lower is better)", 1., "", (0, .20)),
                    ("gmar", "Expected GMAR-F", 1., "", (.60, .93)),
                    ("gmar_average", "Expected GMAR-A", 1., "", (.60, .93))]
    fig, axes = plt.subplots(2, 3, figsize=(190/25.4, 129/25.4), layout="constrained")
    rows = []
    for index, (metric, title, scale, unit, limits) in enumerate(metric_specs):
        ax = axes.flat[index]
        values = [source["means"][f"m{q}"][metric]*scale for q in [1, 2, 4]]
        for q, value in zip([1, 2, 4], values):
            mean_rows = [row[f"m{q}_{metric}"] for row in source["rows"]]
            assert np.isclose(np.mean(mean_rows)*scale, value, atol=1e-11)
            rows.append(dict(metric=metric, target_count=q, mean_exact_expectation=value,
                             unit="percent" if metric == "top10" else "score"))
        ax.plot([1, 2, 4], values, marker="o", ms=4, color=BLUE, linestyle="--")
        ax.set(xticks=[1, 2, 4], xlim=(.5, 4.5), ylim=limits, xlabel="Retained targets")
        ax.set_title(f"({chr(97+index)}) {title}", loc="left", fontsize=8.1)
        ax.grid(axis="y", color=".9", linewidth=.5)
        contrast = source["contrasts"][metric]
        effect = contrast["mean_m4_minus_m1"]*scale
        lo, hi = [v*scale for v in contrast["bug_bootstrap_95ci"]]
        precision = 2 if metric == "top10" else 3
        note = f"4 minus 1: {effect:+.{precision}f}" + (" pp" if metric == "top10" else "")
        note += f"\n95% CI [{lo:.{precision}f}, {hi:.{precision}f}]"
        ax.text(.03, .97, note, transform=ax.transAxes, va="top", fontsize=7.1,
                bbox={"facecolor":"white", "edgecolor":"none", "alpha":1})
        if metric == "top10": ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax = axes.flat[5]
    delta = np.array([row["m4_top10"]-row["m1_top10"] for row in source["rows"]])*100
    delta.sort()
    assert np.all(delta >= -1e-10)
    ax.scatter(np.arange(1, 113), delta, s=6, c=BLUE)
    ax.axhline(0, color=GRAY, linewidth=.6)
    ax.set(xlabel="Faults ordered by change", ylabel="Top-10 change (pp)",
           xlim=(0, 113), ylim=(-4, 100))
    ax.set_title("(f) Per-fault changes (n = 112)", loc="left", fontsize=8.1)
    increased = int(np.sum(delta > 1e-9))
    ax.text(.03, .97, f"{increased}/112 increase\n{112-increased}/112 unchanged",
            transform=ax.transAxes, va="top", fontsize=7.2)
    ax.grid(axis="y", color=".9", linewidth=.5)
    export(fig, "candidate_rq2_target_count", outputs)
    write_csv("rq2_exact_expected_means.csv", rows)
    write_csv("rq2_per_fault_top10_changes.csv", [dict(bug_id=row["bug_id"], project=row["project"],
               m1_expected_top10=row["m1_top10"], m4_expected_top10=row["m4_top10"],
               change_pp=(row["m4_top10"]-row["m1_top10"])*100) for row in source["rows"]])
    return {"n":112, "target_counts":[1,2,4], "increased_expected_top10_faults":increased,
            "uncertainty":"Annotations use frozen paired-bootstrap 95% CI for the 4-minus-1 contrast; points are exact subset expectations averaged over faults.",
            "transformation":"Top-10 multiplied by 100; EXAM retains raw direction. Broken lines guide the eye; no interpolation is estimated.",
            "new_descriptive_summary":"Panel f counts and sorts frozen per-fault expectations; it is not a new localization experiment."}


def rq5_candidate(outputs):
    source = read_json("rq5")
    assert len(source["seed_results"]) == 5
    specs = [("delta_mrr_right_minus_left", "MRR", BLUE),
             ("delta_gmar_f_right_minus_left", "GMAR-F", GRAY),
             ("delta_gmar_a_right_minus_left", "GMAR-A", ORANGE)]
    fig, axes = plt.subplots(1, 3, figsize=(190/25.4, 92/25.4), layout="constrained")
    rows, patterns = [], []
    for index, (key, title, color) in enumerate(specs):
        ax = axes[index]
        bounds = []
        for y, run in enumerate(source["seed_results"]):
            comparison = run["comparisons"]["direct_to_strict"]
            assert comparison["paired"] == 240
            item = comparison["continuous"][key]
            mean, (lo, hi) = item["mean"], item["bootstrap_95ci"]
            assert lo <= mean <= hi
            ax.errorbar(mean, y, xerr=[[mean-lo],[hi-mean]], fmt="o", color=color,
                        ms=4.5, capsize=2.5, linewidth=1.2)
            bounds += [lo, hi, 0]
            rows.append(dict(seed=run["seed"], metric=title, paired_n=240,
                             mean_difference=mean, ci_low=lo, ci_high=hi))
        padding = (max(bounds)-min(bounds))*.17
        ax.set(xlim=(min(bounds)-padding,max(bounds)+padding), ylim=(4.6,-.6),
               yticks=np.arange(5), yticklabels=[str(run["seed"]) for run in source["seed_results"]],
               xlabel="strict minus direct")
        ax.set_title(f"({chr(97+index)}) {title}", loc="left")
        ax.axvline(0, color=GRAY, linestyle="--", linewidth=.8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="x", color=".9", linewidth=.5)
        if index == 0: ax.set_ylabel("Training seed")
    export(fig, "candidate_rq5_seed_effects", outputs)
    for run in source["seed_results"]:
        eligible = [row for row in run["rows"] if row["direct"]["rank"] is not None and row["strict"]["rank"] is not None]
        assert len(eligible) == 240
        both = sum(1/row["strict"]["rank"]-1/row["direct"]["rank"] > 1e-12 and
                   row["strict"]["gmar_a"]-row["direct"]["gmar_a"] < -1e-12 for row in eligible)
        patterns.append(dict(seed=run["seed"], paired_n=240,
                             first_hit_improves_and_average_position_score_declines=both))
    write_csv("rq5_seed_effects.csv", rows)
    write_csv("rq5_joint_pattern_counts.csv", patterns)
    return {"comparison":"direct to strict", "paired_n_per_seed":240,
            "uncertainty":"Frozen error-level paired bootstrap 95% CI, separately for each seed; no combined five-seed CI or enlarged sample.",
            "axes":"Separate limits for each metric; compare direction, not distances across panels.",
            "joint_counts":patterns,
            "joint_count_status":"New descriptive tabulation of frozen per-fault rows, not a replacement for the aggregate inference."}

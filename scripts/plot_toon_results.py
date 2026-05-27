import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hyperagents-toon-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


RUN_PAIRS = [
    {
        "label": "30-sample original",
        "short_label": "30 original",
        "raw": "raw_neutral_baseline",
        "toon": "retrieval_toon_controlled",
    },
    {
        "label": "30-sample repeat",
        "short_label": "30 repeat",
        "raw": "raw_neutral_repeat1",
        "toon": "retrieval_toon_repeat1",
    },
    {
        "label": "60-sample confirmation",
        "short_label": "60 confirm",
        "raw": "raw_neutral_60",
        "toon": "retrieval_toon_60",
    },
]

TOKEN_COLUMNS = ["prompt_tokens", "completion_tokens", "total_tokens"]
RAW_COLOR = "#4C78A8"
TOON_COLOR = "#F58518"
GOOD_COLOR = "#54A24B"
BAD_COLOR = "#E45756"
NEUTRAL_COLOR = "#8E8E93"
GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#222222"


def _configure_matplotlib():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "savefig.bbox": "tight",
            "savefig.dpi": 180,
        }
    )


def _read_report(runs_dir, run_name):
    with (runs_dir / run_name / "report.json").open() as f:
        return json.load(f)


def _read_token_totals(runs_dir, run_name):
    totals = Counter()
    count = 0
    with (runs_dir / run_name / "token_usage.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            record = json.loads(line)
            for column in TOKEN_COLUMNS:
                totals[column] += int(record[column])
    totals["records"] = count
    return totals


def _read_predictions(runs_dir, run_name):
    csv.field_size_limit(sys.maxsize)
    with (runs_dir / run_name / "predictions.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def _is_correct(row):
    return row["prediction"].strip().lower() == row["outcome"].strip().lower()


def _pct(value):
    return 100.0 * value


def _pct_delta(new, old):
    if old == 0:
        return 0.0
    return 100.0 * (new - old) / old


def _collect_summary(runs_dir):
    rows = []
    for pair in RUN_PAIRS:
        raw_report = _read_report(runs_dir, pair["raw"])
        toon_report = _read_report(runs_dir, pair["toon"])
        raw_tokens = _read_token_totals(runs_dir, pair["raw"])
        toon_tokens = _read_token_totals(runs_dir, pair["toon"])

        row = {
            **pair,
            "samples": int(raw_report["total"]),
            "raw_correct": int(raw_report["total_correct"]),
            "toon_correct": int(toon_report["total_correct"]),
            "raw_accuracy": float(raw_report["overall_accuracy"]),
            "toon_accuracy": float(toon_report["overall_accuracy"]),
            "accuracy_delta": float(toon_report["overall_accuracy"])
            - float(raw_report["overall_accuracy"]),
        }
        for column in TOKEN_COLUMNS:
            row[f"raw_{column}"] = int(raw_tokens[column])
            row[f"toon_{column}"] = int(toon_tokens[column])
            row[f"{column}_delta"] = int(toon_tokens[column] - raw_tokens[column])
            row[f"{column}_delta_pct"] = _pct_delta(toon_tokens[column], raw_tokens[column])
        rows.append(row)
    return rows


def _style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def _save(fig, out_dir, name, pdf=None):
    path = out_dir / name
    fig.savefig(path)
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return path


def _bar_label(ax, bars, labels, dy=1.0):
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_COLOR,
        )


def plot_summary_card(summary):
    row60 = summary[-1]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")

    fig.text(
        0.05,
        0.88,
        "TOON paper-review validation",
        fontsize=30,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.text(
        0.05,
        0.79,
        "Same 60-sample pass rate, 68.2% fewer total tokens",
        fontsize=19,
        color="#333333",
    )

    cards = [
        ("60-sample accuracy", "50/60", "raw", "50/60", "TOON", RAW_COLOR),
        ("Total token reduction", "-68.21%", "TOON vs raw", "", "", GOOD_COLOR),
        ("Prompt token reduction", "-72.98%", "TOON vs raw", "", "", GOOD_COLOR),
    ]
    card_w = 0.28
    for idx, (title, value1, label1, value2, label2, color) in enumerate(cards):
        left = 0.05 + idx * 0.31
        rect = plt.Rectangle(
            (left, 0.48),
            card_w,
            0.22,
            transform=fig.transFigure,
            facecolor="#F7F7F8",
            edgecolor="#D0D0D4",
            linewidth=1.2,
        )
        fig.patches.append(rect)
        fig.text(left + 0.02, 0.65, title, fontsize=12, color="#555555")
        fig.text(left + 0.02, 0.56, value1, fontsize=28, fontweight="bold", color=color)
        fig.text(left + 0.02, 0.515, label1, fontsize=10, color="#666666")
        if value2:
            fig.text(left + 0.15, 0.56, value2, fontsize=28, fontweight="bold", color=TOON_COLOR)
            fig.text(left + 0.15, 0.515, label2, fontsize=10, color="#666666")

    table_y = 0.32
    fig.text(0.05, table_y + 0.08, "Controlled comparisons", fontsize=14, fontweight="bold")
    headers = ["Run", "Raw", "TOON", "Accuracy delta", "Total tokens"]
    xs = [0.05, 0.32, 0.43, 0.55, 0.72]
    for x, header in zip(xs, headers):
        fig.text(x, table_y + 0.035, header, fontsize=10, fontweight="bold", color="#555555")
    for i, row in enumerate(summary):
        y = table_y - (i * 0.055)
        fig.text(xs[0], y, row["label"], fontsize=10, color=TEXT_COLOR)
        fig.text(xs[1], y, f'{row["raw_correct"]}/{row["samples"]}', fontsize=10)
        fig.text(xs[2], y, f'{row["toon_correct"]}/{row["samples"]}', fontsize=10)
        fig.text(xs[3], y, f'{row["accuracy_delta"]:+.4f}', fontsize=10)
        fig.text(xs[4], y, f'{row["total_tokens_delta_pct"]:.2f}%', fontsize=10, color=GOOD_COLOR)

    fig.text(
        0.05,
        0.08,
        "Bottom line: no aggregate accuracy damage was observed; residual risk is distributional, with TOON weaker on short papers in the 60-sample slice.",
        fontsize=11,
        color="#444444",
    )
    return fig


def plot_accuracy(summary):
    labels = [row["short_label"] for row in summary]
    x = list(range(len(summary)))
    width = 0.36
    raw_values = [_pct(row["raw_accuracy"]) for row in summary]
    toon_values = [_pct(row["toon_accuracy"]) for row in summary]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    raw_bars = ax.bar([i - width / 2 for i in x], raw_values, width, color=RAW_COLOR, label="Raw neutral")
    toon_bars = ax.bar([i + width / 2 for i in x], toon_values, width, color=TOON_COLOR, label="Retrieval TOON")
    _bar_label(ax, raw_bars, [f'{row["raw_correct"]}/{row["samples"]}' for row in summary])
    _bar_label(ax, toon_bars, [f'{row["toon_correct"]}/{row["samples"]}' for row in summary])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Pass rate stayed effectively matched across validation runs", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    return fig


def plot_token_savings(summary):
    labels = [row["short_label"] for row in summary]
    x = list(range(len(summary)))
    width = 0.36
    raw_values = [row["raw_total_tokens"] / 1_000_000 for row in summary]
    toon_values = [row["toon_total_tokens"] / 1_000_000 for row in summary]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    raw_bars = ax.bar([i - width / 2 for i in x], raw_values, width, color=RAW_COLOR, label="Raw neutral")
    toon_bars = ax.bar([i + width / 2 for i in x], toon_values, width, color=TOON_COLOR, label="Retrieval TOON")
    _bar_label(ax, raw_bars, [f'{v:.2f}M' for v in raw_values], dy=0.03)
    _bar_label(ax, toon_bars, [f'{v:.2f}M' for v in toon_values], dy=0.03)
    for i, row in enumerate(summary):
        ax.text(
            i,
            max(raw_values[i], toon_values[i]) + 0.15,
            f'{row["total_tokens_delta_pct"]:.1f}%',
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=GOOD_COLOR,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total tokens (millions)")
    ax.set_ylim(0, max(raw_values) * 1.25)
    ax.set_title("TOON cut total token use by about two thirds", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    return fig


def plot_token_breakdown(summary):
    labels = [row["short_label"] for row in summary]
    prompt_reductions = [-row["prompt_tokens_delta_pct"] for row in summary]
    total_reductions = [-row["total_tokens_delta_pct"] for row in summary]
    x = list(range(len(summary)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 6))
    prompt_bars = ax.bar([i - width / 2 for i in x], prompt_reductions, width, color="#72B7B2", label="Prompt token reduction")
    total_bars = ax.bar([i + width / 2 for i in x], total_reductions, width, color=GOOD_COLOR, label="Total token reduction")
    _bar_label(ax, prompt_bars, [f"{v:.1f}%" for v in prompt_reductions], dy=1.0)
    _bar_label(ax, total_bars, [f"{v:.1f}%" for v in total_reductions], dy=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 85)
    ax.set_ylabel("Reduction vs raw neutral (%)")
    ax.set_title("Prompt-token savings were stable across reruns", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    return fig


def _collect_flip_data(runs_dir):
    raw_rows = {r["question_id"]: r for r in _read_predictions(runs_dir, "raw_neutral_60")}
    toon_rows = {r["question_id"]: r for r in _read_predictions(runs_dir, "retrieval_toon_60")}
    flips = Counter()
    for question_id, raw in raw_rows.items():
        toon = toon_rows[question_id]
        flips[(_is_correct(raw), _is_correct(toon))] += 1
    return flips


def plot_flip_analysis(flips):
    labels = [
        "Both correct",
        "Raw only",
        "TOON only",
        "Both wrong",
    ]
    values = [
        flips[(True, True)],
        flips[(True, False)],
        flips[(False, True)],
        flips[(False, False)],
    ]
    colors = [GOOD_COLOR, RAW_COLOR, TOON_COLOR, NEUTRAL_COLOR]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=colors)
    _bar_label(ax, bars, [str(v) for v in values], dy=0.4)
    ax.set_ylim(0, max(values) + 8)
    ax.set_ylabel("Samples")
    ax.set_title("60-sample flips were balanced: 6 against TOON, 6 in favor", fontweight="bold")
    _style_axis(ax)
    return fig


def _collect_stratified_data(runs_dir):
    raw_rows = {r["question_id"]: r for r in _read_predictions(runs_dir, "raw_neutral_60")}
    toon_rows = {r["question_id"]: r for r in _read_predictions(runs_dir, "retrieval_toon_60")}

    outcome_rows = []
    by_outcome = defaultdict(lambda: Counter(n=0, raw=0, toon=0))
    for question_id, raw in raw_rows.items():
        toon = toon_rows[question_id]
        outcome = raw["outcome"].strip().lower()
        by_outcome[outcome]["n"] += 1
        by_outcome[outcome]["raw"] += int(_is_correct(raw))
        by_outcome[outcome]["toon"] += int(_is_correct(toon))
    for outcome in ["accept", "reject"]:
        counts = by_outcome[outcome]
        outcome_rows.append(
            {
                "label": outcome.capitalize(),
                "n": counts["n"],
                "raw": counts["raw"],
                "toon": counts["toon"],
                "range": "",
            }
        )

    length_records = sorted(
        (
            {
                "question_id": question_id,
                "chars": len(raw["paper_text"]),
                "raw_correct": int(_is_correct(raw)),
                "toon_correct": int(_is_correct(toon_rows[question_id])),
            }
            for question_id, raw in raw_rows.items()
        ),
        key=lambda record: record["chars"],
    )
    length_rows = []
    bucket_labels = ["Short", "Medium", "Long"]
    bucket_size = len(length_records) // 3
    for idx, label in enumerate(bucket_labels):
        start = idx * bucket_size
        end = (idx + 1) * bucket_size if idx < 2 else len(length_records)
        bucket = length_records[start:end]
        length_rows.append(
            {
                "label": label,
                "n": len(bucket),
                "raw": sum(record["raw_correct"] for record in bucket),
                "toon": sum(record["toon_correct"] for record in bucket),
                "range": f'{bucket[0]["chars"]:,}-{bucket[-1]["chars"]:,}',
            }
        )
    return outcome_rows, length_rows


def plot_stratified(outcome_rows, length_rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, rows, title in [
        (axes[0], outcome_rows, "By decision label"),
        (axes[1], length_rows, "By paper length"),
    ]:
        labels = [row["label"] for row in rows]
        x = list(range(len(rows)))
        width = 0.36
        raw_values = [100.0 * row["raw"] / row["n"] for row in rows]
        toon_values = [100.0 * row["toon"] / row["n"] for row in rows]
        raw_bars = ax.bar([i - width / 2 for i in x], raw_values, width, color=RAW_COLOR, label="Raw neutral")
        toon_bars = ax.bar([i + width / 2 for i in x], toon_values, width, color=TOON_COLOR, label="Retrieval TOON")
        _bar_label(ax, raw_bars, [f'{row["raw"]}/{row["n"]}' for row in rows], dy=1.0)
        _bar_label(ax, toon_bars, [f'{row["toon"]}/{row["n"]}' for row in rows], dy=1.0)
        ax.set_xticks(x)
        if rows and rows[0]["range"]:
            ax.set_xticklabels([f'{row["label"]}\n{row["range"]}' for row in rows], fontsize=9)
        else:
            ax.set_xticklabels(labels)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontweight="bold")
        _style_axis(ax)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("60-sample stratified accuracy", fontsize=16, fontweight="bold", y=1.02)
    return fig


def _write_markdown(out_dir, summary, output_names):
    row60 = summary[-1]
    lines = [
        "# TOON Paper Review Share Summary",
        "",
        "## Headline",
        "",
        (
            f"On the balanced 60-sample confirmation run, raw neutral and retrieval TOON "
            f"both scored {row60['raw_correct']}/{row60['samples']} while TOON used "
            f"{-row60['total_tokens_delta_pct']:.2f}% fewer total tokens."
        ),
        "",
        "## Controlled Checks",
        "",
        "| comparison | raw | TOON | accuracy delta | total token delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['label']} | {row['raw_correct']}/{row['samples']} | "
            f"{row['toon_correct']}/{row['samples']} | {row['accuracy_delta']:+.4f} | "
            f"{row['total_tokens_delta_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Key Takeaways",
            "",
            "- No aggregate accuracy damage was observed across the controlled validation runs.",
            "- Total token savings were stable at about 67-68%.",
            "- Prompt-token savings were stable at about 72-73%.",
            "- On the 60-sample run, individual flips were balanced: 6 raw-only correct and 6 TOON-only correct.",
            "- Residual risk: TOON was weaker on the short-paper bucket, offset by medium and long papers.",
            "",
            "## Plots",
            "",
        ]
    )
    for title, filename in output_names:
        lines.extend([f"### {title}", "", f"![{title}]({filename})", ""])
    lines.extend(
        [
            "## Regenerate",
            "",
            "```bash",
            "UV_CACHE_DIR=/tmp/hyperagents-toon-uv-cache uv run --with matplotlib python scripts/plot_toon_results.py",
            "```",
            "",
        ]
    )
    (out_dir / "share_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_plots(runs_dir, out_dir):
    _configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _collect_summary(runs_dir)
    flips = _collect_flip_data(runs_dir)
    outcome_rows, length_rows = _collect_stratified_data(runs_dir)

    outputs = [
        ("Executive Summary", "00_summary_card.png", plot_summary_card(summary)),
        ("Accuracy Across Runs", "01_accuracy_across_runs.png", plot_accuracy(summary)),
        ("Total Token Use", "02_total_tokens.png", plot_token_savings(summary)),
        ("Reduction Stability", "03_reduction_stability.png", plot_token_breakdown(summary)),
        ("60-Sample Flip Analysis", "04_flip_analysis_60.png", plot_flip_analysis(flips)),
        ("60-Sample Stratified Accuracy", "05_stratified_accuracy_60.png", plot_stratified(outcome_rows, length_rows)),
    ]

    pdf_path = out_dir / "toon_findings_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        output_names = []
        for title, filename, fig in outputs:
            _save(fig, out_dir, filename, pdf=pdf)
            output_names.append((title, filename))

    _write_markdown(out_dir, summary, output_names)
    return output_names, pdf_path


def main():
    parser = argparse.ArgumentParser(description="Generate shareable plots for TOON paper-review validation.")
    parser.add_argument("--runs-dir", default="runs", type=Path)
    parser.add_argument("--out-dir", default=Path("runs/toon_plots"), type=Path)
    args = parser.parse_args()

    output_names, pdf_path = build_plots(args.runs_dir, args.out_dir)
    print(f"Wrote {len(output_names)} PNG plots to {args.out_dir}")
    print(f"Wrote PDF bundle to {pdf_path}")
    print(f"Wrote summary to {args.out_dir / 'share_summary.md'}")


if __name__ == "__main__":
    main()

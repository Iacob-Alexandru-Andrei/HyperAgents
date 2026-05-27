import argparse
import json
from pathlib import Path

import pandas as pd


TOKEN_COLUMNS = ["prompt_tokens", "completion_tokens", "total_tokens"]


def _format_cell(value, floatfmt=None):
    if isinstance(value, float) and floatfmt:
        return format(value, floatfmt)
    return str(value)


def _markdown_table(df, floatfmt=None):
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(_format_cell(row[column], floatfmt) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def _read_report(run_dir):
    path = Path(run_dir) / "report.json"
    with path.open() as f:
        return json.load(f)


def _read_token_log(run_dir):
    path = Path(run_dir) / "token_usage.jsonl"
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["question_id", *TOKEN_COLUMNS])
    for column in TOKEN_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    return df.groupby("question_id", as_index=False)[TOKEN_COLUMNS].sum()


def _metric_delta(csv_report, toon_report, csv_label="csv", toon_label="toon"):
    rows = [
        {
            "metric": "overall_accuracy",
            csv_label: csv_report["overall_accuracy"],
            toon_label: toon_report["overall_accuracy"],
            "delta": toon_report["overall_accuracy"] - csv_report["overall_accuracy"],
        }
    ]
    labels = sorted(
        set(csv_report["accuracy_by_ground_truth"])
        | set(toon_report["accuracy_by_ground_truth"])
    )
    for label in labels:
        csv_label_metrics = csv_report["accuracy_by_ground_truth"].get(label, {})
        toon_label_metrics = toon_report["accuracy_by_ground_truth"].get(label, {})
        for metric in ["precision", "recall"]:
            csv_value = float(csv_label_metrics.get(metric, 0.0))
            toon_value = float(toon_label_metrics.get(metric, 0.0))
            rows.append(
                {
                    "metric": f"{label}_{metric}",
                    csv_label: csv_value,
                    toon_label: toon_value,
                    "delta": toon_value - csv_value,
                }
            )
    return pd.DataFrame(rows)


def _token_delta(csv_dir, toon_dir, csv_label="csv", toon_label="toon"):
    csv_tokens = _read_token_log(csv_dir).rename(
        columns={column: f"{csv_label}_{column}" for column in TOKEN_COLUMNS}
    )
    toon_tokens = _read_token_log(toon_dir).rename(
        columns={column: f"{toon_label}_{column}" for column in TOKEN_COLUMNS}
    )
    merged = csv_tokens.merge(toon_tokens, on="question_id", how="outer").fillna(0)
    for column in TOKEN_COLUMNS:
        merged[f"delta_{column}"] = (
            merged[f"{toon_label}_{column}"] - merged[f"{csv_label}_{column}"]
        )
    return merged.sort_values("question_id")


def _format_percent_delta(delta, baseline):
    if baseline == 0:
        return "n/a"
    return f"{(delta / baseline) * 100:.2f}%"


def write_comparison(
    csv_dir,
    toon_dir,
    out_path,
    title="TOON vs CSV Loader Benchmark",
    csv_label="csv",
    toon_label="toon",
):
    csv_report = _read_report(csv_dir)
    toon_report = _read_report(toon_dir)
    metrics = _metric_delta(csv_report, toon_report, csv_label, toon_label)
    tokens = _token_delta(csv_dir, toon_dir, csv_label, toon_label)

    token_totals = {}
    for column in TOKEN_COLUMNS:
        csv_total = int(tokens[f"{csv_label}_{column}"].sum())
        toon_total = int(tokens[f"{toon_label}_{column}"].sum())
        token_totals[column] = {
            "csv": csv_total,
            "toon": toon_total,
            "delta": toon_total - csv_total,
            "delta_pct": _format_percent_delta(toon_total - csv_total, csv_total),
        }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("## Aggregate Metrics\n\n")
        f.write(_markdown_table(metrics, floatfmt=".4f"))
        f.write("\n\n## Token Totals\n\n")
        f.write(f"| token_type | {csv_label} | {toon_label} | delta | delta_pct |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for column, values in token_totals.items():
            f.write(
                f"| {column} | {values['csv']} | {values['toon']} | "
                f"{values['delta']} | {values['delta_pct']} |\n"
            )
        f.write("\n## Per-Sample Token Delta\n\n")
        f.write(_markdown_table(tokens))
        f.write("\n")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare CSV and TOON benchmark runs.")
    parser.add_argument("--csv", required=True, help="CSV baseline run directory")
    parser.add_argument("--toon", required=True, help="TOON loader run directory")
    parser.add_argument("--out", required=True, help="Markdown output path")
    parser.add_argument(
        "--title",
        default="TOON vs CSV Loader Benchmark",
        help="Markdown report title",
    )
    parser.add_argument("--csv-label", default="csv", help="Left/run label")
    parser.add_argument("--toon-label", default="toon", help="Right/run label")
    args = parser.parse_args()

    write_comparison(
        args.csv,
        args.toon,
        args.out,
        title=args.title,
        csv_label=args.csv_label,
        toon_label=args.toon_label,
    )

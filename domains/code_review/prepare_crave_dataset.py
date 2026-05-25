import argparse
import csv
import random
from pathlib import Path

from datasets import load_dataset

from domains.code_review.utils import encode_patch_text


DATASET_NAME = "TuringEnterprises/CRAVE"
HF_TO_LOCAL_SPLITS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}
OUTPUT_COLUMNS = ("question_id", "patch", "outcome")
LABEL_TO_OUTCOME = {
    "APPROVE": "pass",
    "APPROVED": "pass",
    "REQUEST_CHANGES": "fail",
    "CHANGES_REQUESTED": "fail",
}


def full_dataset_name(split):
    return f"dataset_{split}.csv"


def filtered_dataset_name(split, limit=100):
    return f"dataset_filtered_{limit}_{split}.csv"


def _normalized_label(value):
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def _first_present(row, keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def normalize_crave_row(row, *, split, index):
    label = _first_present(row, ("label", "decision", "review_decision", "outcome"))
    normalized_label = _normalized_label(label)
    if normalized_label not in LABEL_TO_OUTCOME:
        raise ValueError(f"Unsupported CRAVE label: {label!r}")

    patch = _first_present(row, ("patch", "diff"))
    if patch is None:
        raise ValueError(f"Missing CRAVE patch for {split} row {index}")

    return {
        "question_id": f"crave_{split}_{index:06d}",
        "patch": encode_patch_text(patch),
        "outcome": LABEL_TO_OUTCOME[normalized_label],
    }


def filtered_rows(rows, limit=100, seed=42):
    if limit <= 0:
        return []

    buckets = {"pass": [], "fail": []}
    for row in rows:
        outcome = row["outcome"]
        if outcome not in buckets:
            raise ValueError(f"Unsupported outcome: {outcome!r}")
        buckets[outcome].append(row)

    rng = random.Random(seed)
    shuffled = {}
    for outcome, bucket in buckets.items():
        shuffled[outcome] = list(bucket)
        rng.shuffle(shuffled[outcome])

    target = min(limit, len(rows))
    base_per_label = target // 2
    pass_count = min(len(shuffled["pass"]), base_per_label + target % 2)
    fail_count = min(len(shuffled["fail"]), base_per_label)

    selected = shuffled["pass"][:pass_count] + shuffled["fail"][:fail_count]
    remaining_slots = target - len(selected)
    if remaining_slots > 0:
        leftovers = shuffled["pass"][pass_count:] + shuffled["fail"][fail_count:]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining_slots])

    rng.shuffle(selected)
    return selected


def write_rows(rows, path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(output_dir, dataset_name=DATASET_NAME, filtered_size=100, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for hf_split, local_split in HF_TO_LOCAL_SPLITS.items():
        dataset = load_dataset(dataset_name, split=hf_split)
        rows = [
            normalize_crave_row(row, split=local_split, index=index)
            for index, row in enumerate(dataset)
        ]
        filtered = filtered_rows(rows, limit=filtered_size, seed=seed)

        write_rows(rows, output_dir / full_dataset_name(local_split))
        write_rows(filtered, output_dir / filtered_dataset_name(local_split, filtered_size))

        stats[local_split] = {
            "full": len(rows),
            "filtered": len(filtered),
            "pass": sum(row["outcome"] == "pass" for row in rows),
            "fail": sum(row["outcome"] == "fail" for row in rows),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert TuringEnterprises/CRAVE into patch-only pass/fail CSV splits."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where dataset CSV files will be written.",
    )
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--filtered_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = prepare_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        filtered_size=args.filtered_size,
        seed=args.seed,
    )
    for split, split_stats in stats.items():
        print(
            f"{split}: full={split_stats['full']} filtered={split_stats['filtered']} "
            f"pass={split_stats['pass']} fail={split_stats['fail']}"
        )


if __name__ == "__main__":
    main()

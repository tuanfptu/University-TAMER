"""Combine evaluation reports into the exact table used by the thesis."""

import argparse
import csv
import json
from pathlib import Path


def load_metrics(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)["overall"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=directory containing university/metrics.json and hme100k/metrics.json",
    )
    parser.add_argument("--output", default="outputs/ablation_metrics.csv")
    args = parser.parse_args()
    rows = []
    for value in args.run:
        name, root_text = value.split("=", 1)
        root = Path(root_text)
        university = load_metrics(root / "university" / "metrics.json")
        hme = load_metrics(root / "hme100k" / "metrics.json")
        rows.append(
            {
                "model": name,
                "university_ExpRate": university["ExpRate"],
                "university_le_1": university["ExpRate_le_1"],
                "university_le_2": university["ExpRate_le_2"],
                "university_TER": university["TokenErrorRate"],
                "hme100k_ExpRate": hme["ExpRate"],
                "hme100k_TER": hme["TokenErrorRate"],
                "ValidLaTeX": university["ValidLaTeX"],
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote {}".format(output))
    print("| Model | Uni ExpRate | Uni <=1 | Uni <=2 | Uni TER | HME ExpRate | HME TER | Valid LaTeX |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {model} | {university_ExpRate:.4f} | {university_le_1:.4f} | {university_le_2:.4f} | "
            "{university_TER:.4f} | {hme100k_ExpRate:.4f} | {hme100k_TER:.4f} | {ValidLaTeX:.4f} |".format(**row)
        )


if __name__ == "__main__":
    main()

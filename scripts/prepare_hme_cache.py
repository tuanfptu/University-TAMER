"""Create compact PNG replay/validation caches from HME100K pickle files."""

import argparse
import csv
import pickle
import random
from pathlib import Path

import cv2

from tamer.university.image_io import write_image


def read_captions(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split()
            if parts:
                rows.append((parts[0], " ".join(parts[1:])))
    return rows


def export_split(source: Path, output: Path, name: str, count: int, seed: int) -> None:
    captions = read_captions(source / name / "caption.txt")
    rng = random.Random(seed)
    rng.shuffle(captions)
    selected = captions[:count]
    selected_names = {row[0] for row in selected}
    print("Loading {} (one-time cache construction)...".format(source / name / "images.pkl"))
    with (source / name / "images.pkl").open("rb") as stream:
        images = pickle.load(stream)
    image_dir = output / "images" / name
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, (sample_id, label) in enumerate(selected, start=1):
        image = images[sample_id]
        image_path = image_dir / (Path(sample_id).stem + ".png")
        if not write_image(image_path, image):
            raise IOError(str(image_path))
        manifest_rows.append(
            {
                "sample_id": "hme_" + Path(sample_id).stem,
                "image_path": image_path.relative_to(output).as_posix(),
                "label": label,
                "category": "hme100k",
                "source": "hme100k",
                "token_count": len(label.split()),
            }
        )
        if index % 2500 == 0:
            print("Exported {}/{} {} images".format(index, len(selected), name))
    del images
    manifest_path = output / ("replay.csv" if name == "train" else "validation.csv")
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hme100k", required=True)
    parser.add_argument("--output", default="data/hme_cache")
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    source = Path(args.hme100k).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    export_split(source, output, "train", args.replay_size, args.seed)
    export_split(source, output, "test", args.validation_size, args.seed + 1)
    print("HME replay cache ready. Original HME100K files were not modified.")


if __name__ == "__main__":
    main()

"""Build trial-local features from the independent_v1 cache."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.v4_features.feature_families import FAMILIES
from src.v4_features.workflow import build_subject_cache, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-cache", type=Path, required=True)
    parser.add_argument("--window-sec", type=int, choices=(2, 4), required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES,
                        default=["F1H", "F1W", "F1B", "F2", "F2p", "F3"])
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 31)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    families = list(dict.fromkeys(args.families))
    subjects = list(dict.fromkeys(args.subjects))
    if min(subjects) < 1:
        parser.error("Participant IDs must be positive")

    rows = []
    for subject in subjects:
        windows = build_subject_cache(args.data_root, args.out_cache, subject, args.window_sec,
                                      families, overwrite=args.overwrite)
        rows.append({"subject": subject, "windows": windows})
        print(f"P{subject:02d} {args.window_sec}s: {windows} windows", flush=True)
    scale_directory = args.out_cache / f"{args.window_sec}s"
    write_json(scale_directory / "complete.json", {
        "complete": True, "window_seconds": args.window_sec, "families": families, "participants": rows,
        "total_windows": sum(row["windows"] for row in rows),
    })


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
from typing import Optional

from run_real_data import plot_noisy_rank_shift, run_real_data_noisy_judge_experiment


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_DIR = BASE_DIR / "results"
NOISY_ARCHIVE_DIR = SUMMARY_DIR / "noisy_summaries"
DATASET_NAMES = ("chatbot_arena", "mtbench", "ultrafeedback", "in_house")


def load_noisy_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_noisy_summary(dataset_name: Optional[str]) -> dict:
    if dataset_name is not None:
        archive_path = NOISY_ARCHIVE_DIR / f"noisy_judge_summary_{dataset_name}.json"
        if not archive_path.exists():
            raise FileNotFoundError(f"missing noisy summary: {archive_path}")
        return load_noisy_summary(archive_path)

    merged = {}
    for current_dataset_name in DATASET_NAMES:
        archive_path = NOISY_ARCHIVE_DIR / f"noisy_judge_summary_{current_dataset_name}.json"
        if archive_path.exists():
            merged.update(load_noisy_summary(archive_path))
    if not merged:
        raise FileNotFoundError(f"no noisy summary files found in {NOISY_ARCHIVE_DIR}")
    return merged


def build_output_path(dataset_name: Optional[str], max_noisy_step: Optional[int]) -> Path:
    parts = ["noisy_exact_rank_match_rate"]
    if dataset_name is not None:
        parts.append(dataset_name)
    if max_noisy_step is not None:
        parts.append(f"through_{int(max_noisy_step)}")
    return SUMMARY_DIR / ("_".join(parts) + "_custom.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_NAMES)
    parser.add_argument("--max-noisy-step", type=int)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--use-existing-summary",
        action="store_true",
        help="Read existing noisy summary json files and only redraw plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.use_existing_summary:
        noisy_summary = load_existing_noisy_summary(args.dataset)
    else:
        noisy_summary = run_real_data_noisy_judge_experiment(
            random_seed=args.random_seed,
            dataset_name=args.dataset,
            max_noisy_step=args.max_noisy_step,
        )

    output_path = build_output_path(args.dataset, args.max_noisy_step)
    plot_noisy_rank_shift(noisy_summary, max_step=args.max_noisy_step, output_path=str(output_path))
    print(output_path)

"""Aggregate the pre-registered three-seed biomedical study."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "runs/biomedical_pilot"
OUT = ROOT / "runs/biomedical_confirmation"
SEEDS = (42, 43, 44)
GROUPS = ("L", "G", "S")
TASKS = (
    "mmlu_clinical_knowledge",
    "mmlu_professional_medicine",
    "mmlu_medical_genetics",
    "mmlu_anatomy",
    "mmlu_high_school_world_history",
)


def score(path: Path, task: str) -> float:
    result = json.loads(path.read_text())["results"][task]
    return 100.0 * result["acc,none"]


def result_path(seed: int, group: str) -> Path:
    if seed == 42:
        return PILOT / f"{group}.json"
    return OUT / f"{group}_seed{seed}.json"


def main() -> None:
    values = {
        str(seed): {
            group: {task: score(result_path(seed, group), task) for task in TASKS}
            for group in GROUPS
        }
        for seed in SEEDS
    }
    aggregate = {}
    for group in GROUPS:
        aggregate[group] = {}
        for task in TASKS:
            xs = np.array([values[str(seed)][group][task] for seed in SEEDS])
            aggregate[group][task] = {
                "mean": float(xs.mean()),
                "std": float(xs.std(ddof=1)),
            }
    paired = {}
    for comparison, left, right in (("G-S", "G", "S"), ("G-L", "G", "L")):
        paired[comparison] = {}
        for task in TASKS:
            ds = np.array(
                [values[str(seed)][left][task] - values[str(seed)][right][task] for seed in SEEDS]
            )
            paired[comparison][task] = {
                "per_seed": {str(seed): float(delta) for seed, delta in zip(SEEDS, ds)},
                "mean": float(ds.mean()),
                "std": float(ds.std(ddof=1)),
                "positive_seeds": int((ds > 0).sum()),
            }
    primary = "mmlu_clinical_knowledge"
    conclusion = {
        "primary": primary,
        "g_beats_s_all_seeds": paired["G-S"][primary]["positive_seeds"] == len(SEEDS),
        "g_beats_l_all_seeds": paired["G-L"][primary]["positive_seeds"] == len(SEEDS),
    }
    (OUT / "summary.json").write_text(
        json.dumps(
            {"seeds": SEEDS, "values": values, "aggregate": aggregate, "paired": paired,
             "conclusion": conclusion},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

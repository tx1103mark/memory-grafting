"""Aggregate biomedical teacher block ablation results."""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/biomedical_teacher_layer_study"
LAYERS = (4, 6, 8, 12)
SEEDS = (42, 43, 44)
TASKS = (
    "mmlu_clinical_knowledge",
    "mmlu_professional_medicine",
    "mmlu_medical_genetics",
    "mmlu_anatomy",
    "mmlu_high_school_world_history",
)
T_CRIT_DF2 = 4.302652729911275


def accuracy(path: Path, task: str) -> float:
    return 100.0 * json.loads(path.read_text())["results"][task]["acc,none"]


def l_path(seed: int) -> Path:
    if seed == 42:
        return ROOT / "runs/biomedical_pilot/L.json"
    return ROOT / f"runs/biomedical_confirmation/L_seed{seed}.json"


def stats(xs: list[float]) -> dict[str, object]:
    a = np.asarray(xs, dtype=float)
    sd = float(a.std(ddof=1))
    margin = T_CRIT_DF2 * sd / math.sqrt(len(a))
    return {
        "per_seed": {str(seed): float(value) for seed, value in zip(SEEDS, a)},
        "mean": float(a.mean()),
        "std": sd,
        "ci95": [float(a.mean() - margin), float(a.mean() + margin)],
        "positive_seeds": int((a > 0).sum()),
        "nonnegative_seeds": int((a >= 0).sum()),
    }


def main() -> None:
    values: dict[str, object] = {}
    for layer in LAYERS:
        layer_values = {}
        for seed in SEEDS:
            layer_values[str(seed)] = {
                group: {
                    task: accuracy(OUT / f"{group}_T{layer}_seed{seed}.json", task)
                    for task in TASKS
                }
                for group in ("G", "S")
            }
            layer_values[str(seed)]["L"] = {task: accuracy(l_path(seed), task) for task in TASKS}
        values[str(layer)] = layer_values

    aggregate = {}
    for layer in LAYERS:
        aggregate[str(layer)] = {}
        for task in TASKS:
            g = [values[str(layer)][str(seed)]["G"][task] for seed in SEEDS]
            s = [values[str(layer)][str(seed)]["S"][task] for seed in SEEDS]
            l = [values[str(layer)][str(seed)]["L"][task] for seed in SEEDS]
            aggregate[str(layer)][task] = {
                "G": stats(g),
                "S": stats(s),
                "L": stats(l),
                "G-S": stats([x - y for x, y in zip(g, s)]),
                "G-L": stats([x - y for x, y in zip(g, l)]),
            }

    alignment = {}
    for layer in LAYERS:
        path = ROOT / ("runs/biomedical_alignment" if layer == 12 else f"runs/biomedical_alignment_T{layer}") / "results.json"
        report = json.loads(path.read_text())
        alignment[str(layer)] = {
            "selected_transform": report["selected_transform"],
            "selected_objective": report["selected_objective"],
            "heldout": report["final_test"]["test"],
        }

    primary = "mmlu_clinical_knowledge"
    ranking = sorted(
        LAYERS,
        key=lambda layer: (
            aggregate[str(layer)][primary]["G-S"]["mean"],
            aggregate[str(layer)][primary]["G-L"]["mean"],
        ),
        reverse=True,
    )
    output = {
        "protocol": {"teacher_blocks": LAYERS, "student_block": 1, "seeds": SEEDS, "primary": primary},
        "alignment": alignment,
        "values": values,
        "aggregate": aggregate,
        "ranking_by_primary_g_minus_s_then_g_minus_l": ranking,
    }
    (OUT / "summary.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

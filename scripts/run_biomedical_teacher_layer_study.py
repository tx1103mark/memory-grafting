"""Biomedical teacher-source block ablation with fixed student block 1."""
import concurrent.futures
import argparse
import fcntl
import json
import os
import queue
import shutil
import subprocess
import time
from pathlib import Path

from scripts.run_layer_study import call

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/biomedical_teacher_layer_study"
LOG = ROOT / "logs/biomedical_teacher_layer_study"
NEW_LAYERS = (4, 6, 8)
SEEDS = (42, 43, 44)
GPUS = (2, 3, 4, 5, 7)
TASKS = (
    "mmlu_clinical_knowledge,mmlu_professional_medicine,"
    "mmlu_medical_genetics,mmlu_anatomy,mmlu_high_school_world_history"
)


def wait_free(gpu: int, minimum: int = 30000) -> None:
    while int(
        subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    ) < minimum:
        time.sleep(30)


def prepare_layer(layer: int, gpu: int) -> None:
    raw = ROOT / f"memory_biomedical/T{layer}"
    alignment = ROOT / f"runs/biomedical_alignment_T{layer}"
    aligned = ROOT / f"memory_biomedical_aligned/T{layer}"
    wait_free(gpu)
    if not (raw / "manifest.json").exists():
        call(
            "scripts.build_memory",
            ["--teacher-block", layer, "--data-dir", "data/biomedical", "--output-dir", raw, "--no-random"],
            gpu,
            LOG / f"T{layer}.build.log",
        )
    if not (alignment / "aligned_table.pt").exists():
        call(
            "scripts.run_alignment_study",
            [
                "--data-dir", "data/biomedical",
                "--memory-dir", raw,
                "--output-dir", alignment,
                "--samples", 60000,
                "--student-block", 1,
                "--teacher-block", layer,
            ],
            gpu,
            LOG / f"T{layer}.alignment.log",
        )
    if not (aligned / "manifest.json").exists():
        call(
            "scripts.install_aligned_memory",
            ["--source", alignment, "--base-memory", raw, "--dest", aligned],
            gpu,
            LOG / f"T{layer}.install.log",
        )


def copy_t12_references() -> None:
    for seed in SEEDS:
        source_dir = ROOT / ("runs/biomedical_pilot" if seed == 42 else "runs/biomedical_confirmation")
        for group in ("G", "S"):
            source_name = f"{group}.json" if seed == 42 else f"{group}_seed{seed}.json"
            destination = OUT / f"{group}_T12_seed{seed}.json"
            if not destination.exists():
                shutil.copy2(source_dir / source_name, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=list(NEW_LAYERS))
    args = parser.parse_args()
    layers = tuple(dict.fromkeys(args.layers))
    if not layers or any(layer < 1 for layer in layers):
        raise ValueError("Teacher layers must be positive one-based block numbers")
    if len(layers) > len(GPUS):
        raise ValueError(f"At most {len(GPUS)} new layers can be prepared in one run")
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT / "complete").unlink(missing_ok=True)

    preparation_gpus = GPUS[:len(layers)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(preparation_gpus)) as pool:
        list(pool.map(prepare_layer, layers, preparation_gpus))
    copy_t12_references()

    jobs: queue.Queue[tuple[int, int, str]] = queue.Queue()
    for seed in SEEDS:
        for layer in layers:
            for group in ("G", "S"):
                jobs.put((seed, layer, group))

    def worker(gpu: int) -> list[dict[str, object]]:
        errors = []
        while True:
            try:
                seed, layer, group = jobs.get_nowait()
            except queue.Empty:
                return errors
            label = f"{group}_T{layer}_seed{seed}"
            run = ROOT / "runs" / f"biomedical_layer_{label}"
            try:
                wait_free(gpu)
                if not (run / "complete.json").exists():
                    args: list[object] = [
                        "--group", group,
                        "--tokens", 2_000_000,
                        "--seed", seed,
                        "--micro-batch", 2,
                        "--accumulation", 2,
                        "--student-block", 1,
                        "--data-dir", "data/biomedical",
                        "--run-name", run.name,
                        "--memory-dir", f"memory_biomedical_aligned/T{layer}",
                        "--aligned-init", "low-lr",
                    ]
                    if (run / "last.pt").exists():
                        args += ["--resume", run / "last.pt"]
                    call("scripts.train", args, gpu, LOG / f"{label}.train.log")
                output = OUT / f"{label}.json"
                if not output.exists():
                    call(
                        "scripts.evaluate",
                        ["--checkpoint", run / "last.pt", "--tasks", TASKS, "--output", output],
                        gpu,
                        LOG / f"{label}.eval.log",
                    )
                if seed == 42:
                    diagnostic = OUT / f"{label}.diagnostic.json"
                    if not diagnostic.exists():
                        call(
                            "scripts.diagnose_memory",
                            ["--checkpoint", run / "last.pt", "--output", diagnostic],
                            gpu,
                            LOG / f"{label}.diagnostic.log",
                        )
                print("DONE", label, flush=True)
            except Exception as exc:
                errors.append({"seed": seed, "teacher_block": layer, "group": group, "error": repr(exc)})
                print("FAILED", label, repr(exc), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        errors = sum(pool.map(worker, GPUS), [])
    (OUT / "errors.json").write_text(json.dumps(errors, indent=2))
    if errors:
        raise RuntimeError("Inspect errors.json and rerun")
    call("scripts.summarize_biomedical_teacher_layers", [], 2, LOG / "summary.log")
    (OUT / "complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()

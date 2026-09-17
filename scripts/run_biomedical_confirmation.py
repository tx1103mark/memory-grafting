"""Run the pre-registered multi-seed biomedical confirmation study."""
import concurrent.futures
import fcntl
import json
import os
import queue
import subprocess
import time
from pathlib import Path

from scripts.run_layer_study import call

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/biomedical_confirmation"
LOG = ROOT / "logs/biomedical_confirmation"
TASKS = (
    "mmlu_clinical_knowledge,mmlu_professional_medicine,"
    "mmlu_medical_genetics,mmlu_anatomy,mmlu_high_school_world_history"
)


def wait_free(gpu: int) -> None:
    while int(
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    ) < 30000:
        time.sleep(30)


def main() -> None:
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    jobs: queue.Queue[tuple[int, str]] = queue.Queue()
    # Keep the informative G/S pair early while still fitting L on both GPUs.
    for seed in (43, 44):
        for group in ("G", "S", "L"):
            jobs.put((seed, group))

    def worker(gpu: int) -> list[dict[str, object]]:
        errors = []
        while True:
            try:
                seed, group = jobs.get_nowait()
            except queue.Empty:
                return errors
            run = ROOT / "runs" / f"biomedical_{group}_seed{seed}"
            label = f"{group}_seed{seed}"
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
                        "--snapshot-tokens", 500_000, 1_000_000,
                    ]
                    if group != "L":
                        args += [
                            "--memory-dir", "memory_biomedical_aligned/T12",
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
                if group != "L":
                    diagnostic = OUT / f"{label}.diagnostic.json"
                    if not diagnostic.exists():
                        call(
                            "scripts.diagnose_memory",
                            ["--checkpoint", run / "last.pt", "--output", diagnostic],
                            gpu,
                            LOG / f"{label}.diagnostic.log",
                        )
                print("DONE", label, flush=True)
            except Exception as exc:  # keep the other GPU useful and make reruns resumable
                errors.append({"seed": seed, "group": group, "error": repr(exc)})
                print("FAILED", label, repr(exc), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        errors = sum(pool.map(worker, (2, 3)), [])
    (OUT / "errors.json").write_text(json.dumps(errors, indent=2))
    if errors:
        raise RuntimeError("Inspect errors.json and rerun")
    call("scripts.summarize_biomedical_confirmation", [], 2, LOG / "summary.log")
    (OUT / "complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()

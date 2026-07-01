import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# Paths
NEXARA_DIR = Path("/home/zeus/Nexara")
CHECKPOINTS_DIR = NEXARA_DIR / "checkpoints" / "stage1"
SCRIPTS_DIR = NEXARA_DIR / "scripts"
PYTHON_BIN = "/home/zeus/miniconda3/envs/cloudspace/bin/python"

MILESTONES = [30000, 40000, 50000, 75000, 90000, 100000]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    os.chdir(NEXARA_DIR)
    log("Remote monitor started.")
    processed_milestones = set()

    # Pre-populate already processed milestones from existing JSON files
    if CHECKPOINTS_DIR.exists():
        for m in MILESTONES:
            results_file = CHECKPOINTS_DIR / f"benchmark_results_{m // 1000}k.json"
            if results_file.exists():
                log(f"Milestone {m} already has benchmark results. Skipping.")
                processed_milestones.add(m)

    while True:
        try:
            # 1. Check for step_*.pt files to preserve milestone checkpoints
            if CHECKPOINTS_DIR.exists():
                for p in CHECKPOINTS_DIR.glob("step_*.pt"):
                    name = p.name
                    if name.endswith("_milestone.pt"):
                        continue
                    step_str = name.replace("step_", "").replace(".pt", "")
                    if not step_str.isdigit():
                        continue
                    step = int(step_str)

                    if step in MILESTONES and step not in processed_milestones:
                        log(f"Detected checkpoint for milestone step {step}: {p}")

                        # Wait 10 seconds to make sure it's fully written
                        time.sleep(10)

                        milestone_ckpt = CHECKPOINTS_DIR / f"step_{step}_milestone.pt"
                        log(f"Copying {p} to {milestone_ckpt}...")
                        shutil.copy2(p, milestone_ckpt)

                        # Run benchmark
                        output_json = CHECKPOINTS_DIR / f"benchmark_results_{step // 1000}k.json"
                        log(f"Running benchmark for step {step}...")
                        cmd = [
                            PYTHON_BIN,
                            str(SCRIPTS_DIR / "benchmark_generation.py"),
                            "--checkpoint",
                            str(milestone_ckpt),
                            "--output",
                            str(output_json),
                        ]
                        res = subprocess.run(cmd, capture_output=True, text=True)
                        if res.returncode == 0:
                            log(
                                f"Benchmark completed successfully for step {step}. Saved to {output_json}"
                            )
                            processed_milestones.add(step)
                        else:
                            log(f"Benchmark failed for step {step} with error:\n{res.stderr}")

            # 2. Check if the training process is still running
            ps_check = subprocess.run(["pgrep", "-f", "scripts/train_long.py"], capture_output=True)
            training_running = ps_check.returncode == 0

            final_pt = CHECKPOINTS_DIR / "final.pt"
            latest_pt = CHECKPOINTS_DIR / "latest.pt"

            # If training has stopped
            if not training_running:
                log("Training process is not running.")

                # Check if we should conclude (either final.pt exists, or we reached/passed the max steps)
                # Let's wait a few seconds in case it just finished and is flushing files
                time.sleep(15)

                # Verify if final.pt or latest.pt exists
                active_ckpt = None
                if final_pt.exists():
                    active_ckpt = final_pt
                elif latest_pt.exists():
                    active_ckpt = latest_pt

                if active_ckpt:
                    log(f"Training ended. Active checkpoint: {active_ckpt}")

                    # Read the final step count using a simple inline python script
                    py_cmd = [
                        PYTHON_BIN,
                        "-c",
                        f"import torch; ckpt = torch.load('{active_ckpt}', map_location='cpu'); print(ckpt.get('step', 0))",
                    ]
                    step_res = subprocess.run(py_cmd, capture_output=True, text=True)
                    last_step = 0
                    if step_res.returncode == 0:
                        try:
                            last_step = int(step_res.stdout.strip())
                            log(f"Detected last step: {last_step}")
                        except ValueError:
                            pass

                    if last_step > 0:
                        # Copy the final/latest checkpoint to a milestone file
                        milestone_ckpt = CHECKPOINTS_DIR / f"step_{last_step}_milestone.pt"
                        if not milestone_ckpt.exists():
                            log(f"Copying final checkpoint {active_ckpt} to {milestone_ckpt}...")
                            shutil.copy2(active_ckpt, milestone_ckpt)

                        # Run the benchmark for this final step if not already done
                        output_json = (
                            CHECKPOINTS_DIR / f"benchmark_results_{last_step // 1000}k.json"
                        )
                        if not output_json.exists():
                            log(f"Running final benchmark for step {last_step}...")
                            cmd = [
                                PYTHON_BIN,
                                str(SCRIPTS_DIR / "benchmark_generation.py"),
                                "--checkpoint",
                                str(milestone_ckpt),
                                "--output",
                                str(output_json),
                            ]
                            res = subprocess.run(cmd, capture_output=True, text=True)
                            if res.returncode == 0:
                                log(
                                    f"Final benchmark completed successfully. Saved to {output_json}"
                                )
                            else:
                                log(f"Final benchmark failed with error:\n{res.stderr}")

                    # 3. Export weights from best.pt (or active_ckpt if best.pt doesn't exist)
                    export_source = CHECKPOINTS_DIR / "best.pt"
                    if not export_source.exists():
                        export_source = active_ckpt

                    log(f"Exporting clean weights from {export_source}...")
                    export_cmd = [
                        PYTHON_BIN,
                        str(SCRIPTS_DIR / "export_checkpoint.py"),
                        "--checkpoint",
                        str(export_source),
                        "--output-pt",
                        str(CHECKPOINTS_DIR / "clean_model.pt"),
                        "--output-json",
                        str(CHECKPOINTS_DIR / "config.json"),
                    ]
                    res_export = subprocess.run(export_cmd, capture_output=True, text=True)
                    log(f"Export result:\n{res_export.stdout}\n{res_export.stderr}")

                    # 4. Trigger shutdown to stop billing
                    log("All tasks completed. Shutting down the remote studio...")
                    time.sleep(5)
                    subprocess.run(["sudo", "shutdown", "-h", "now"])
                    break
                else:
                    log("Training process is not running and no checkpoint found. Waiting...")

        except Exception as e:
            log(f"Error in monitor loop: {e}")

        time.sleep(30)


if __name__ == "__main__":
    main()

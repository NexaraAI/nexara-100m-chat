import os
import sys
import time
import subprocess
import argparse


def check_pid(pid):
    """Check if process with pid is running."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pid", type=int, required=True, help="PID of the Stage B process to wait for."
    )
    args = parser.parse_args()

    pid = args.pid
    print(f"Waiting for Stage B process (PID {pid}) to finish...", flush=True)

    # Loop and check every 60 seconds
    while check_pid(pid):
        time.sleep(60)

    print(f"Stage B process (PID {pid}) has finished. Preparing Stage C...", flush=True)
    time.sleep(5)  # Wait a bit to ensure files are flushed/closed

    # Determine checkpoint to resume from
    possible_checkpoints = [
        "checkpoints/stage1_100m/stage_b_10000/best.pt",
        "checkpoints/stage1_100m/stage_b_10000/latest.pt",
        "checkpoints/stage1_100m/stage_b_10000/final.pt",
    ]
    checkpoint_path = None
    for cp in possible_checkpoints:
        if os.path.exists(cp):
            checkpoint_path = cp
            break

    if not checkpoint_path:
        # Check if there are other checkpoints in the directory
        import glob

        pts = glob.glob("checkpoints/stage1_100m/stage_b_10000/*.pt")
        if pts:
            checkpoint_path = pts[0]
            print(
                f"Warning: Target checkpoints not found. Defaulting to first match: {checkpoint_path}",
                flush=True,
            )

    if not checkpoint_path:
        print(
            "Error: No checkpoints found in checkpoints/stage1_100m/stage_b_10000/!",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    print(f"Found checkpoint to resume from: {checkpoint_path}", flush=True)

    # Read base config nexara_tiny_100m.toml
    config_path = "configs/nexara_tiny_100m.toml"
    if not os.path.exists(config_path):
        print(f"Error: Base config not found at {config_path}", file=sys.stderr, flush=True)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_text = f.read()

    # Create Stage C config text with resume_from set
    stage_c_config_text = config_text.replace(
        'resume_from = ""', f'resume_from = "{checkpoint_path}"'
    )

    stage_c_config_path = "configs/nexara_tiny_100m_stage_c.toml"
    with open(stage_c_config_path, "w", encoding="utf-8") as f:
        f.write(stage_c_config_text)
    print(f"Wrote Stage C configuration to {stage_c_config_path}", flush=True)

    # Launch Stage C pretraining in a new tmux session
    python_bin = "/home/zeus/miniconda3/envs/cloudspace/bin/python"
    log_dir = "logs/stage1_100m/stage_c_100000"
    output_dir = "checkpoints/stage1_100m/stage_c_100000"

    cmd = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "nexara_100m_stage_c",
        f"cd /home/zeus/Nexara && {python_bin} scripts/train_stage1.py --config {stage_c_config_path} --max-steps 100000 --output-dir {output_dir} --log-dir {log_dir}",
    ]

    print(f"Launching Stage C pretraining via: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print("Stage C launched successfully in tmux session 'nexara_100m_stage_c'!", flush=True)


if __name__ == "__main__":
    main()

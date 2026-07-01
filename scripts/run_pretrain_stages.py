import subprocess
import os
import sys


def run_command(cmd):
    print(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if res.returncode != 0:
        print(f"Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    python_bin = "/home/zeus/miniconda3/envs/cloudspace/bin/python"

    # 1. Run Stage A (1,000 steps)
    print("=== Starting Stage A Pretraining (1,000 steps) ===")
    stage_a_cmd = [
        python_bin,
        "scripts/train_stage1.py",
        "--config",
        "configs/nexara_tiny_100m.toml",
        "--max-steps",
        "1000",
        "--output-dir",
        "checkpoints/stage1_100m/stage_a_1000",
        "--log-dir",
        "logs/stage1_100m/stage_a_1000",
    ]
    run_command(stage_a_cmd)

    # Check if Stage A produced a checkpoint
    checkpoint_path = "checkpoints/stage1_100m/stage_a_1000/best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Stage A checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print("=== Stage A Pretraining Completed successfully! ===")

    # 2. Prepare Stage B Configuration
    print("Preparing Stage B configuration...")
    with open("configs/nexara_tiny_100m.toml", "r", encoding="utf-8") as f:
        config_text = f.read()

    # Replace resume_from = "" with the checkpoint path
    # Usually it is written as resume_from = ""
    stage_b_config_text = config_text.replace(
        'resume_from = ""', f'resume_from = "{checkpoint_path}"'
    )

    stage_b_config_path = "configs/nexara_tiny_100m_stage_b.toml"
    with open(stage_b_config_path, "w", encoding="utf-8") as f:
        f.write(stage_b_config_text)
    print(f"Wrote Stage B configuration to {stage_b_config_path}")

    # 3. Run Stage B (10,000 steps)
    print("=== Starting Stage B Pretraining (10,000 steps) ===")
    stage_b_cmd = [
        python_bin,
        "scripts/train_stage1.py",
        "--config",
        stage_b_config_path,
        "--max-steps",
        "10000",
        "--output-dir",
        "checkpoints/stage1_100m/stage_b_10000",
        "--log-dir",
        "logs/stage1_100m/stage_b_10000",
    ]
    run_command(stage_b_cmd)
    print("=== Stage B Pretraining Completed successfully! ===")


if __name__ == "__main__":
    main()

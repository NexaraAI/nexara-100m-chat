import subprocess
import sys
import time
from lightning_sdk import Studio


def main():
    print("=== Starting Stage C Training ===", flush=True)
    cmd = [
        sys.executable,
        "scripts/train_stage1.py",
        "--config",
        "configs/nexara_tiny_100m_stage_c.toml",
    ]

    res = subprocess.run(cmd)

    print(f"Training process finished with exit code: {res.returncode}", flush=True)

    time.sleep(30)

    print("Stopping Lightning AI Studio to prevent further billing...", flush=True)
    try:
        s = Studio()
        s.stop()
    except Exception as e:
        print(f"Error stopping studio: {e}", flush=True)
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()

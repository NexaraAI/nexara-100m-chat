import subprocess
import sys
from pathlib import Path


def main():
    print("=== Launching SFT Overfit Validation (10-example CPU Check) ===")
    local_dir = Path(__file__).resolve().parent.parent
    config_path = local_dir / "configs" / "stage2_sft.toml"

    # Run SFT train with overfit flag
    cmd = [sys.executable, "scripts/train_sft.py", "--config", str(config_path), "--overfit"]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()

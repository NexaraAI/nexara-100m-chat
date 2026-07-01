import time
import subprocess
import os
import signal
from lightning_sdk import Studio

# 3.75 hours in seconds
WAIT_SECONDS = int(3.75 * 3600)  # 13500 seconds

print(
    f"Auto-stop script initialized. Waiting for {WAIT_SECONDS} seconds before shutting down...",
    flush=True,
)
time.sleep(WAIT_SECONDS)

print("Time limit reached. Stopping training process...", flush=True)

# Find training PID and terminate it
try:
    p = subprocess.Popen(["pgrep", "-f", "train_stage1.py"], stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    pids = [int(pid.strip()) for pid in stdout.split() if pid.strip()]
    for pid in pids:
        print(f"Sending SIGINT to training process PID {pid}...", flush=True)
        os.kill(pid, signal.SIGINT)
except Exception as e:
    print(f"Error terminating training process: {e}", flush=True)

# Wait 30 seconds for checkpoints to finish saving
time.sleep(30)

print("Shutting down Lightning AI Studio to prevent further billing...", flush=True)
try:
    s = Studio()
    s.stop()
except Exception as e:
    print(f"Error stopping studio via SDK: {e}", flush=True)

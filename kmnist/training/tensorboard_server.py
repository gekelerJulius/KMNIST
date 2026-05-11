import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CONFIG import TENSORBOARD
from kmnist.utils.paths import logs_dir


def main() -> None:
    log_dir = logs_dir()

    print(f"Waiting for Lightning logs in: {log_dir}")
    while not log_dir.exists():
        time.sleep(1)

    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
        "--load_fast=false",
        "--host",
        TENSORBOARD.host,
        "--port",
        str(TENSORBOARD.port),
    ]

    process = subprocess.Popen(command)
    print("TensorBoard started.")
    print(f"Open http://localhost:{TENSORBOARD.port}")

    def shutdown_handler(signum, frame):
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    return_code = process.wait()
    sys.exit(return_code)


if __name__ == "__main__":
    main()

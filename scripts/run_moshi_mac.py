"""Start Moshi on an Apple Silicon Mac (M4 and newer) via the MLX backend.

Default is the 4-bit Moshiko checkpoint (kyutai/moshiko-mlx-q4, about 4.8 GB).
That fits a 16 GB Mac. Use --q8 (about 8.2 GB) when unified memory is 24 GB
or more.

The web UI listens on http://127.0.0.1:8998 and uses the microphone.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument(
        "--q4",
        action="store_const",
        const=4,
        dest="bits",
        help="4-bit weights (default). About 4.8 GB.",
    )
    quant.add_argument(
        "--q8",
        action="store_const",
        const=8,
        dest="bits",
        help="8-bit weights. About 8.2 GB. Prefer this at 24 GB unified memory or more.",
    )
    parser.set_defaults(bits=4)
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Microphone CLI instead of the web UI.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8998)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system = platform.system()
    machine = platform.machine()
    if system != "Darwin" or machine != "arm64":
        print(
            "Moshi MLX runs on Apple Silicon only "
            f"(this process is {system} {machine}).\n"
            "On the Mac Pro M4, from this repo:\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            "  python scripts/run_moshi_mac.py"
        )
        return 2

    if args.cli:
        cmd = [sys.executable, "-m", "moshi_mlx.local", "-q", str(args.bits)]
    else:
        cmd = [
            sys.executable,
            "-m",
            "moshi_mlx.local_web",
            "-q",
            str(args.bits),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    print("running:", " ".join(cmd))
    if not args.cli:
        print(f"web UI: http://{args.host}:{args.port}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())

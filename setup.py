#!/usr/bin/env python3
"""
Cross‑platform bootstrapper for **tox_ko_classification**
========================================================
Run **once** from the project root (Linux **or** Windows):

    # Linux / WSL2
    $ python setup.py

    :: Windows PowerShell
    PS> python setup.py

The script will…
1. Validate **Python ≥ 3.11.8**
2. Detect an **NVIDIA GPU & CUDA** via *nvidia‑smi* (works on both OSes)
3. Install the matching **CUDA‑enabled** PyTorch *2.6* wheel
4. Install all remaining packages pinned in *requirements.txt* (torch excluded)
5. Verify PyTorch can see at least one GPU

Supported systems
-----------------
* **Windows 10/11** with CUDA 11.8 – 12.4 & NVIDIA driver ≥ 535
* **Linux** (Ubuntu 20.04+, WSL2 OK) with the same CUDA/driver combo

If no GPU or mismatched CUDA is found we fall back to CPU wheels so the rest of
the repo can still run.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
REQ_FILE = PROJECT_ROOT / "requirements.txt"
TORCH_VERSION = "2.6.0"  # keep in sync with requirements

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sh(cmd: str, check: bool = True) -> str:
    """Execute *cmd* and return stdout. Raises on failure when *check*=True."""
    print(f"\x1b[34m▶ {cmd}\x1b[0m")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout.strip()


def require_python(min_tuple=(3, 11, 8)) -> None:
    if sys.version_info < min_tuple:
        ver = ".".join(map(str, min_tuple))
        raise RuntimeError(f"Python {ver}+ required, found {platform.python_version()}")


def detect_cuda() -> Optional[str]:
    """Return CUDA version string like '12.1' or *None* if *nvidia‑smi* missing."""
    try:
        output = sh("nvidia-smi", check=False)
    except FileNotFoundError:
        return None

    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
    return match.group(1) if match else None


def pip_install_torch(cuda: Optional[str]) -> None:
    """Install torch/vision/audio **2.6** with or without CUDA support."""
    if cuda:
        cuda_tag = f"cu{cuda.replace('.', '')}"
        idx_url = f"https://download.pytorch.org/whl/{cuda_tag}"
        wheel_desc = f"CUDA {cuda}"
    else:
        idx_url = "https://download.pytorch.org/whl/cpu"
        wheel_desc = "CPU‑only"

    print(f"Installing PyTorch {TORCH_VERSION} ({wheel_desc}) …")
    sh(
        "pip install --upgrade --extra-index-url {} torch=={} torchvision torchaudio".format(
            idx_url, TORCH_VERSION
        )
    )


def parse_reqs(path: Path) -> list[str]:
    """Return requirements sans torch/vision/audio to avoid duplicates."""
    skip_pkgs = ("torch", "torchvision", "torchaudio")
    tokens: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if any(line.startswith(s) for s in skip_pkgs):
            continue
        tokens.append(line)
    return tokens


def install_requirements() -> None:
    pkgs = parse_reqs(REQ_FILE)
    if pkgs:
        print("Installing remaining dependencies …")
        sh(f"pip install --upgrade {' '.join(pkgs)}")


def verify_torch(cuda_expected: Optional[str]) -> None:
    import importlib

    torch = importlib.import_module("torch")
    print("\nPyTorch diagnostic:")
    print("  version       :", torch.__version__)
    print("  cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  gpu name      :", torch.cuda.get_device_name(0))
        print("  cuda runtime  :", torch.version.cuda)
        if cuda_expected and not torch.version.cuda.startswith(cuda_expected.split(".")[0]):
            print("\x1b[33m⚠ Possible CUDA version mismatch.\x1b[0m")


# ---------------------------------------------------------------------------
# Bootstrap flow
# ---------------------------------------------------------------------------

def main():
    os.chdir(PROJECT_ROOT)
    require_python()

    # Upgrade pip quietly
    sh("python -m pip install --upgrade pip", check=True)

    cuda_ver = detect_cuda()
    if cuda_ver:
        print(f"Detected NVIDIA GPU – CUDA {cuda_ver}")
    else:
        print("No compatible NVIDIA GPU detected; proceeding with CPU wheels.")

    pip_install_torch(cuda_ver)
    install_requirements()
    verify_torch(cuda_ver)

    print("\n✅  Setup complete. Try training with:\n   $ python train.py\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print("\x1b[31m✖ Setup failed:\x1b[0m", e)
        sys.exit(1)

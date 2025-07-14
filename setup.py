#!/usr/bin/env python
# setup.py  –  환경 부트스트랩 스크립트
import os, sys, subprocess, shutil, platform, re, pathlib, venv
from importlib.metadata import version as _v

MIN_PY = (3, 11, 8)
TORCH_VERSION = "2.6.0"
CUDA_WHEELS = {  # 최소 대응
    "12.2": "cu122",
    "12.1": "cu121",
    "12.0": "cu120",
    "11.8": "cu118",
    "11.7": "cu117",
}
def check_python():
    if sys.version_info < MIN_PY:
        sys.exit(f"❌ Python {MIN_PY[0]}.{MIN_PY[1]}.{MIN_PY[2]} 이상이 필요합니다.")

def run(cmd, **kw):
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd, **kw)

def detect_cuda():
    try:
        out = subprocess.check_output(["nvidia-smi", "-q"], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"CUDA Version\s+:\s+(\d+\.\d+)", out)
        if m:
            ver = m.group(1)
            tag = CUDA_WHEELS.get(ver)
            if tag:
                return ver, tag
            # fallback – 가장 근접한 버전 매핑
            for v,pytag in CUDA_WHEELS.items():
                if ver.startswith(v.split(".")[0]):
                    return ver, pytag
        print("⚠️  CUDA 버전 매핑을 찾지 못했습니다. CPU 전용으로 진행합니다.")
    except FileNotFoundError:
        print("ℹ️  NVIDIA GPU를 찾지 못했습니다.")
    return None, None

def pip_install(pkgs, extra_index=None):
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs
    if extra_index:
        cmd.extend(["--index-url", extra_index])
    run(cmd)

def main():
    check_python()
    os_type = platform.system()
    cuda_ver, torch_tag = detect_cuda()

    # 1) PyTorch
    if torch_tag:
        pip_install(
            [f"torch=={TORCH_VERSION}+{torch_tag}", "torchvision", "torchaudio"],
            extra_index="https://download.pytorch.org/whl/" + torch_tag
        )
    else:
        pip_install([f"torch=={TORCH_VERSION}+cpu"], extra_index="https://download.pytorch.org/whl/cpu")

    # 2) bitsandbytes
    if torch_tag:
        bb_pkg = "bitsandbytes" if os_type != "Windows" else "bitsandbytes-win"
        pip_install([bb_pkg])
    else:
        print("ℹ️  GPU가 없으므로 bitsandbytes 설치를 건너뜁니다.")

    # 3) 그 외 requirements
    reqs = pathlib.Path("requirements.txt").read_text().split()
    skip = {"torch", "bitsandbytes", "bitsandbytes-win"}
    filtered = [p for p in reqs if not any(p.startswith(s) for s in skip)]
    if filtered:
        pip_install(filtered)

    # 4) 검증
    import torch, textwrap
    print("\n✅ 설치 완료!")
    print(f"PyTorch {torch.__version__} | CUDA 지원: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    main()

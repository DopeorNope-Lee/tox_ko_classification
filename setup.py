"""
환경 설정 파일 (setup_environment.py)
====================================

이 파일은 Windows 11 환경에서 텍스트 분류 프로젝트를 위한 환경을 설정합니다.
학생들이 처음 프로젝트를 시작할 때 이 파일을 실행하여 필요한 환경을 구성할 수 있습니다.

이 버전에서는 샘플 데이터를 생성하지 않습니다.

사용법:
    python setup_environment.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Python 버전을 확인합니다."""
    print("Python 버전 확인 중...")
    version = sys.version_info
    # Python 3.11.8 버전을 사용하신다고 하셨지만, 호환성을 위해 3.8 이상을 확인합니다.
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("오류: Python 3.8 이상이 필요합니다.")
        print(f"현재 버전: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python 버전 확인 완료: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """요구사항 파일(requirements.txt)에 명시된 패키지들을 설치합니다."""
    print("필요한 패키지 설치 중...")
    
    # 설치할 패키지 목록 (버전 명시)
    packages = [
        "torch>=2.6.0",
        "transformers==4.53.0",
        "datasets==3.6.0",
        "pandas==2.3.0",
        "numpy==1.24.3",
        "peft==0.15.2",
        "bitsandbytes==0.46.0",
        "scikit-learn==1.7.0",
        "evaluate==0.4.4",
        "protobuf==3.20.3",
        "pydantic==2.11.7",
        "tqdm>=4.66.4",
        "sentencepiece",
        "rouge_score",
        "bert_score",
        "trl==0.19.1",
        "seaborn",
        "accelerate" # transformers, peft 등과 함께 효율적인 학습/추론을 위해 필요합니다.
    ]
    
    # torch 설치 시 CUDA 버전에 맞는 명령어를 사용할 수 있도록 별도 처리 (옵션)
    # 사용자가 직접 환경에 맞는 torch를 설치하는 것을 권장하지만, 스크립트에서는 일반 버전을 설치합니다.
    print("PyTorch, bitsandbytes 등 일부 패키지는 설치에 시간이 다소 소요될 수 있습니다.")

    for package in packages:
        try:
            print(f"설치 시도: {package}")
            # --no-cache-dir 옵션을 추가하여 캐시 문제 방지
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])
            print(f"'{package}' 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"오류: '{package}' 설치에 실패했습니다.")
            print(f"에러 메시지: {e}")
            print("인터넷 연결을 확인하거나, 관리자 권한으로 터미널을 실행해 보세요.")
            print("특정 패키지(예: torch, bitsandbytes)는 시스템 환경에 따라 별도 설치가 필요할 수 있습니다.")
            return False
    
    print("모든 패키지 설치가 성공적으로 완료되었습니다.")
    return True

def create_directories():
    """프로젝트에 필요한 디렉토리들을 생성합니다."""
    print("디렉토리 생성 중...")
    
    directories = [
        "data",        # 데이터 파일 저장 (폴더만 생성)
        "models",      # 학습된 모델 저장
        "checkpoints", # 학습 중 중간 저장 지점
        "logs"         # 로그 파일 저장
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"'{dir_name}' 디렉토리 생성/확인 완료")

def main():
    """메인 설정 함수"""
    print("=" * 60)
    print("텍스트 분류 프로젝트 환경 설정을 시작합니다...")
    print("=" * 60)
    
    # 1. Python 버전 확인
    if not check_python_version():
        return
    print("-" * 50)
    
    # 2. 디렉토리 생성
    create_directories()
    print("-" * 50)

    # 3. 필수 패키지 설치
    if not install_requirements():
        print("\n환경 설정 중 오류가 발생했습니다. 메시지를 확인하고 다시 시도해 주세요.")
        return
    
    print("=" * 60)
    print("환경 설정이 성공적으로 완료되었습니다!")
    print("=" * 60)
    print("\n다음 단계를 진행해 주세요:")
    print("  1. `data` 폴더에 학습시킬 데이터를 준비하세요.")
    print("  2. (필요시) README.md 파일을 읽고 프로젝트 구조를 파악하세요.")
    print("  3. `train.py`를 실행하여 모델을 학습시키세요.")
    print("  4. `inference.py`를 실행하여 새로운 텍스트를 분류해 보세요.")

if __name__ == "__main__":
    main()

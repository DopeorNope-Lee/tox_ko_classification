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

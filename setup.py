#!/usr/bin/env python3
"""
한국어 악성 댓글 분류 프로젝트 환경 설정 스크립트

이 스크립트는 Windows 11 환경의 GPU 내장 노트북에서 프로젝트 실행에 필요한 환경을 설정합니다.
GPU 설정, CUDA 확인, Windows 특화 환경 변수 설정을 수행합니다.

사용법:
    python setup.py

주의사항:
    - 이 스크립트는 train.py, inference.py, quantization.py 실행 전에 먼저 실행해야 합니다.
    - Windows 11 환경의 GPU 내장 노트북에 최적화되어 있습니다.
    - 노트북 GPU의 메모리 제약을 고려한 설정이 적용됩니다.
"""

import os
import sys
import torch
import platform
import subprocess
from pathlib import Path


def check_windows_environment():
    """Windows 환경 확인"""
    print("=== Windows 환경 확인 ===")
    
    if platform.system() != "Windows":
        print("⚠️  이 스크립트는 Windows 환경에 최적화되어 있습니다.")
        print("   다른 운영체제에서는 일부 기능이 제한될 수 있습니다.")
    else:
        print(f"✅ Windows 환경: {platform.system()} {platform.release()}")
        
        # Windows 버전 확인
        try:
            result = subprocess.run(['ver'], capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                print(f"   Windows 버전: {result.stdout.strip()}")
        except:
            pass
    
    print()


def check_python_version():
    """Python 버전 확인"""
    print("=== Python 버전 확인 ===")
    print(f"Python 버전: {sys.version}")
    
    # Python 3.8 이상 권장
    if sys.version_info < (3, 8):
        print("⚠️  경고: Python 3.8 이상을 권장합니다.")
    else:
        print("✅ Python 버전이 적절합니다.")
    print()


def check_system_info():
    """시스템 정보 확인"""
    print("=== 시스템 정보 ===")
    print(f"운영체제: {platform.system()} {platform.release()}")
    print(f"아키텍처: {platform.machine()}")
    print(f"프로세서: {platform.processor()}")
    
    # 메모리 정보 확인 (Windows)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"시스템 메모리: {memory.total / (1024**3):.1f}GB (사용 가능: {memory.available / (1024**3):.1f}GB)")
    except ImportError:
        print("메모리 정보: psutil 패키지가 설치되지 않아 확인할 수 없습니다.")
    
    print()


def setup_gpu_environment():
    """GPU 환경 설정 (노트북 최적화)"""
    print("=== GPU 환경 설정 (노트북 최적화) ===")
    
    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        # GPU 정보 출력
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU 사용 가능: {gpu_count}개 GPU 발견")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # 노트북 GPU 메모리 경고
            if gpu_memory < 4.0:
                print(f"   ⚠️  GPU 메모리가 적습니다. 배치 크기를 줄여야 할 수 있습니다.")
            elif gpu_memory < 8.0:
                print(f"   ⚠️  GPU 메모리가 제한적입니다. 모델 크기에 주의하세요.")
        
        # 노트북 GPU 최적화 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print(f"   기본 GPU로 설정: GPU 0")
        
        # CUDA 버전 확인
        cuda_version = torch.version.cuda
        if cuda_version:
            print(f"   CUDA 버전: {cuda_version}")
        
        # cuDNN 버전 확인
        if torch.backends.cudnn.is_available():
            print(f"   cuDNN 사용 가능: {torch.backends.cudnn.version()}")
        
        # 노트북 GPU 최적화 설정
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 디버깅을 위한 동기 실행
        print("   CUDA_LAUNCH_BLOCKING=1 (디버깅 모드)")
        
        print("✅ GPU 환경 설정 완료 (노트북 최적화)")
        
    else:
        print("⚠️  GPU를 찾을 수 없습니다. CPU를 사용합니다.")
        print("   - 학습 속도가 매우 느릴 수 있습니다.")
        print("   - 대용량 모델의 경우 메모리 부족이 발생할 수 있습니다.")
        print("   - GPU 드라이버를 업데이트하거나 CUDA를 설치해보세요.")
        
        # CPU 사용 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("✅ CPU 환경 설정 완료")
    
    print()


def setup_windows_environment_variables():
    """Windows 특화 환경 변수 설정"""
    print("=== Windows 특화 환경 변수 설정 ===")
    
    # PyTorch 관련 환경 변수 (Windows 최적화)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # 노트북 GPU 메모리 제약 고려
    print("   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 (노트북 최적화)")
    
    # Windows 특화 설정
    os.environ['OMP_NUM_THREADS'] = '1'  # OpenMP 스레드 수 제한
    print("   OMP_NUM_THREADS=1 (Windows 최적화)")
    
    # Transformers 관련 환경 변수
    os.environ['TRANSFORMERS_CACHE'] = './cache'
    print("   TRANSFORMERS_CACHE=./cache")
    
    # HuggingFace Hub 캐시 설정
    os.environ['HF_HOME'] = './cache'
    print("   HF_HOME=./cache")
    
    # 로그 레벨 설정 (경고 메시지 줄이기)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    print("   TOKENIZERS_PARALLELISM=false")
    
    # Windows 경로 설정
    os.environ['PATH'] = os.environ.get('PATH', '') + ';.\\cache'
    print("   PATH에 cache 디렉토리 추가")
    
    print("✅ Windows 특화 환경 변수 설정 완료")
    print()


def create_directories():
    """필요한 디렉토리 생성 (Windows 경로)"""
    print("=== 디렉토리 생성 ===")
    
    directories = [
        "cache",
        "model-checkpoints",
        "model-checkpoints\\kobert",
        "final-model",
        "bnb-4bit",
        "logs"  # 로그 파일 저장용
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}\\")
    
    print("✅ 디렉토리 생성 완료")
    print()


def check_dependencies():
    """필요한 라이브러리 확인 (Windows 최적화)"""
    print("=== 의존성 라이브러리 확인 ===")
    
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'pandas',
        'numpy',
        'evaluate',
        'peft',
        'accelerate',
        'bitsandbytes'
    ]
    
    # Windows에서 추가로 권장하는 패키지
    recommended_packages = [
        'psutil',  # 시스템 정보 확인
        'tqdm'     # 진행률 표시
    ]
    
    missing_packages = []
    missing_recommended = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (설치 필요)")
            missing_packages.append(package)
    
    for package in recommended_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} (권장)")
        except ImportError:
            print(f"   ⚠️  {package} (권장, 설치 안됨)")
            missing_recommended.append(package)
    
    if missing_packages:
        print(f"\n⚠️  다음 필수 패키지들이 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n다음 명령어로 설치하세요:")
        print("   pip install -r requirements.txt")
        print()
    else:
        print("\n✅ 모든 필수 패키지가 설치되어 있습니다.")
        
        if missing_recommended:
            print(f"\n💡 다음 권장 패키지들을 설치하면 더 나은 경험을 할 수 있습니다:")
            for package in missing_recommended:
                print(f"   - {package}")
            print("   pip install psutil tqdm")
            print()
    
    return len(missing_packages) == 0


def test_basic_functionality():
    """기본 기능 테스트 (Windows 최적화)"""
    print("=== 기본 기능 테스트 ===")
    
    try:
        # PyTorch 테스트
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        print("   ✅ PyTorch 기본 연산")
        
        # GPU 테스트 (GPU가 있는 경우)
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = x_gpu + y_gpu
            print("   ✅ GPU 연산")
            
            # GPU 메모리 사용량 확인
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"   GPU 메모리 사용량: {gpu_memory_allocated:.1f}MB")
        
        # Transformers 테스트
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)
        tokens = tokenizer("테스트 문장", return_tensors="pt")
        print("   ✅ Transformers 토크나이저")
        
        # Windows 경로 테스트
        test_file = Path("cache") / "test.txt"
        test_file.write_text("test")
        test_file.unlink()  # 테스트 파일 삭제
        print("   ✅ Windows 파일 시스템")
        
        print("✅ 기본 기능 테스트 완료")
        
    except Exception as e:
        print(f"   ❌ 테스트 실패: {e}")
        return False
    
    print()
    return True


def print_windows_usage_instructions():
    """Windows 환경 사용법 안내"""
    print("=== Windows 환경 사용법 안내 ===")
    print("환경 설정이 완료되었습니다! 이제 다음 순서로 프로젝트를 실행할 수 있습니다:")
    print()
    print("1. 모델 학습:")
    print("   python train.py")
    print("   (노트북 GPU 메모리 제약으로 인해 배치 크기를 줄여야 할 수 있습니다)")
    print()
    print("2. 모델 양자화 (선택사항, 메모리 절약):")
    print("   python quantization.py")
    print("   (노트북에서는 양자화를 권장합니다)")
    print()
    print("3. 모델 추론:")
    print("   python inference.py")
    print()
    print("💡 Windows 노트북 최적화 팁:")
    print("   - GPU 메모리가 부족하면 train.py의 BATCH_SIZE를 줄이세요")
    print("   - 학습 중에는 다른 프로그램을 종료하여 GPU 메모리를 확보하세요")
    print("   - 양자화된 모델을 사용하면 메모리 사용량을 크게 줄일 수 있습니다")
    print("   - 각 스크립트의 설정값은 파일 하단에서 수정할 수 있습니다")
    print()


def main():
    """메인 함수"""
    print("🚀 한국어 악성 댓글 분류 프로젝트 환경 설정 (Windows 11 + GPU 노트북)")
    print("=" * 70)
    print()
    
    # Windows 환경 확인
    check_windows_environment()
    
    # Python 버전 확인
    check_python_version()
    
    # 시스템 정보 확인
    check_system_info()
    
    # GPU 환경 설정
    setup_gpu_environment()
    
    # Windows 특화 환경 변수 설정
    setup_windows_environment_variables()
    
    # 디렉토리 생성
    create_directories()
    
    # 의존성 확인
    dependencies_ok = check_dependencies()
    
    # 기본 기능 테스트
    if dependencies_ok:
        test_ok = test_basic_functionality()
    else:
        test_ok = False
    
    print("=" * 70)
    
    if dependencies_ok and test_ok:
        print("🎉 Windows 환경 설정이 성공적으로 완료되었습니다!")
        print_windows_usage_instructions()
    else:
        print("⚠️  환경 설정에 문제가 있습니다.")
        print("의존성 패키지를 설치하고 다시 실행해주세요.")
        print()
        print("설치 명령어:")
        print("   pip install -r requirements.txt")
        print("   pip install psutil tqdm  # 권장 패키지")
    
    return 0 if dependencies_ok and test_ok else 1


if __name__ == "__main__":
    exit(main()) 
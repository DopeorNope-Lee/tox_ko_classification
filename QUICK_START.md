# 빠른 시작 가이드 (Quick Start Guide)

이 가이드는 한국어 악성 댓글 분류 모델을 실행하는 방법을 설명합니다.

## 사전 요구사항

- **Python**: 3.8 이상
- **GPU**: CUDA 지원 GPU (권장, 최소 4GB VRAM)
- **RAM**: 최소 8GB
- **저장공간**: 최소 2GB

### 운영체제
- Windows 10/11
- macOS 10.15 이상
- Ubuntu 18.04 이상

## 시작하기

### 1. 환경 설정 (1분)

```bash
# 저장소 클론
git clone <repository-url>
cd tox_ko_classification

# 가상환경 생성 및 활성화
python -m venv tox_env
source tox_env/bin/activate  # Linux/Mac
# 또는
tox_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 학습 (3-5분)

```bash
# 기본 설정으로 학습 시작
python train.py

# 또는 커스텀 설정 (더 빠른 학습)
python train.py --epochs 10 --batch_size 64
```

**학습 과정:**
- 데이터 로딩 및 전처리
- KoBERT 모델 설정 및 LoRA 적용
- 학습 진행 (약 20분, GPU 사용시)
- 모델이 `model-checkpoints/kobert/`에 저장됨

### 3. 추론 테스트 (30초)

```bash
# 단일 텍스트 예측
python inference.py --text "안녕하세요! 좋은 하루 되세요."

# 파일에서 여러 텍스트 예측
python inference.py --file examples/test_texts.txt

# 대화형 모드
python inference.py --interactive
```

## 예상 결과

### 학습 완료 시
```
=== 최종 평가 결과 ===
평가 손실: 0.324
정확도: 0.882
F1 점수: 0.876
```

### 추론 결과 예시
```
입력 텍스트: 안녕하세요! 좋은 하루 되세요.
예측 결과: none
신뢰도: 0.923

입력 텍스트: 당신은 정말 멍청한 사람이에요.
예측 결과: toxic
신뢰도: 0.891
```

## 선택사항: 모델 양자화

학습이 완료되면 모델을 양자화하여 크기를 줄일 수 있습니다.

```bash
# 양자화 실행
python quantization.py --checkpoint_path model-checkpoints/kobert/checkpoint-1100

# 양자화된 모델로 추론
python inference.py --text "테스트 문장" --model_path bnb-4bit
```

## 문제 해결

### 일반적인 오류

#### 1. CUDA 메모리 부족
```bash
# 배치 크기 줄이기
python train.py --batch_size 32
```

#### 2. 모델 로딩 실패
```bash
# 모델 경로 확인
ls model-checkpoints/kobert/
ls bnb-4bit/
```

#### 3. 의존성 설치 오류
```bash
# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"
```

## 다음 단계

빠른 시작을 완료했다면 다음 가이드들을 참조하세요:

- **[학습 가이드](TRAINING_GUIDE.md)**: 상세한 학습 과정과 하이퍼파라미터 튜닝
- **[양자화 가이드](QUANTIZATION_GUIDE.md)**: 모델 최적화 방법
- **[사용법 예시](USAGE_EXAMPLES.md)**: 웹 서비스, 배치 처리 등 고급 활용법

## 성능 최적화 팁

### GPU 메모리 최적화
```bash
# 더 작은 배치 크기 사용
python train.py --batch_size 32 --gradient_accumulation_steps 4
```

### 빠른 학습
```bash
# 더 적은 에포크로 학습
python train.py --epochs 5 --learning_rate 1e-4
```
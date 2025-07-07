# 한국어 악성 댓글 분류 모델 (Korean Toxic Comment Classification)

## 프로젝트 목적

이 프로젝트는 한국어 댓글을 분석하여 악성 댓글(toxic)과 일반 댓글(none)을 분류하는 머신러닝 모델을 구현한 교육용 자료입니다. 

### 주요 기능
- 한국어 댓글의 악성 여부 자동 분류
- KoBERT 기반의 fine-tuning 모델
- LoRA(Low-Rank Adaptation)를 활용한 효율적인 학습
- 4-bit 양자화를 통한 모델 최적화

## 프로젝트 구조

```
tox_ko_classification/
├── README.md                           # 프로젝트 개요
├── QUICK_START.md                      # 빠른 시작 가이드
├── TRAINING_GUIDE.md                   # 상세 학습 가이드
├── QUANTIZATION_GUIDE.md               # 양자화 가이드
├── USAGE_EXAMPLES.md                   # 사용법 예시
├── requirements.txt                    # 의존성 패키지 목록
├── train.py                           # 모델 학습 스크립트
├── quantization.py                    # 모델 양자화 스크립트
├── inference.py                       # 추론 스크립트
├── korean-malicious-comments-dataset/  # 데이터셋 폴더
│   ├── Dataset.csv                    # 원본 데이터
│   ├── README.md                      # 데이터셋 상세 설명
│   └── LICENSE                        # 라이선스
├── examples/                          # 예시 파일들
│   └── test_texts.txt                # 테스트용 텍스트
└── results/                          # 결과 및 성능 분석
    └── README.md                     # 학습 결과 요약
```

## 빠른 시작

### 1. 환경 설정
```bash
# 저장소 클론
git clone <repository-url>
cd tox_ko_classification

# 가상환경 생성 (권장)
python -m venv tox_env
source tox_env/bin/activate  # Linux/Mac
# 또는
tox_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 학습
```bash
# 기본 설정으로 학습
python train.py

# 커스텀 파라미터로 학습
python train.py --epochs 20 --batch_size 128 --learning_rate 5e-5
```

### 3. 모델 양자화
```bash
# 학습된 모델을 4-bit로 양자화
python quantization.py --checkpoint_path model-checkpoints/kobert/checkpoint-1100
```

### 4. 추론 테스트
```bash
# 단일 텍스트 예측
python inference.py --text "안녕하세요! 좋은 하루 되세요."

# 파일에서 여러 텍스트 예측
python inference.py --file examples/test_texts.txt

# 대화형 모드
python inference.py --interactive
```

## 학습 결과

### 모델 성능 비교
| 모델 | 정확도 (%) | F1 점수 | 메모리 사용량 |
|------|------------|---------|---------------|
| KcBERT | **90.6** | 0.901 | ~420MB |
| **KoBERT + LoRA** | **88.2** | 0.876 | ~105MB |
| Attention Bi-LSTM | 85.8 | 0.852 | ~50MB |

### 현재 구현 모델
- **KoBERT + LoRA**: 88.2% 정확도
- **4-bit 양자화**: 메모리 사용량 75% 감소

## 주요 스크립트 설명

### `train.py`
- 데이터 로딩 및 전처리
- KoBERT 모델 설정 및 LoRA 적용
- 학습 과정 모니터링 및 체크포인트 저장

### `quantization.py`
- 학습된 LoRA 모델 로드 및 병합
- 4-bit 양자화 적용
- 최적화된 모델 저장

### `inference.py`
- 학습된 모델을 사용한 예측
- 단일/배치/대화형 모드 지원
- 신뢰도 점수 제공

## 실행 환경

### 시스템 요구사항
- Python 3.8+
- CUDA 지원 GPU (권장)
- 최소 8GB RAM

### 주요 라이브러리 버전
- PyTorch 2.1.0
- Transformers 4.35.0
- PEFT 0.6.0
- Datasets 2.14.0

## 상세 가이드

- **[빠른 시작 가이드](QUICK_START.md)**: 5분 만에 프로젝트 실행하기
- **[학습 가이드](TRAINING_GUIDE.md)**: 모델 학습 과정 상세 설명
- **[양자화 가이드](QUANTIZATION_GUIDE.md)**: 모델 최적화 방법
- **[사용법 예시](USAGE_EXAMPLES.md)**: 다양한 활용 방법

## 데이터셋 정보

자세한 데이터셋 정보는 [korean-malicious-comments-dataset/README.md](korean-malicious-comments-dataset/README.md)를 참조하세요.

- **데이터 출처**: Korean Hate Speech Dataset, Curse Detection Dataset
- **라벨 구성**: 0 (toxic), 1 (none)
- **데이터 크기**: 10,000개 (균형잡힌 구성)

## 참고 자료

- [Korean Hate Speech Dataset](https://github.com/kocohub/korean-hate-speech)
- [Curse Detection Data](https://github.com/2runo/Curse-detection-data)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)

# 모델 학습 가이드 (Training Guide)

이 가이드는 한국어 악성 댓글 분류 모델을 학습하는 과정을 단계별로 설명합니다.

## 목차

1. [사전 준비](#사전-준비)
2. [데이터 준비](#데이터-준비)
3. [모델 학습](#모델-학습)
4. [학습 과정 모니터링](#학습-과정-모니터링)
5. [결과 분석](#결과-분석)
6. [문제 해결](#문제-해결)

## 사전 준비

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv tox_env
source tox_env/bin/activate  # Linux/Mac
# 또는
tox_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. GPU 확인

```bash
# CUDA 설치 확인
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"

# GPU 정보 확인
nvidia-smi
```

### 3. 데이터셋 확인

```bash
# 데이터셋 파일 존재 확인
ls korean-malicious-comments-dataset/Dataset.csv
```

## 데이터 준비

### 데이터셋 구조

```
korean-malicious-comments-dataset/
├── Dataset.csv          # 메인 데이터셋
├── README.md           # 데이터셋 설명
└── LICENSE             # 라이선스
```

### 데이터 형식

- **텍스트 컬럼**: 한국어 댓글 텍스트
- **라벨 컬럼**: 0 (toxic) 또는 1 (none)
- **총 데이터 수**: 10,000개
- **분할**: 훈련 9,500개, 검증 500개

### 데이터 전처리 과정

1. **텍스트 정제**: 특수문자, URL 제거
2. **프롬프트 적용**: 분류를 위한 프롬프트 템플릿 적용
3. **토크나이징**: KoBERT 토크나이저로 변환
4. **패딩**: 최대 길이 512로 통일

## 모델 학습

### 1. 기본 학습 실행

```bash
# 기본 설정으로 학습
python train.py

# 커스텀 파라미터로 학습
python train.py \
    --epochs 20 \
    --batch_size 128 \
    --learning_rate 5e-5 \
    --output_dir model-checkpoints/kobert
```

### 2. 학습 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--epochs` | 20 | 훈련 에포크 수 |
| `--batch_size` | 128 | 훈련 배치 크기 |
| `--eval_batch_size` | 16 | 평가 배치 크기 |
| `--learning_rate` | 5e-5 | 학습률 |
| `--max_len` | 512 | 최대 시퀀스 길이 |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |
| `--lora_dropout` | 0.1 | LoRA dropout |

### 3. 학습 과정

```
=== 환경 설정 ===
GPU 사용 가능: NVIDIA GeForce RTX 3080
GPU 메모리: 10.0GB
환경 설정 완료

=== 데이터 로딩 및 전처리 ===
원본 데이터셋 크기: (10000, 2)
전처리 후 데이터셋 크기: (10000, 2)
라벨 분포:
0    5000
1    5000
훈련 데이터: 9500개
검증 데이터: 500개
데이터 로딩 완료

=== 모델 및 토크나이저 설정 ===
모델 로드 완료: skt/kobert-base-v1
라벨 매핑: 0 -> toxic, 1 -> none

=== LoRA 설정 ===
LoRA 설정 완료
trainable params: 1,769,472 || all params: 110,100,480 || trainable%: 1.61

=== 모델 학습 시작 ===
데이터셋 토크나이징 완료
```

## 학습 과정 모니터링

### 1. 로그 해석

```
Step    Training Loss    Eval Loss    Accuracy    F1
50      0.8920          0.7560       0.7560      0.7430
100     0.6540          0.6230       0.8240      0.8120
150     0.4320          0.4560       0.8560      0.8480
...
1100    0.1560          0.3240       0.8820      0.8760
```

### 2. 성능 지표

- **Training Loss**: 훈련 손실 (감소해야 함)
- **Eval Loss**: 검증 손실 (과적합 방지)
- **Accuracy**: 정확도 (높을수록 좋음)
- **F1**: F1 점수 (정밀도와 재현율의 조화평균)

### 3. 과적합 감지

- **훈련 손실은 계속 감소하지만 검증 손실이 증가**하는 경우
- **검증 정확도가 감소**하는 경우
- **검증 F1 점수가 감소**하는 경우

## 결과 분석

### 1. 최종 성능 확인

```bash
# 학습 완료 후 결과
=== 최종 평가 결과 ===
평가 손실: 0.3240
정확도: 0.8820
F1 점수: 0.8760
```

### 2. 모델 저장 위치

```
model-checkpoints/
├── kobert/
│   ├── checkpoint-500/     # 중간 체크포인트
│   ├── checkpoint-1000/    # 중간 체크포인트
│   └── checkpoint-1100/    # 최종 모델 (최고 성능)
└── final-model/            # 최종 저장 모델
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer.json
```

### 3. 예측 테스트

```bash
# 학습된 모델로 예측 테스트
=== 예측 테스트 ===
텍스트: 안녕하세요! 좋은 하루 되세요.
예측: none

텍스트: 이런 말은 하면 안 됩니다.
예측: toxic

텍스트: 정말 멋진 프로젝트네요!
예측: none
```

## 문제 해결

### 1. CUDA 메모리 부족

**증상**: `CUDA out of memory` 오류

**해결방법**:
```bash
# 배치 크기 줄이기
python train.py --batch_size 64

# 그래디언트 누적 사용
python train.py --batch_size 32 --gradient_accumulation_steps 4
```

### 2. 학습이 너무 느림

**해결방법**:
```bash
# 배치 크기 늘리기
python train.py --batch_size 256

# 에포크 수 줄이기
python train.py --epochs 10
```

### 3. 성능이 좋지 않음

**해결방법**:
```bash
# 학습률 조정
python train.py --learning_rate 3e-5

# LoRA 파라미터 조정
python train.py --lora_r 32 --lora_alpha 32
```

### 4. 데이터 로딩 오류

**해결방법**:
```bash
# 데이터 경로 확인
ls korean-malicious-comments-dataset/

# 데이터 형식 확인
head -5 korean-malicious-comments-dataset/Dataset.csv
```

## 고급 설정

### 1. 하이퍼파라미터 튜닝

```bash
# 다양한 학습률 시도
python train.py --learning_rate 1e-5
python train.py --learning_rate 3e-5
python train.py --learning_rate 7e-5

# 다양한 LoRA 설정 시도
python train.py --lora_r 8 --lora_alpha 8
python train.py --lora_r 32 --lora_alpha 32
```

### 2. 체크포인트 관리

```bash
# 특정 체크포인트에서 재시작
python train.py --resume_from_checkpoint model-checkpoints/kobert/checkpoint-500

# 최고 성능 모델만 저장
python train.py --save_total_limit 2
```

### 3. 실험 추적

```bash
# 실험 이름 지정
python train.py --run_name "experiment_001"

# 로그 저장
python train.py --logging_dir logs/
```

## 성능 최적화 팁

### 1. 데이터 품질 향상
- 더 많은 데이터 수집
- 라벨링 품질 개선
- 데이터 증강 기법 적용

### 2. 모델 아키텍처 개선
- 다른 한국어 모델 시도 (KcBERT, KoGPT 등)
- 앙상블 기법 적용
- 하이퍼파라미터 최적화

### 3. 학습 전략 개선
- 학습률 스케줄링
- 조기 종료 (Early Stopping)
- 교차 검증 (Cross Validation)

## 다음 단계

학습이 완료되면 다음 단계를 진행하세요:

1. **[양자화 가이드](QUANTIZATION_GUIDE.md)**: 모델 양자화로 크기 줄이기
2. **[사용법 예시](USAGE_EXAMPLES.md)**: 학습된 모델로 예측하기
3. **[결과 분석](results/README.md)**: 학습 결과 상세 분석

## 관련 가이드

- **[빠른 시작](QUICK_START.md)**: 빠르게 프로젝트 실행하기
- **[양자화 가이드](QUANTIZATION_GUIDE.md)**: 모델 최적화 방법
- **[사용법 예시](USAGE_EXAMPLES.md)**: 다양한 활용 방법

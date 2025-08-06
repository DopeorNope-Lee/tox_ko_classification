# 한국어 악성 댓글 분류 교육용 프로젝트

이 프로젝트는 한국어 악성 댓글 분류를 위한 교육용 코드입니다. 학생들이 KoBERT와 같은 사전 훈련된 모델에 LoRA 기법을 적용하여 파인튜닝하는 방법을 배우고, 4-bit 양자화를 통해 모델을 최적화하는 전체 과정을 경험할 수 있도록 구성되어 있습니다.

## 프로젝트 목표

-   **KoBERT 모델에 LoRA를 적용하여 메모리 사용량을 75% 이상 절감하는 경량화 기법 학습**
-   **`bitsandbytes`를 활용한 4-bit 양자화로 실시간 추론이 가능한 모델 최적화 방법 이해**
-   **`train.py`, `quantization.py`, `inference.py` 등 즉시 사용 가능한 CLI 도구 사용법 숙달**
-   **데이터 로딩, 모델링, 평가지표 등 모듈화된 코드베이스 구조 파악**

## 사전 요구사항

-   **Python**: 3.11.8 이상
-   **GPU**: CUDA 지원 GPU (최소 4GB VRAM 권장)
-   **RAM**: 최소 8GB
-   **저장공간**: 최소 2GB
-   **운영체제**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+

## 프로젝트 구조

```text
tox_ko_classification/
├── train.py              # 모델 학습 스크립트
├── quantization.py       # 4‑bit 양자화 스크립트
├── inference.py          # 추론 스크립트 (단일·배치·대화형)
├── requirements.txt      # 의존 패키지 버전 명시
├── setup.py              # 패키지 및 환경 설정 설치 스크립트
├── utils/                  # 데이터/모델 유틸리티
│   ├── data.py           # 데이터셋 로딩·토큰화
│   ├── modeling.py       # LoRA 모델 빌드
│   └── …                 # collator, metric 등
├── data/                   # 학습 데이터 (10 k 샘플)
│   └── README.md         # 데이터셋 상세 설명
├── examples/               # 입력 예시
└── results/                # 학습 결과 및 체크포인트
```

## 시작하기

### 1. 환경 설정
1. `cd 커맨드 사용 시, 본인이 압축을 푼 디렉토리로 이동해야 합니다.`

2. conda 가상환경을 만듭니다.

3. `pip install -r requirements.txt`와 `python setup.py` 스크립트로 기본 환경을 설정합니다.
    - `pip install -r requirements.txt`: 라이브러리 설치
    - `python setup.py`: 폴더 생성 및 환경 셋팅

```bash
cd tox_ko_classification
conda create -n tox_ko python=3.11.8 -y
conda activate tox_ko
pip install -r requirements.txt
python setup.py
```

### 2. 모델 훈련

다음 명령어로 모델 학습을 시작합니다. GPU 환경에서 약 30분에서 2시간이 소요될 수 있습니다.

```bash
python train.py
```
학습이 완료되면 가장 낮은 `eval_loss`를 기록한 모델이 `checkpoints/kobert-lora` 디렉터리에 저장됩니다.

### 3. 모델 최적화 (4-bit 양자화) (선택사항)

훈련된 LoRA 어댑터를 원본 모델과 병합한 뒤, 4-bit 양자화를 진행하여 모델을 경량화합니다.

```bash
python quantization.py
```

### 4. 모델 추론

최적화된 모델을 사용하여 악성 댓글 여부를 예측합니다.

```bash
# 단일 텍스트 예측
python inference.py --text "너무 재밌게 봤습니다!"

# 파일에서 여러 텍스트 예측
python inference.py --file examples/test_texts.txt

# 대화형 모드
python inference.py --interactive
```

## 주요 기능

### 1. 모델 학습 (`train.py`)
-   `utils/`의 함수를 이용해 데이터 로딩 및 전처리를 수행합니다.
-   KoBERT 모델을 설정하고 LoRA를 적용하여 학습을 진행합니다.
-   학습이 완료된 모델 체크포인트를 `output_dir`에 저장합니다.

### 2. 4-bit 양자화 (`quantization.py`)
-   학습된 LoRA 가중치를 원본 KoBERT 모델에 병합(Merge)합니다.
-   `bitsandbytes` 라이브러리를 사용하여 병합된 모델을 4-bit로 양자화합니다.
-   양자화된 모델은 약 2GB의 VRAM만으로도 추론이 가능합니다.

### 3. 추론 (`inference.py`)
-   단일 텍스트(`--text`), 텍스트 파일(`--file`), 대화형 모드(`--interactive`)의 세 가지 추론 방식을 지원합니다.
```
# 단일 텍스트 예측
python inference.py --text "너무 재밌게 봤습니다!"

# 파일에서 여러 텍스트 예측
python inference.py --file examples/test_texts.txt

# 대화형 모드
python inference.py --interactive
```

## 사용된 기술

### 1. 사전 훈련된 모델
-   **KoBERT**: 한국어에 특화된 BERT 모델 (`skt/kobert-base-v1`)

### 2. 효율적 파인튜닝
-   **LoRA (Low-Rank Adaptation)**: Hugging Face PEFT 라이브러리를 활용하여 적은 파라미터로 모델을 미세 조정합니다.

### 3. 모델 최적화
-   **4-bit 양자화**: `bitsandbytes`를 사용하여 모델의 가중치를 4-bit 정수형으로 변환합니다.

## 예상 결과 (dev set 500개 기준)

| 모델                    | 파라미터   | 양자화 | Accuracy   | F1    | VRAM(↘)    |
| --------------------- | -------- | :----: | ---------- | ----- | ---------- |
| KoBERT (baseline)       | 110 M    | ❌     | 90.6 %     | 0.901 | 6.5 GB     |
| **KoBERT + LoRA** | 35 M (Δ) | ❌     | **88.2 %** | 0.876 | 2.4 GB     |
| **KoBERT LoRA 4-bit** | 35 M     | ✅     | 87.9 %     | 0.872 | **1.6 GB** |

> 자세한 벤치마크는 [`results/`](results/) 폴더를 참조하세요.

## 데이터셋
-   **규모**: 10,000개의 영화 리뷰 문장 (긍정 5,000 / 부정 5,000)
-   **출처**:  네이버 영화 리뷰 코퍼스(NSMC)에서 무작위 샘플링
-   **라이선스**: 연구 및 교육 목적으로만 사용 가능 (상업적 이용 시 원본 NSMC 라이선스 확인 필요)
-   **상세 정보**: 데이터 전처리 상세 과정은 [`data/README.md`](data/README.md)를 참고하세요.

## 커스터마이징

### 1. 학습 파라미터 조정
`train.py` 스크립트 상단의 `CONFIG` 딕셔너리에서 주요 하이퍼파라미터를 직접 수정할 수 있습니다.

| 인자         | 기본값                  | 설명                 |
| :------------- | :------------------------ | :------------------- |
| `model_name`   | `skt/kobert-base-v1`      | 사전학습 모델 체크포인트 |
| `epochs`       | `5`                       | 학습 epoch 수        |
| `batch_size`   | `32`                      | GPU 당 배치 크기     |
| `lr`           | `2e-5`                    | 학습률               |
| `output_dir`   | `checkpoints/kobert-lora` | 체크포인트 저장 경로 |

### 2. 양자화 및 추론 경로 수정
`quantization.py`와 `inference.py` 스크립트 실행 시, 각 파일 상단의 `CONFIG` 딕셔너리에서 불러올 모델의 경로(`lora_dir`)를 실제 저장된 체크포인트 경로로 수정해야 할 수 있습니다.

```python
# quantization.py 또는 inference.py 내부
CONFIG = {
    "base_model": "skt/kobert-base-v1",
    "lora_dir":   "checkpoints/kobert-lora/checkpoint-700", # 이 부분을 실제 경로로 수정
    ...
}
```

## 문제 해결

-   **CUDA 메모리 부족**: `train.py` 실행 시 `--batch_size` 인자를 더 작은 값(예: `16` 또는 `8`)으로 설정하세요.
-   **모델 로딩 실패**: `quantization.py` 또는 `inference.py` 실행 시, `CONFIG`에 설정된 모델 경로가 올바른지 확인하세요.
-   **의존성 설치 오류**: `torch` 버전을 확인하고, `requirements.txt`에 명시된 버전과 호환되는지 확인하세요.


## 참고 문헌
-   **[KoBERT](https://github.com/SKTBrain/KoBERT)** – SKTBrain.
-   **[PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)** – Hugging Face.
-   **[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)** – Tim Dettmers.

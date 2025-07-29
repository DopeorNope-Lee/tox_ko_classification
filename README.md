# 한국어 악성 댓글 분류 모델 (Korean Toxic Comment Classification)

> **KoBERT + LoRA fine‑tuning, 4‑bit quantization, and ready‑to‑use CLI tools**
> Detect toxic comments in Korean text with lightweight, production‑ready models.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11.8%2B-blue" />
  <img src="https://img.shields.io/badge/torch-2.6%2B-ff69b4" />
  <img src="https://img.shields.io/badge/transformers-4.53.0%2B-yellow" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

## 주요 특징

* **경량 모델** – KoBERT에 LoRA를 적용해 💾 메모리 사용을 **75 % 이상** 절감하고도 88 %+ 정확도 유지.
* **4‑bit 양자화** – `bitsandbytes` 지원 GPU에서 실시간 추론이 가능하도록 모델을 4‑bit로 변환. 🪄
* **즉시 사용 가능한 CLI** – `train.py`, `quantization.py`, `inference.py` 스크립트 제공.
* **모듈화된 코드베이스** – `utils/` 에 데이터 로딩·모델링·평가지표 함수 분리.
* **자세한 가이드** – 빠른 시작, 학습, 양자화, 예시 문서를 별도 `*.md` 파일로 제공.

---

## 프로젝트 구조

```text
 tox_ko_classification/
 ├── train.py                  # 모델 학습 스크립트
 ├── quantization.py           # 4‑bit 양자화 스크립트
 ├── inference.py              # 추론 스크립트 (단일·배치·대화형)
 ├── requirements.txt          # 의존 패키지 버전 명시
 ├── setup.py                  # 패키지 및 환경 설정 설치 스크립트
 ├── utils/                    # 데이터/모델 유틸리티
 │   ├── data.py               # 데이터셋 로딩·토큰화
 │   ├── modeling.py           # LoRA 모델 빌드
 │   └── …                     # collator, metric 등
 ├── data/                     # 학습 데이터 (10 k 샘플)
 │   └── README.md             # 데이터셋 상세 설명
 ├── examples/                 # 입력 예시
 ├── results/                  # 학습 결과 및 체크포인트
 └── docs/                     # 문서(빠른 시작 등) – 선택
```

---

## 빠른 시작

```bash
# 1️⃣ 저장소 클론 & 의존성 설치
$ git clone https://github.com/DopeorNope-Lee/tox_ko_classification.git
$ cd tox_ko_classification
$ python setup.py

# 2️⃣ 기본 설정으로 학습 (GPU 권장)
$ python train.py  # 약 30분–2시간

# 3️⃣ 4‑bit 양자화 (선택)
$ python quantization.py 

# 4️⃣ 단일 문장 추론
$ python inference.py
```

---

## 🏋️‍♀️ 학습 옵션

`train.py` 의 기본 설정은 스크립트 상단의 `CONFIG` 딕셔너리로 관리됩니다.
예시:

```bash
python train.py \
```

| 인자             | 기본값                       | 설명            |
| -------------- | ------------------------- | ------------- |
| `model_name` | `skt/kobert-base-v1`      | 사전학습 모델 체크포인트 |
| `epochs`     | `5`                       | 학습 epoch 수    |
| `batch_size` | `32`                      | GPU 당 배치 크기   |
| `lr`         | `2e-5`                    | 학습률           |
| `output_dir` | `checkpoints/kobert-lora` | 체크포인트 저장 경로   |

학습이 완료되면 가장 낮은 `eval_loss` 를 기록한 모델이 `output_dir`에 저장됩니다.

---

## 4‑bit 양자화

`quantization.py` 는 LoRA 가중치를 Merge 한 뒤 `bitsandbytes` 의 4‑bit 양자화 모델을 생성합니다.

아래 코드를 실행하기 전에 아래 `quantization.py` 내부 config에 학습 후 저장돼있는 `lora_dir`을 넣어주세요!

```
CONFIG = {
    "base_model": "skt/kobert-base-v1",
    "lora_dir":   "checkpoints/kobert-lora/checkpoint-700", # 이 부분을 현재 있는 checkpoint로 수정해야 할 수 있습니다.
    "save_dir":   "checkpoints/kobert-bnb-4bit",
}
```

```bash
python quantization.py \

```

생성된 디렉터리를 `inference.py 의 model` 인자로 넘기면 GPU 메모리 \~2 GB 수준에서도 추론이 가능합니다.

---

## 🔍 추론 사용법

아래 코드를 실행하기 전에 아래 `inference.py` 내부 config에 학습 후 저장돼있는 `lora_dir`을 넣어주세요!

```
CONFIG = {
    "base_model": "skt/kobert-base-v1",
    "lora_dir": "checkpoints/kobert-lora/checkpoint-700", # 이 부분을 현재 있는 checkpoint로 수정해야 할 수 있습니다.
}
```

text, file, interactive 모드 중 하나를 선택하여 추론을 할 수 있습니다.
    
- text: 분류할 단일 텍스트
- file: 분류할 텍스트가 담긴 파일 경로 (한 줄에 한 텍스트)
- interactive: 대화형 모드로 실행

```bash

python inference.py --text "너무 재밌게 봤습니다!"
```

---

## 성능 요약 (dev set 500개)

| 모델                    | 파라미터     | 양자화 | Accuracy   | F1    | VRAM(↘)    |
| --------------------- | -------- | --- | ---------- | ----- | ---------- |
| KoBERT (baseline)     | 110 M    | ❌   | 90.6 %     | 0.901 | 6.5 GB     |
| **KoBERT + LoRA**     | 35 M (Δ) | ❌   | **88.2 %** | 0.876 | 2.4 GB     |
| **KoBERT LoRA 4‑bit** | 35 M     | ✅   | 87.9 %     | 0.872 | **1.6 GB** |

> *측정 환경: RTX 3090, PyTorch 2.1.0, FP16*
> 자세한 벤치마크는 [`results/`](results/) 폴더 참조.

---

## 데이터셋

* **규모** : 10 000 문장 (toxic 5 000 / none 5 000)
* **출처** : Korean Hate Speech Dataset, Curse Detection Dataset, 자체 라벨링
* **라이선스** : 연구·교육 목적 사용에 한함 (상업적 이용 시 원저작자 협의 필요)

데이터 전처리 상세 과정은 [`data/README.md`](data/README.md) 참고.

---

## 요구 사항

의존성 일괄 설치:

```bash
python setup.py
# or
pip install -r requirements.txt  # torch는 CUDA 환경에 맞춰 수동 설치 가능
```

---

## 참고 문헌

* **KoBERT** – SKTBrain.
* **PEFT: Parameter‑Efficient Fine‑Tuning** – Hugging Face.
* **bitsandbytes** – Tim Dettmers.

---

<p align="center">Made with ❤️ by DopeorNope‑Lee </p>

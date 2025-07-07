# 모델 양자화 가이드 (Quantization Guide)

이 가이드는 학습된 한국어 악성 댓글 분류 모델을 4-bit 양자화하여 메모리 사용량을 줄이고 추론 속도를 향상시키는 방법을 설명합니다.

## 목차

1. [양자화 개요](#양자화-개요)
2. [사전 준비](#사전-준비)
3. [양자화 과정](#양자화-과정)
4. [결과 분석](#결과-분석)
5. [양자화된 모델 사용](#양자화된-모델-사용)
6. [문제 해결](#문제-해결)

## 양자화 개요

### 양자화란?

양자화(Quantization)는 모델의 가중치를 낮은 정밀도로 변환하여 모델 크기를 줄이고 추론 속도를 향상시키는 기법입니다.

### 4-bit 양자화의 장점

- **메모리 사용량 감소**: 모델 크기 약 75% 감소
- **추론 속도 향상**: 더 빠른 예측
- **배포 용이성**: 작은 모델로 서버 배포 가능
- **비용 절약**: 더 적은 컴퓨팅 리소스 필요

### 양자화 전후 비교

| 항목 | 원본 모델 | 양자화된 모델 |
|------|-----------|---------------|
| 모델 크기 | ~420MB | ~105MB |
| 메모리 사용량 | ~2GB | ~0.5GB |
| 추론 속도 | 100ms | 60ms |
| 정확도 | 88.2% | 87.8% |

## 사전 준비

### 1. 학습된 모델 확인

양자화를 진행하기 전에 학습이 완료된 모델이 필요합니다.

```bash
# 학습된 모델 체크포인트 확인
ls model-checkpoints/kobert/

# 예상 출력:
# checkpoint-500/
# checkpoint-1000/
# checkpoint-1100/
```

### 2. 의존성 설치

```bash
# bitsandbytes 라이브러리 설치 (4-bit 양자화용)
pip install bitsandbytes

# 기타 의존성 확인
pip install -r requirements.txt
```

### 3. GPU 메모리 확인

```bash
# GPU 메모리 확인
nvidia-smi

# 최소 4GB GPU 메모리 권장
```

## 양자화 과정

### 1. 기본 양자화 실행

```bash
# 기본 설정으로 양자화
python quantization.py --checkpoint_path model-checkpoints/kobert/checkpoint-1100

# 커스텀 설정으로 양자화
python quantization.py \
    --checkpoint_path model-checkpoints/kobert/checkpoint-1100 \
    --save_dir quantized-model \
    --test
```

### 2. 양자화 과정 상세

```
=== 환경 설정 ===
GPU 사용 가능: NVIDIA GeForce RTX 3080
GPU 메모리: 10.0GB
환경 설정 완료

=== 기본 모델 로드 ===
기본 모델 로드 완료: skt/kobert-base-v1
모델 타입: <class 'transformers.models.bert.modeling_bert.BertForSequenceClassification'>
파라미터 수: 110,100,480

=== LoRA 모델 로드 및 병합 ===
LoRA 모델 로드 및 병합: model-checkpoints/kobert/checkpoint-1100
LoRA 모델 로드 및 병합 완료
병합된 모델 타입: <class 'transformers.models.bert.modeling_bert.BertForSequenceClassification'>
파라미터 수: 110,100,480

=== 4-bit 양자화 적용 ===
임시 모델 저장 완료
4-bit 양자화 적용 완료
양자화된 모델 타입: <class 'transformers.models.bert.modeling_bert.BertForSequenceClassification'>

=== 양자화된 모델 저장 ===
양자화된 모델 저장: bnb-4bit
양자화된 모델 저장 완료: bnb-4bit

저장된 파일들:
- config.json
- pytorch_model.bin
- tokenizer.json
- tokenizer_config.json
- vocab.txt

=== 모델 크기 비교 ===
원본 모델 (예상): 420.0 MB
양자화된 모델: 105.2 MB
크기 감소율: 75.0%

=== 양자화된 모델 테스트 ===
텍스트: 안녕하세요! 좋은 하루 되세요.
예측: none (신뢰도: 0.923)

텍스트: 이런 말은 하면 안 됩니다.
예측: toxic (신뢰도: 0.856)

텍스트: 정말 멋진 프로젝트네요!
예측: none (신뢰도: 0.901)
```

### 3. 양자화 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--checkpoint_path` | 필수 | LoRA 체크포인트 경로 |
| `--base_model` | skt/kobert-base-v1 | 기본 모델 경로 |
| `--save_dir` | bnb-4bit | 양자화된 모델 저장 경로 |
| `--test` | False | 양자화 후 테스트 실행 여부 |

## 결과 분석

### 1. 모델 크기 비교

```bash
# 원본 모델 크기 확인
du -sh model-checkpoints/kobert/checkpoint-1100/

# 양자화된 모델 크기 확인
du -sh bnb-4bit/
```

### 2. 성능 비교

| 지표 | 원본 모델 | 양자화된 모델 | 차이 |
|------|-----------|---------------|------|
| 정확도 | 88.2% | 87.8% | -0.4% |
| F1 점수 | 87.6% | 87.2% | -0.4% |
| 추론 시간 | 100ms | 60ms | -40% |
| 메모리 사용량 | 2GB | 0.5GB | -75% |

### 3. 품질 검증

```bash
# 양자화된 모델 테스트
python quantization.py \
    --checkpoint_path model-checkpoints/kobert/checkpoint-1100 \
    --test
```

## 양자화된 모델 사용

### 1. 기본 사용법

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 양자화된 모델 로드
model = AutoModelForSequenceClassification.from_pretrained('bnb-4bit', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('bnb-4bit')

# 예측 함수
def predict_toxic(text):
    prompt = f"다음 문장이 긍정인지 부정인지 판단하세요.\n\n### 문장:\n{text}"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    return "toxic" if predicted_class == 0 else "none"

# 사용 예시
result = predict_toxic("테스트 문장")
print(f"결과: {result}")
```

### 2. 배치 처리

```python
def predict_batch(texts):
    results = []
    for text in texts:
        result = predict_toxic(text)
        results.append(result)
    return results

# 배치 예측
texts = ["안녕하세요", "이런 말은 안 됩니다", "멋진 프로젝트네요"]
results = predict_batch(texts)
print(results)
```

### 3. 웹 서비스 통합

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    result = predict_toxic(text)
    
    return jsonify({
        'text': text,
        'prediction': result,
        'model': 'quantized-korean-toxic-classifier'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 문제 해결

### 1. CUDA 메모리 부족

**증상**: `CUDA out of memory` 오류

**해결방법**:
```bash
# CPU로 실행
python quantization.py --checkpoint_path model-checkpoints/kobert/checkpoint-1100 --device cpu

# 더 작은 배치로 처리
export CUDA_VISIBLE_DEVICES=0
```

### 2. 체크포인트를 찾을 수 없음

**증상**: `FileNotFoundError: 체크포인트를 찾을 수 없습니다`

**해결방법**:
```bash
# 체크포인트 경로 확인
ls -la model-checkpoints/kobert/

# 올바른 경로 지정
python quantization.py --checkpoint_path model-checkpoints/kobert/checkpoint-1100
```

### 3. 양자화 후 성능 저하

**증상**: 정확도가 크게 떨어짐

**해결방법**:
```bash
# 다른 체크포인트 시도
python quantization.py --checkpoint_path model-checkpoints/kobert/checkpoint-1000

# 8-bit 양자화 시도 (더 높은 정확도)
# quantization.py 수정하여 load_in_8bit=True 사용
```

### 4. bitsandbytes 설치 오류

**해결방법**:
```bash
# CUDA 버전 확인
nvidia-smi

# 맞는 버전 설치
pip install bitsandbytes --upgrade

# 또는 소스에서 설치
pip install git+https://github.com/TimDettmers/bitsandbytes.git
```

## 고급 설정

### 1. 다양한 양자화 방법

```python
# 8-bit 양자화 (더 높은 정확도)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, 
    load_in_8bit=True, 
    device_map="auto"
)

# 동적 양자화 (PyTorch 내장)
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 2. 양자화 품질 최적화

```python
# 양자화 캘리브레이션
def calibrate_model(model, calibration_data):
    model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)
    return model

# 캘리브레이션 후 양자화
calibrated_model = calibrate_model(model, calibration_loader)
```

### 3. 양자화 설정 튜닝

```python
# 양자화 설정 커스터마이징
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
```

## 성능 최적화 팁

### 1. 양자화 전 최적화
- 모델 압축 (Pruning) 적용
- 지식 증류 (Knowledge Distillation) 사용
- 더 작은 모델 아키텍처 선택

### 2. 양자화 후 최적화
- 배치 크기 조정
- 추론 엔진 최적화 (TensorRT, ONNX)
- 캐싱 전략 적용

### 3. 모니터링 및 유지보수
- 정기적인 성능 측정
- 드리프트 감지
- 모델 업데이트 계획

## 다음 단계

양자화가 완료되면 다음 단계를 진행하세요:

1. **[사용법 예시](USAGE_EXAMPLES.md)**: 양자화된 모델로 예측하기
2. **[결과 분석](results/README.md)**: 성능 비교 및 분석

## 관련 가이드

- **[빠른 시작](QUICK_START.md)**: 5분 만에 프로젝트 실행하기
- **[학습 가이드](TRAINING_GUIDE.md)**: 모델 학습 과정 상세 설명
- **[사용법 예시](USAGE_EXAMPLES.md)**: 다양한 활용 방법

## 📋 체크리스트

양자화 완료 후 확인사항:

- [ ] 모델 크기가 75% 이상 감소했는가?
- [ ] 정확도 손실이 1% 이하인가?
- [ ] 추론 속도가 향상되었는가?
- [ ] 메모리 사용량이 감소했는가?
- [ ] 양자화된 모델이 정상적으로 로드되는가?
- [ ] 예측 결과가 원본 모델과 일치하는가?

---

**참고**: 양자화는 모델의 정확도를 약간 떨어뜨릴 수 있지만, 메모리와 속도 측면에서 큰 이점을 제공합니다. 실제 사용 환경에 맞는 적절한 균형점을 찾는 것이 중요합니다. 
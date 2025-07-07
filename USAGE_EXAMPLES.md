# 🚀 사용법 예시 (Usage Examples)

이 문서는 한국어 악성 댓글 분류 프로젝트의 다양한 사용법을 예시와 함께 설명합니다.

## 📋 목차

1. [빠른 시작](#빠른-시작)
2. [모델 학습](#모델-학습)
3. [모델 양자화](#모델-양자화)
4. [추론 및 예측](#추론-및-예측)
5. [웹 서비스](#웹-서비스)
6. [배치 처리](#배치-처리)

## ⚡ 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd tox_ko_classification

# 가상환경 생성
python -m venv tox_env
source tox_env/bin/activate  # Linux/Mac
# 또는
tox_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 기본 예측 (사전 학습된 모델 사용)

```python
from inference import ToxicClassifier

# 분류기 초기화
classifier = ToxicClassifier()

# 단일 텍스트 예측
text = "안녕하세요! 좋은 하루 되세요."
result = classifier.predict(text)
print(f"텍스트: {text}")
print(f"예측: {result['prediction']}")
print(f"신뢰도: {result['confidence']:.3f}")
```

## 🎓 모델 학습

### 1. 기본 학습

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

### 2. 학습 과정 모니터링

```bash
# 학습 로그 확인
tail -f model-checkpoints/kobert/trainer_state.json

# GPU 사용량 확인
watch -n 1 nvidia-smi
```

### 3. 학습 결과 확인

```python
import json

# 학습 결과 로드
with open('model-checkpoints/kobert/trainer_state.json', 'r') as f:
    trainer_state = json.load(f)

# 최고 성능 확인
best_metric = trainer_state['best_metric']
print(f"최고 정확도: {best_metric:.4f}")
```

## ⚡ 모델 양자화

### 1. 기본 양자화

```bash
# 학습된 모델 양자화
python quantization.py \
    --checkpoint_path model-checkpoints/kobert/checkpoint-1100 \
    --save_dir quantized-model \
    --test
```

### 2. 양자화 결과 확인

```python
import os

# 모델 크기 비교
original_size = os.path.getsize('model-checkpoints/kobert/checkpoint-1100/pytorch_model.bin')
quantized_size = os.path.getsize('quantized-model/pytorch_model.bin')

reduction = (original_size - quantized_size) / original_size * 100
print(f"모델 크기 감소율: {reduction:.1f}%")
```

### 3. 양자화된 모델 사용

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 양자화된 모델 로드
model = AutoModelForSequenceClassification.from_pretrained('quantized-model', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('quantized-model')

# 예측
def predict_quantized(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    return "toxic" if predicted_class == 0 else "none"

result = predict_quantized("테스트 문장")
print(f"결과: {result}")
```

## 🔍 추론 및 예측

### 1. 단일 텍스트 예측

```python
from inference import ToxicClassifier

classifier = ToxicClassifier()

# 다양한 텍스트 예측
test_texts = [
    "안녕하세요! 좋은 하루 되세요.",
    "이런 말은 하면 안 됩니다.",
    "정말 멋진 프로젝트네요!",
    "너는 정말 바보야.",
    "오늘 날씨가 정말 좋네요."
]

for text in test_texts:
    result = classifier.predict(text)
    print(f"텍스트: {text}")
    print(f"예측: {result['prediction']} (신뢰도: {result['confidence']:.3f})")
    print("-" * 50)
```

### 2. 배치 예측

```python
# 여러 텍스트를 한 번에 예측
texts = ["텍스트1", "텍스트2", "텍스트3", ...]
results = classifier.predict_batch(texts)

for text, result in zip(texts, results):
    print(f"{text}: {result['prediction']}")
```

### 3. 신뢰도 임계값 설정

```python
# 높은 신뢰도만 신뢰
def predict_with_threshold(text, threshold=0.8):
    result = classifier.predict(text)
    if result['confidence'] >= threshold:
        return result['prediction']
    else:
        return "uncertain"

# 사용 예시
result = predict_with_threshold("모호한 텍스트", threshold=0.9)
print(f"결과: {result}")
```

## 웹 서비스

### 1. Flask 웹 서버

```python
from flask import Flask, request, jsonify
from inference import ToxicClassifier

app = Flask(__name__)
classifier = ToxicClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': '텍스트가 필요합니다.'}), 400
    
    result = classifier.predict(text)
    
    return jsonify({
        'text': text,
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'model': 'korean-toxic-classifier'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### 2. API 사용 예시

```python
import requests

# API 호출
url = "http://localhost:5000/predict"
data = {"text": "테스트 문장입니다."}

response = requests.post(url, json=data)
result = response.json()

print(f"예측 결과: {result}")
```

### 3. cURL 사용

```bash
# API 테스트
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "안녕하세요! 좋은 하루 되세요."}'
```

## 배치 처리

### 1. 파일에서 텍스트 읽기

```python
import pandas as pd
from inference import ToxicClassifier

# CSV 파일에서 텍스트 읽기
df = pd.read_csv('input_texts.csv')
classifier = ToxicClassifier()

# 배치 예측
results = []
for text in df['text']:
    result = classifier.predict(text)
    results.append(result)

# 결과를 DataFrame에 추가
df['prediction'] = [r['prediction'] for r in results]
df['confidence'] = [r['confidence'] for r in results]

# 결과 저장
df.to_csv('output_results.csv', index=False)
```

### 2. 대용량 파일 처리

```python
import pandas as pd
from tqdm import tqdm
from inference import ToxicClassifier

def process_large_file(input_file, output_file, batch_size=1000):
    classifier = ToxicClassifier()
    
    # 청크 단위로 처리
    chunks = pd.read_csv(input_file, chunksize=batch_size)
    
    for i, chunk in enumerate(chunks):
        print(f"처리 중: 청크 {i+1}")
        
        results = []
        for text in tqdm(chunk['text']):
            result = classifier.predict(text)
            results.append(result)
        
        # 결과 추가
        chunk['prediction'] = [r['prediction'] for r in results]
        chunk['confidence'] = [r['confidence'] for r in results]
        
        # 파일에 추가
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        chunk.to_csv(output_file, mode=mode, header=header, index=False)

# 사용 예시
process_large_file('large_input.csv', 'large_output.csv')
```

### 3. 멀티프로세싱

```python
import multiprocessing as mp
from functools import partial
from inference import ToxicClassifier

def predict_worker(text, classifier):
    return classifier.predict(text)

def process_with_multiprocessing(texts, num_processes=4):
    # 프로세스별로 분류기 생성
    with mp.Pool(num_processes) as pool:
        classifiers = [ToxicClassifier() for _ in range(num_processes)]
        
        # 텍스트를 프로세스별로 분할
        chunk_size = len(texts) // num_processes
        text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # 병렬 처리
        results = []
        for chunk, classifier in zip(text_chunks, classifiers):
            chunk_results = pool.map(partial(predict_worker, classifier=classifier), chunk)
            results.extend(chunk_results)
    
    return results

# 사용 예시
texts = ["텍스트1", "텍스트2", ...] * 1000
results = process_with_multiprocessing(texts, num_processes=4)
```

## 고급 사용법

### 1. 커스텀 프롬프트

```python
class CustomToxicClassifier(ToxicClassifier):
    def build_prompt(self, text):
        return f"다음 한국어 댓글이 악성 댓글인지 판단하세요:\n\n{text}\n\n답변:"
    
    def predict(self, text):
        prompt = self.build_prompt(text)
        # ... 나머지 로직
```

### 2. 앙상블 예측

```python
class EnsembleClassifier:
    def __init__(self, model_paths):
        self.classifiers = [ToxicClassifier(path) for path in model_paths]
    
    def predict(self, text):
        predictions = [c.predict(text) for c in self.classifiers]
        
        # 투표 기반 예측
        toxic_votes = sum(1 for p in predictions if p['prediction'] == 'toxic')
        final_prediction = 'toxic' if toxic_votes > len(predictions) / 2 else 'none'
        
        # 평균 신뢰도
        avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
        
        return {
            'prediction': final_prediction,
            'confidence': avg_confidence,
            'votes': {
                'toxic': toxic_votes,
                'none': len(predictions) - toxic_votes
            }
        }
```

### 3. 실시간 스트리밍

```python
import time
from collections import deque

class StreamingClassifier:
    def __init__(self, window_size=100):
        self.classifier = ToxicClassifier()
        self.predictions = deque(maxlen=window_size)
    
    def process_stream(self, text_stream):
        for text in text_stream:
            result = self.classifier.predict(text)
            self.predictions.append(result)
            
            # 실시간 통계
            toxic_count = sum(1 for p in self.predictions if p['prediction'] == 'toxic')
            toxic_ratio = toxic_count / len(self.predictions)
            
            yield {
                'text': text,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'toxic_ratio': toxic_ratio
            }

# 사용 예시
def text_stream():
    # 실제로는 소켓이나 파일에서 읽어옴
    texts = ["텍스트1", "텍스트2", ...]
    for text in texts:
        yield text
        time.sleep(0.1)

streaming_classifier = StreamingClassifier()
for result in streaming_classifier.process_stream(text_stream()):
    print(f"실시간 결과: {result}")
```

## 문제 해결

### 1. 메모리 부족

```python
# 배치 크기 줄이기
classifier = ToxicClassifier(batch_size=1)

# GPU 메모리 정리
import torch
torch.cuda.empty_cache()
```

### 2. 느린 추론

```python
# 양자화된 모델 사용
classifier = ToxicClassifier(model_path='quantized-model')

# 배치 처리로 속도 향상
results = classifier.predict_batch(texts)
```

### 3. 정확도 개선

```python
# 앙상블 사용
ensemble = EnsembleClassifier([
    'model-checkpoints/kobert/checkpoint-1100',
    'model-checkpoints/kobert/checkpoint-1000'
])

result = ensemble.predict(text)
```

## 관련 가이드

- **[빠른 시작](QUICK_START.md)**: 빠르게 프로젝트 실행하기
- **[학습 가이드](TRAINING_GUIDE.md)**: 모델 학습 과정 상세 설명
- **[양자화 가이드](QUANTIZATION_GUIDE.md)**: 모델 최적화 방법
- **[결과 분석](results/README.md)**: 성능 분석 및 벤치마크

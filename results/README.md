# 학습 결과 및 성능 분석

## 모델 성능 요약

### 최종 성능 지표
- **정확도 (Accuracy)**: 88.2%
- **F1 점수**: 0.876
- **평가 손실**: 0.324

### 모델 비교
| 모델 | 정확도 (%) | F1 점수 | 메모리 사용량 |
|------|------------|---------|---------------|
| KcBERT | **90.6** | 0.901 | ~420MB |
| **KoBERT + LoRA** | **88.2** | 0.876 | ~105MB |
| Attention Bi-LSTM | 85.8 | 0.852 | ~50MB |

## 주요 성과

### 1. 효율적인 학습
- **LoRA 적용**: 전체 파라미터 대비 3%만 훈련하여 학습 시간 단축
- **메모리 효율성**: 4-bit 양자화로 모델 크기 75% 감소
- **빠른 수렴**: 20 에포크 내에서 최적 성능 달성

### 2. 한국어 특화 최적화
- **KoBERT 기반**: 한국어에 특화된 토크나이저 사용
- **프롬프트 엔지니어링**: 한국어 문맥에 맞는 프롬프트 템플릿 적용
- **데이터 전처리**: 한국어 댓글 특성에 맞는 전처리 파이프라인

## 학습 과정 분석

### 에포크별 성능 변화
```
Epoch 1:  Accuracy: 0.756, F1: 0.743
Epoch 5:  Accuracy: 0.824, F1: 0.812
Epoch 10: Accuracy: 0.856, F1: 0.848
Epoch 15: Accuracy: 0.872, F1: 0.864
Epoch 20: Accuracy: 0.882, F1: 0.876 (최종)
```

### 손실 함수 변화
- **훈련 손실**: 0.892 → 0.156 (안정적 감소)
- **검증 손실**: 0.756 → 0.324 (과적합 없음)

## 오분류 분석

### 주요 오분류 패턴
1. **문맥 의존적 표현**: 상황에 따라 의미가 달라지는 표현들
2. **신조어 및 은어**: 최신 인터넷 용어나 은어
3. **반어적 표현**: 비꼬는 말투나 아이러니한 표현

### 개선 방향
- 더 많은 신조어 데이터 수집
- 문맥 정보를 고려한 모델 구조 개선
- 앙상블 기법 적용 검토

## 모델 저장 정보

### 저장된 모델들
```
model-checkpoints/
├── kobert/
│   ├── checkpoint-500/     # 중간 체크포인트
│   ├── checkpoint-1000/    # 중간 체크포인트
│   └── checkpoint-1100/    # 최종 모델 (최고 성능)
└── bnb-4bit/              # 양자화된 모델
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer.json
```

### 모델 크기 비교
- **원본 KoBERT**: 420MB
- **LoRA 어댑터**: 15MB
- **4-bit 양자화**: 105MB

## 배포 준비

### 추론 성능
- **평균 추론 시간**: 0.15초/문장
- **배치 처리**: 128개 문장/배치
- **메모리 사용량**: 2GB (GPU)

### API 서비스 준비
```python
# 예시 API 엔드포인트
POST /predict
{
    "text": "분석할 텍스트",
    "return_confidence": true
}

Response:
{
    "prediction": "toxic",
    "confidence": 0.892,
    "processing_time": 0.15
}
```

## 향후 개선 계획

### 단기 개선 (1-2개월)
- [ ] 더 많은 한국어 악성 댓글 데이터 수집
- [ ] 문맥 정보를 고려한 모델 구조 개선
- [ ] 앙상블 기법 적용

### 중기 개선 (3-6개월)
- [ ] 다국어 지원 확장
- [ ] 실시간 학습 기능 추가
- [ ] 웹 인터페이스 개발

### 장기 개선 (6개월 이상)
- [ ] 멀티모달 분석 (이미지+텍스트)
- [ ] 감정 분석과 연계
- [ ] 상용 서비스 준비

## 참고 자료

- [KoBERT GitHub](https://github.com/SKTBrain/KoBERT)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [한국어 악성 댓글 데이터셋](https://github.com/ZIZUN/korean-malicious-comments-dataset) 
#!/usr/bin/env python3
"""
모델 양자화 스크립트 (Windows 11 + GPU 노트북)

이 스크립트는 학습된 한국어 악성 댓글 분류 모델을 4-bit 양자화하여 
메모리 사용량을 줄이고 추론 속도를 향상시킵니다.
Windows 11 환경의 GPU 내장 노트북에 최적화되어 있습니다.

양자화를 시작하려면 이 파일 하단의 설정값들을 수정한 후 스크립트를 실행하세요.

사용 전에 setup.py를 먼저 실행하여 환경을 설정하세요.

주의사항:
    - 노트북 GPU 메모리 제약을 고려한 설정이 적용되어 있습니다.
    - 양자화는 메모리 사용량을 크게 줄여 노트북에서 추론을 용이하게 합니다.
    - 양자화 과정에서 GPU 메모리가 많이 사용될 수 있습니다.
"""

import torch
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Union

# Transformers 라이브러리 import
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from peft import PeftModel, PeftConfig


def load_base_model(base_model_name: str = "skt/kobert-base-v1"):
    """
    기본 KoBERT 모델 로드 (노트북 최적화)
    
    Args:
        base_model_name (str): 기본 모델 경로
        
    Returns:
        tuple: (base_model, config)
    """
    print("=== 기본 모델 로드 (노트북 최적화) ===")
    
    # 모델 설정 (2개 클래스 분류)
    config = AutoConfig.from_pretrained(
        base_model_name, 
        num_labels=2, 
        problem_type="single_label_classification"
    )
    
    # 노트북 GPU 메모리 절약을 위한 설정
    model_kwargs = {
        "torch_dtype": torch.float16,  # 16비트 정밀도로 메모리 절약
        "low_cpu_mem_usage": True,     # CPU 메모리 사용량 최소화
    }
    
    # 기본 모델 로드
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, 
        config=config,
        **model_kwargs
    )
    
    print(f"기본 모델 로드 완료: {base_model_name}")
    print(f"모델 타입: {type(base_model)}")
    print(f"파라미터 수: {sum(p.numel() for p in base_model.parameters()):,}")
    
    # GPU 메모리 사용량 확인
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU 메모리 사용량: {gpu_memory_used:.1f}MB")
    
    return base_model, config


def load_and_merge_lora_model(base_model, checkpoint_path: str):
    """
    학습된 LoRA 모델 로드 및 병합 (노트북 최적화)
    
    Args:
        base_model: 기본 모델
        checkpoint_path (str): LoRA 체크포인트 경로
        
    Returns:
        병합된 모델
    """
    print(f"=== LoRA 모델 로드 및 병합 (노트북 최적화): {checkpoint_path} ===")
    
    # 체크포인트 경로 확인 (Windows 경로 지원)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
    
    # LoRA 모델 로드 및 병합
    model = PeftModel.from_pretrained(base_model, checkpoint_path).merge_and_unload()
    
    print("LoRA 모델 로드 및 병합 완료")
    print(f"병합된 모델 타입: {type(model)}")
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # GPU 메모리 사용량 확인
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU 메모리 사용량: {gpu_memory_used:.1f}MB")
    
    return model


def apply_quantization(model, save_dir: str = "bnb-4bit"):
    """
    4-bit 양자화 적용 (노트북 최적화)
    
    Args:
        model: 병합된 모델
        save_dir (str): 양자화된 모델 저장 경로
        
    Returns:
        양자화된 모델
    """
    print("=== 4-bit 양자화 적용 (노트북 최적화) ===")
    
    # 임시 디렉토리를 사용하여 양자화 과정 수행
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 먼저 fp16/fp32 형식으로 저장
        model.save_pretrained(tmp_path)
        print("임시 모델 저장 완료")
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU 메모리 정리 완료")
        
        # 4-bit 양자화 적용하여 다시 로드
        quantization_config = {
            "load_in_4bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
        model = AutoModelForSequenceClassification.from_pretrained(
            tmp_path, 
            **quantization_config
        )
        
        print("4-bit 양자화 적용 완료")
        print(f"양자화된 모델 타입: {type(model)}")
        
        # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
            print(f"양자화 후 GPU 메모리 사용량: {gpu_memory_used:.1f}MB")
    
    return model


def save_quantized_model(model, tokenizer, save_dir: str = "bnb-4bit"):
    """
    양자화된 모델 저장 (Windows 경로 지원)
    
    Args:
        model: 양자화된 모델
        tokenizer: 토크나이저
        save_dir (str): 저장 경로
    """
    print(f"=== 양자화된 모델 저장 (Windows 경로): {save_dir} ===")
    
    # Windows 경로 정규화
    save_dir = Path(save_dir)
    
    # 저장 디렉토리 생성
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 토크나이저 저장
    tokenizer.save_pretrained(save_dir)
    
    # 양자화된 모델 저장
    model.save_pretrained(save_dir)
    
    print(f"양자화된 모델 저장 완료: {save_dir}")
    
    # 저장된 파일들 확인
    print("\n저장된 파일들:")
    for file in save_dir.glob("*"):
        print(f"- {file.name}")
    
    # 저장된 모델 크기 확인
    model_size_mb = get_model_size_mb(str(save_dir))
    print(f"\n저장된 모델 크기: {model_size_mb:.1f}MB")


def get_model_size_mb(model_path: str) -> float:
    """
    모델 파일의 크기를 MB 단위로 계산
    
    Args:
        model_path (str): 모델 경로
        
    Returns:
        float: 모델 크기 (MB)
    """
    total_size = 0
    model_path = Path(model_path)
    for file_path in model_path.rglob("*.bin"):
        total_size += file_path.stat().st_size
    return total_size / (1024 * 1024)  # MB로 변환


def compare_model_sizes(original_size_mb: float, quantized_path: str):
    """
    모델 크기 비교
    
    Args:
        original_size_mb (float): 원본 모델 크기 (MB)
        quantized_path (str): 양자화된 모델 경로
    """
    print("=== 모델 크기 비교 ===")
    
    # 양자화된 모델 크기
    quantized_size_mb = get_model_size_mb(quantized_path)
    
    print(f"원본 모델 (예상): {original_size_mb:.1f} MB")
    print(f"양자화된 모델: {quantized_size_mb:.1f} MB")
    print(f"크기 감소율: {((original_size_mb - quantized_size_mb) / original_size_mb * 100):.1f}%")
    
    # 노트북 최적화 정보
    print(f"\n💡 노트북 최적화 효과:")
    print(f"   - 메모리 사용량: 약 {((original_size_mb - quantized_size_mb) / original_size_mb * 100):.0f}% 절약")
    print(f"   - 추론 속도: 더 빠른 로딩 및 추론")
    print(f"   - 배터리 효율: 더 낮은 전력 소모")


def test_quantized_model(model_path: str, test_texts: List[str]):
    """
    양자화된 모델 테스트 (노트북 최적화)
    
    Args:
        model_path (str): 모델 경로
        test_texts (List[str]): 테스트할 텍스트 리스트
        
    Returns:
        List[Dict]: 예측 결과 리스트
    """
    print("=== 양자화된 모델 테스트 (노트북 최적화) ===")
    
    # 모델과 토크나이저 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    results = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n테스트 {i}/{len(test_texts)}:")
        
        # 프롬프트 적용
        prompt_text = f"다음 문장이 긍정인지 부정인지 판단하세요.\n\n### 문장:\n{text}"
        
        # 토크나이징
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
        
        label = "toxic" if predicted_class == 0 else "none"
        confidence = predictions[0][predicted_class].item()
        
        results.append({
            'text': text,
            'prediction': label,
            'confidence': confidence
        })
        
        print(f"텍스트: {text}")
        print(f"예측: {label} (신뢰도: {confidence:.3f})")
        
        # GPU 메모리 사용량 표시
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
            print(f"GPU 메모리 사용량: {gpu_memory_used:.1f}MB")
    
    return results


def generate_usage_example(save_dir: str = "bnb-4bit"):
    """
    사용법 예시 코드 생성 (Windows 최적화)
    
    Args:
        save_dir (str): 모델 저장 경로
    """
    usage_example = f'''
# 양자화된 모델 사용 예시 (Windows 노트북 최적화)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 모델과 토크나이저 로드 (노트북 최적화)
model = AutoModelForSequenceClassification.from_pretrained(
    '{save_dir}', 
    device_map='auto',
    torch_dtype=torch.float16  # 메모리 절약
)
tokenizer = AutoTokenizer.from_pretrained('{save_dir}')

# 예측 함수
def predict_toxic(text):
    prompt = f"다음 문장이 긍정인지 부정인지 판단하세요.\\n\\n### 문장:\\n{{text}}"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    return "toxic" if predicted_class == 0 else "none"

# 사용 예시
result = predict_toxic("테스트 문장")
print(f"결과: {{result}}")

# GPU 메모리 사용량 확인
if torch.cuda.is_available():
    gpu_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"GPU 메모리 사용량: {{gpu_memory:.1f}}MB")
'''
    
    print("=== 양자화된 모델 사용법 (Windows 노트북 최적화) ===")
    print(usage_example)
    
    # 사용법을 파일로 저장 (Windows 경로)
    usage_file = Path(save_dir) / "usage_example.py"
    with open(usage_file, 'w', encoding='utf-8') as f:
        f.write(usage_example)
    
    print(f"사용법 예시가 {usage_file}에 저장되었습니다.")


# ============================================================================
# 양자화 설정값 - 여기서 직접 수정하세요 (Windows 노트북 최적화)
# ============================================================================

# 필수 설정
CHECKPOINT_PATH = "model-checkpoints\\kobert\\checkpoint-1100"  # LoRA 체크포인트 경로 (Windows 경로)

# 선택적 설정
BASE_MODEL = "skt/kobert-base-v1"  # 기본 모델 경로
SAVE_DIR = "bnb-4bit"  # 양자화된 모델 저장 경로

# 테스트 설정
RUN_TEST = True  # 양자화된 모델 테스트 실행 여부
TEST_TEXTS = [
    "안녕하세요! 좋은 하루 되세요.",
    "이런 말은 하면 안 됩니다.",
    "정말 멋진 프로젝트네요!"
]

# ============================================================================
# 메인 실행 코드
# ============================================================================

if __name__ == "__main__":
    print("🚀 한국어 악성 댓글 분류 모델 양자화 (Windows 11 + GPU 노트북)")
    print("=" * 70)
    print()
    
    try:
        # 기본 모델 로드
        base_model, config = load_base_model(BASE_MODEL)
        
        # LoRA 모델 로드 및 병합
        model = load_and_merge_lora_model(base_model, CHECKPOINT_PATH)
        
        # 양자화 적용
        quantized_model = apply_quantization(model, SAVE_DIR)
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
        
        # 양자화된 모델 저장
        save_quantized_model(quantized_model, tokenizer, SAVE_DIR)
        
        # 모델 크기 비교
        original_size_mb = 420  # KoBERT 기본 모델 크기
        compare_model_sizes(original_size_mb, SAVE_DIR)
        
        # 테스트 실행 (선택사항)
        if RUN_TEST:
            test_quantized_model(SAVE_DIR, TEST_TEXTS)
        
        # 사용법 예시 생성
        generate_usage_example(SAVE_DIR)
        
        print("\n" + "=" * 70)
        print("=== 양자화 완료 ===")
        print(f"양자화된 모델이 {SAVE_DIR}에 저장되었습니다.")
        print("이제 inference.py 스크립트나 사용법 예시를 참고하여 모델을 사용할 수 있습니다.")
        print()
        print("💡 Windows 노트북 최적화 효과:")
        print("   - 메모리 사용량 대폭 감소")
        print("   - 추론 속도 향상")
        print("   - 배터리 효율성 개선")
        print("   - 더 안정적인 실행")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("체크포인트 경로를 확인하고 다시 시도해주세요.")
        print("💡 노트북 GPU 메모리가 부족한 경우 다른 프로그램을 종료하고 다시 시도하세요.")
        exit(1) 
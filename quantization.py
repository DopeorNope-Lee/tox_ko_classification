#!/usr/bin/env python3
"""
모델 양자화 스크립트

이 스크립트는 학습된 한국어 악성 댓글 분류 모델을 4-bit 양자화하여 
메모리 사용량을 줄이고 추론 속도를 향상시킵니다.

사용법:
    python quantization.py --checkpoint_path model-checkpoints/kobert/checkpoint-1100
"""

import os
import argparse
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


def setup_environment():
    """환경 설정 및 GPU 확인"""
    print("=== 환경 설정 ===")
    
    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")
    
    print("환경 설정 완료\n")


def load_base_model(base_model_name: str = "skt/kobert-base-v1"):
    """
    기본 KoBERT 모델 로드
    
    Args:
        base_model_name (str): 기본 모델 경로
        
    Returns:
        tuple: (base_model, config)
    """
    print("=== 기본 모델 로드 ===")
    
    # 모델 설정 (2개 클래스 분류)
    config = AutoConfig.from_pretrained(
        base_model_name, 
        num_labels=2, 
        problem_type="single_label_classification"
    )
    
    # 기본 모델 로드
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, config=config)
    
    print(f"기본 모델 로드 완료: {base_model_name}")
    print(f"모델 타입: {type(base_model)}")
    print(f"파라미터 수: {sum(p.numel() for p in base_model.parameters()):,}")
    
    return base_model, config


def load_and_merge_lora_model(base_model, checkpoint_path: str):
    """
    학습된 LoRA 모델 로드 및 병합
    
    Args:
        base_model: 기본 모델
        checkpoint_path (str): LoRA 체크포인트 경로
        
    Returns:
        병합된 모델
    """
    print(f"=== LoRA 모델 로드 및 병합: {checkpoint_path} ===")
    
    # 체크포인트 경로 확인
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
    
    # LoRA 모델 로드 및 병합
    model = PeftModel.from_pretrained(base_model, checkpoint_path).merge_and_unload()
    
    print("LoRA 모델 로드 및 병합 완료")
    print(f"병합된 모델 타입: {type(model)}")
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def apply_quantization(model, save_dir: str = "bnb-4bit"):
    """
    4-bit 양자화 적용
    
    Args:
        model: 병합된 모델
        save_dir (str): 양자화된 모델 저장 경로
        
    Returns:
        양자화된 모델
    """
    print("=== 4-bit 양자화 적용 ===")
    
    # 임시 디렉토리를 사용하여 양자화 과정 수행
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 먼저 fp16/fp32 형식으로 저장
        model.save_pretrained(tmp_path)
        print("임시 모델 저장 완료")
        
        # 4-bit 양자화 적용하여 다시 로드
        model = AutoModelForSequenceClassification.from_pretrained(
            tmp_path, 
            load_in_4bit=True, 
            device_map="auto"
        )
        
        print("4-bit 양자화 적용 완료")
        print(f"양자화된 모델 타입: {type(model)}")
    
    return model


def save_quantized_model(model, tokenizer, save_dir: str = "bnb-4bit"):
    """
    양자화된 모델 저장
    
    Args:
        model: 양자화된 모델
        tokenizer: 토크나이저
        save_dir (str): 저장 경로
    """
    print(f"=== 양자화된 모델 저장: {save_dir} ===")
    
    # 저장 디렉토리 생성
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 토크나이저 저장
    tokenizer.save_pretrained(save_dir)
    
    # 양자화된 모델 저장
    model.save_pretrained(save_dir)
    
    print(f"양자화된 모델 저장 완료: {save_dir}")
    
    # 저장된 파일들 확인
    print("\n저장된 파일들:")
    for file in Path(save_dir).glob("*"):
        print(f"- {file.name}")


def get_model_size_mb(model_path: str) -> float:
    """
    모델 파일의 크기를 MB 단위로 계산
    
    Args:
        model_path (str): 모델 경로
        
    Returns:
        float: 모델 크기 (MB)
    """
    total_size = 0
    for file_path in Path(model_path).rglob("*.bin"):
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


def test_quantized_model(model_path: str, test_texts: List[str]):
    """
    양자화된 모델 테스트
    
    Args:
        model_path (str): 모델 경로
        test_texts (List[str]): 테스트할 텍스트 리스트
        
    Returns:
        List[Dict]: 예측 결과 리스트
    """
    print("=== 양자화된 모델 테스트 ===")
    
    # 모델과 토크나이저 로드
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    results = []
    
    for text in test_texts:
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
        print()
    
    return results


def generate_usage_example(save_dir: str = "bnb-4bit"):
    """
    사용법 예시 코드 생성
    
    Args:
        save_dir (str): 모델 저장 경로
    """
    usage_example = f'''
# 양자화된 모델 사용 예시
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 모델과 토크나이저 로드
model = AutoModelForSequenceClassification.from_pretrained('{save_dir}', device_map='auto')
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
'''
    
    print("=== 양자화된 모델 사용법 ===")
    print(usage_example)
    
    # 사용법을 파일로 저장
    usage_file = Path(save_dir) / "usage_example.py"
    with open(usage_file, 'w', encoding='utf-8') as f:
        f.write(usage_example)
    
    print(f"사용법 예시가 {usage_file}에 저장되었습니다.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="모델 양자화")
    
    # 필수 인자
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True,
        help="LoRA 체크포인트 경로 (예: model-checkpoints/kobert/checkpoint-1100)"
    )
    
    # 선택적 인자
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="skt/kobert-base-v1",
        help="기본 모델 경로"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="bnb-4bit",
        help="양자화된 모델 저장 경로"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="양자화된 모델 테스트 실행"
    )
    
    args = parser.parse_args()
    
    # 환경 설정
    setup_environment()
    
    try:
        # 기본 모델 로드
        base_model, config = load_base_model(args.base_model)
        
        # LoRA 모델 로드 및 병합
        model = load_and_merge_lora_model(base_model, args.checkpoint_path)
        
        # 양자화 적용
        quantized_model = apply_quantization(model, args.save_dir)
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
        
        # 양자화된 모델 저장
        save_quantized_model(quantized_model, tokenizer, args.save_dir)
        
        # 모델 크기 비교
        original_size_mb = 420  # KoBERT 기본 모델 크기
        compare_model_sizes(original_size_mb, args.save_dir)
        
        # 테스트 실행 (선택사항)
        if args.test:
            test_texts = [
                "안녕하세요! 좋은 하루 되세요.",
                "이런 말은 하면 안 됩니다.",
                "정말 멋진 프로젝트네요!"
            ]
            test_quantized_model(args.save_dir, test_texts)
        
        # 사용법 예시 생성
        generate_usage_example(args.save_dir)
        
        print("\n=== 양자화 완료 ===")
        print(f"양자화된 모델이 {args.save_dir}에 저장되었습니다.")
        print("이제 inference.py 스크립트나 사용법 예시를 참고하여 모델을 사용할 수 있습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("체크포인트 경로를 확인하고 다시 시도해주세요.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
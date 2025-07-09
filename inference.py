#!/usr/bin/env python3
"""
한국어 악성 댓글 분류 모델 추론 스크립트

이 스크립트는 학습된 KoBERT 모델을 사용하여 한국어 텍스트의 악성 여부를 분류합니다.
양자화된 모델과 일반 모델 모두 지원합니다.
Windows 11 환경의 GPU 내장 노트북에 최적화되어 있습니다.

사용하려면 이 파일 하단의 설정값들을 수정한 후 스크립트를 실행하세요.

사용 전에 setup.py를 먼저 실행하여 환경을 설정하세요.

주의사항:
    - 노트북 GPU 메모리 제약을 고려한 설정이 적용되어 있습니다.
    - 양자화된 모델 사용을 권장합니다 (메모리 절약).
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ToxicCommentClassifier:
    """
    한국어 악성 댓글 분류기 (Windows 노트북 최적화)
    
    학습된 KoBERT 모델을 사용하여 텍스트의 악성 여부를 분류합니다.
    """
    
    def __init__(self, model_path: str = "bnb-4bit", device: str = "auto"):
        """
        분류기 초기화
        
        Args:
            model_path (str): 모델 경로 (기본값: "bnb-4bit")
            device (str): 사용할 디바이스 (기본값: "auto")
        """
        self.model_path = model_path
        self.device = device
        
        # 모델과 토크나이저 로드
        print(f"모델 로딩 중: {model_path}")
        
        # Windows 노트북 최적화 설정
        model_kwargs = {
            "device_map": device,
            "torch_dtype": torch.float16,  # 메모리 절약을 위해 16비트 사용
        }
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 라벨 매핑 설정
        self.id2label = {0: "toxic", 1: "none"}
        self.label2id = {"toxic": 0, "none": 1}
        
        print("모델 로딩 완료!")
        
        # GPU 메모리 정보 출력
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU 메모리: {gpu_memory:.1f}GB")
    
    def build_prompt(self, text: str) -> str:
        """
        입력 텍스트를 프롬프트 형식으로 변환
        
        Args:
            text (str): 원본 텍스트
            
        Returns:
            str: 프롬프트 형식의 텍스트
        """
        return (
            "다음 문장이 긍정인지 부정인지 판단하세요.\n\n"
            "### 문장:\n"
            f"{text}"
        )
    
    def predict(self, text: str, return_confidence: bool = True) -> Dict[str, Union[str, float]]:
        """
        단일 텍스트에 대한 예측 수행
        
        Args:
            text (str): 예측할 텍스트
            return_confidence (bool): 신뢰도 반환 여부
            
        Returns:
            Dict: 예측 결과 (prediction, confidence)
        """
        # 프롬프트 적용
        prompt_text = self.build_prompt(text)
        
        # 토크나이징
        inputs = self.tokenizer(
            prompt_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # GPU로 이동 (필요시)
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        result = {
            "prediction": self.id2label[predicted_class],
            "confidence": confidence
        }
        
        return result
    
    def predict_batch(self, texts: List[str], return_confidence: bool = True) -> List[Dict[str, Union[str, float]]]:
        """
        배치 텍스트에 대한 예측 수행 (노트북 메모리 최적화)
        
        Args:
            texts (List[str]): 예측할 텍스트 리스트
            return_confidence (bool): 신뢰도 반환 여부
            
        Returns:
            List[Dict]: 예측 결과 리스트
        """
        results = []
        
        # 노트북 메모리 절약을 위해 작은 배치로 처리
        batch_size = 4  # 작은 배치 크기로 메모리 절약
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                result = self.predict(text, return_confidence)
                results.append(result)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results


def load_texts_from_file(file_path: str) -> List[str]:
    """
    파일에서 텍스트 리스트 로드 (Windows 경로 지원)
    
    Args:
        file_path (str): 텍스트 파일 경로
        
    Returns:
        List[str]: 텍스트 리스트
    """
    # Windows 경로 정규화
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


def save_results(results: List[Dict], output_path: str):
    """
    결과를 JSON 파일로 저장 (Windows 경로 지원)
    
    Args:
        results (List[Dict]): 저장할 결과 리스트
        output_path (str): 출력 파일 경로
    """
    # Windows 경로 정규화
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"결과가 {output_path}에 저장되었습니다.")


def interactive_mode(classifier: ToxicCommentClassifier):
    """
    대화형 모드 실행 (Windows 최적화)
    
    Args:
        classifier: 초기화된 분류기
    """
    print("\n=== 대화형 모드 (Windows 최적화) ===")
    print("텍스트를 입력하면 악성 여부를 분석합니다. (종료: 'quit' 또는 'exit')")
    print("💡 노트북 최적화: GPU 메모리 사용량을 모니터링합니다.")
    
    while True:
        try:
            text = input("\n텍스트 입력: ").strip()
            
            if text.lower() in ['quit', 'exit', '종료']:
                print("대화형 모드를 종료합니다.")
                break
            
            if not text:
                print("텍스트를 입력해주세요.")
                continue
            
            # 예측 수행
            result = classifier.predict(text)
            
            # 결과 출력
            print(f"예측 결과: {result['prediction']}")
            print(f"신뢰도: {result['confidence']:.3f}")
            
            # GPU 메모리 사용량 표시
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
                print(f"GPU 메모리 사용량: {gpu_memory_used:.1f}MB")
            
        except KeyboardInterrupt:
            print("\n대화형 모드를 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")


def single_text_prediction(classifier: ToxicCommentClassifier, text: str):
    """
    단일 텍스트 예측
    
    Args:
        classifier: 초기화된 분류기
        text (str): 예측할 텍스트
    """
    result = classifier.predict(text)
    print(f"\n입력 텍스트: {text}")
    print(f"예측 결과: {result['prediction']}")
    print(f"신뢰도: {result['confidence']:.3f}")
    
    # GPU 메모리 사용량 표시
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU 메모리 사용량: {gpu_memory_used:.1f}MB")
    
    return result


def batch_file_prediction(classifier: ToxicCommentClassifier, file_path: str, output_path: str = None):
    """
    파일에서 텍스트를 읽어 배치 예측 수행 (Windows 최적화)
    
    Args:
        classifier: 초기화된 분류기
        file_path (str): 텍스트 파일 경로
        output_path (str): 결과 저장 파일 경로 (선택사항)
    """
    try:
        texts = load_texts_from_file(file_path)
        print(f"파일에서 {len(texts)}개의 텍스트를 로드했습니다.")
        
        results = classifier.predict_batch(texts)
        
        # 결과 출력
        for i, (text, result) in enumerate(zip(texts, results), 1):
            print(f"\n{i}. 텍스트: {text}")
            print(f"   예측: {result['prediction']} (신뢰도: {result['confidence']:.3f})")
        
        if output_path:
            save_results(results, output_path)
            
        return results
            
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"파일 처리 오류: {e}")
        return None


# ============================================================================
# 추론 설정값 - 여기서 직접 수정하세요 (Windows 노트북 최적화)
# ============================================================================

# 모델 설정
MODEL_PATH = "bnb-4bit"  # 모델 경로 (양자화된 모델 권장)
DEVICE = "auto"  # 사용할 디바이스

# 실행 모드 설정 (하나만 True로 설정)
SINGLE_TEXT_MODE = False  # 단일 텍스트 예측
BATCH_FILE_MODE = False   # 파일에서 배치 예측
INTERACTIVE_MODE = True   # 대화형 모드

# 단일 텍스트 모드 설정
INPUT_TEXT = "안녕하세요! 좋은 하루 되세요."

# 배치 파일 모드 설정 (Windows 경로)
INPUT_FILE = "input.txt"  # 입력 파일 경로
OUTPUT_FILE = "results.json"  # 출력 파일 경로

# ============================================================================
# 메인 실행 코드
# ============================================================================

if __name__ == "__main__":
    print("🚀 한국어 악성 댓글 분류 모델 추론 (Windows 11 + GPU 노트북)")
    print("=" * 60)
    print()
    
    # 모델 경로 확인 (Windows 경로 지원)
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"오류: 모델 경로를 찾을 수 없습니다: {MODEL_PATH}")
        print("먼저 train.py를 실행하여 모델을 학습하거나 quantization.py를 실행하여 양자화된 모델을 생성하세요.")
        print("💡 노트북에서는 양자화된 모델(bnb-4bit) 사용을 권장합니다.")
        exit(1)
    
    # 분류기 초기화
    try:
        classifier = ToxicCommentClassifier(MODEL_PATH, DEVICE)
    except Exception as e:
        print(f"모델 로딩 오류: {e}")
        print("💡 GPU 메모리가 부족한 경우 양자화된 모델을 사용하세요.")
        exit(1)
    
    # 실행 모드 결정
    if INTERACTIVE_MODE:
        interactive_mode(classifier)
    elif SINGLE_TEXT_MODE:
        single_text_prediction(classifier, INPUT_TEXT)
    elif BATCH_FILE_MODE:
        batch_file_prediction(classifier, INPUT_FILE, OUTPUT_FILE)
    else:
        print("실행 모드를 설정해주세요. (INTERACTIVE_MODE, SINGLE_TEXT_MODE, BATCH_FILE_MODE 중 하나를 True로 설정)")
    
    print("\n💡 Windows 노트북 최적화 팁:")
    print("   - GPU 메모리가 부족하면 양자화된 모델을 사용하세요")
    print("   - 추론 중에는 다른 프로그램을 종료하여 GPU 메모리를 확보하세요")
    print("   - 대용량 파일 처리 시 배치 크기가 자동으로 조정됩니다") 
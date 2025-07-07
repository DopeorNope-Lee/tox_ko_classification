#!/usr/bin/env python3
"""
한국어 악성 댓글 분류 모델 추론 스크립트

이 스크립트는 학습된 KoBERT 모델을 사용하여 한국어 텍스트의 악성 여부를 분류합니다.
양자화된 모델과 일반 모델 모두 지원합니다.

사용법:
    python inference.py --text "분석할 텍스트"
    python inference.py --file input.txt
    python inference.py --interactive
"""

import argparse
import torch
import json
from pathlib import Path
from typing import Dict, List, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ToxicCommentClassifier:
    """
    한국어 악성 댓글 분류기
    
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
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 라벨 매핑 설정
        self.id2label = {0: "toxic", 1: "none"}
        self.label2id = {"toxic": 0, "none": 1}
        
        print("모델 로딩 완료!")
    
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
        배치 텍스트에 대한 예측 수행
        
        Args:
            texts (List[str]): 예측할 텍스트 리스트
            return_confidence (bool): 신뢰도 반환 여부
            
        Returns:
            List[Dict]: 예측 결과 리스트
        """
        results = []
        
        for text in texts:
            result = self.predict(text, return_confidence)
            results.append(result)
        
        return results


def load_texts_from_file(file_path: str) -> List[str]:
    """
    파일에서 텍스트 리스트 로드
    
    Args:
        file_path (str): 텍스트 파일 경로
        
    Returns:
        List[str]: 텍스트 리스트
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


def save_results(results: List[Dict], output_path: str):
    """
    결과를 JSON 파일로 저장
    
    Args:
        results (List[Dict]): 저장할 결과 리스트
        output_path (str): 출력 파일 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"결과가 {output_path}에 저장되었습니다.")


def interactive_mode(classifier: ToxicCommentClassifier):
    """
    대화형 모드 실행
    
    Args:
        classifier: 초기화된 분류기
    """
    print("\n=== 대화형 모드 ===")
    print("텍스트를 입력하면 악성 여부를 분석합니다. (종료: 'quit' 또는 'exit')")
    
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
            
        except KeyboardInterrupt:
            print("\n대화형 모드를 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="한국어 악성 댓글 분류 모델 추론",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python inference.py --text "분석할 텍스트"
  python inference.py --file input.txt --output results.json
  python inference.py --interactive
        """
    )
    
    # 인자 설정
    parser.add_argument(
        "--text", 
        type=str, 
        help="분석할 단일 텍스트"
    )
    parser.add_argument(
        "--file", 
        type=str, 
        help="분석할 텍스트가 포함된 파일 경로"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="대화형 모드 실행"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="bnb-4bit", 
        help="모델 경로 (기본값: bnb-4bit)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="결과 저장 파일 경로 (JSON 형식)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="사용할 디바이스 (기본값: auto)"
    )
    
    args = parser.parse_args()
    
    # 모델 경로 확인
    if not Path(args.model).exists():
        print(f"오류: 모델 경로를 찾을 수 없습니다: {args.model}")
        print("먼저 train.ipynb를 실행하여 모델을 학습하거나 quantization.ipynb를 실행하여 양자화된 모델을 생성하세요.")
        return
    
    # 분류기 초기화
    try:
        classifier = ToxicCommentClassifier(args.model, args.device)
    except Exception as e:
        print(f"모델 로딩 오류: {e}")
        return
    
    # 실행 모드 결정
    if args.interactive:
        interactive_mode(classifier)
    elif args.text:
        # 단일 텍스트 예측
        result = classifier.predict(args.text)
        print(f"\n입력 텍스트: {args.text}")
        print(f"예측 결과: {result['prediction']}")
        print(f"신뢰도: {result['confidence']:.3f}")
        
        if args.output:
            save_results([result], args.output)
    elif args.file:
        # 파일에서 텍스트 로드 및 예측
        try:
            texts = load_texts_from_file(args.file)
            print(f"파일에서 {len(texts)}개의 텍스트를 로드했습니다.")
            
            results = classifier.predict_batch(texts)
            
            # 결과 출력
            for i, (text, result) in enumerate(zip(texts, results), 1):
                print(f"\n{i}. 텍스트: {text}")
                print(f"   예측: {result['prediction']} (신뢰도: {result['confidence']:.3f})")
            
            if args.output:
                save_results(results, args.output)
                
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다: {args.file}")
        except Exception as e:
            print(f"파일 처리 오류: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 
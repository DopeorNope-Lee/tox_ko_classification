"""
한국어 악성 댓글 분류 모델 추론 스크립트

이 스크립트는 4-bit로 양자화되고 병합된 KoBERT 모델을 사용하여 추론합니다.

사용법:
    python inference.py --text "분석할 텍스트"
    python inference.py --file input.txt
    python inference.py --interactive
"""
import os
# -- 환경 설정 --
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
# PeftModel은 더 이상 필요 없으며, BitsAndBytesConfig를 추가합니다.
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

# -- 모델 설정 --
CONFIG = {
    "model_path": "checkpoints/kobert-lora",  # 경량화 시, checkpoints/kobert-bnb-4bit 등의 경로 입력
}

# -- 레이블 정의 --
LABEL_MAP = {
    0: "악성",
    1: "정상"
}


def load_model(model_path: str = CONFIG["model_path"]):
    """
    병합 및 4-bit 양자화된 KoBERT 모델과 토크나이저를 로드합니다.
    """
    print(f"'{model_path}'에서 병합 및 양자화된 모델을 로드하는 중...")

    # 1. 모델을 양자화할 때 사용했던 것과 동일한 BitsAndBytesConfig를 정의합니다.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 2. 최종 모델 경로에서 모델과 토크나이저를 직접 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=bnb_config, # 양자화 설정을 전달합니다.
        device_map="auto"
    )
    
    # PeftModel을 사용한 병합 로직은 더 이상 필요 없습니다.
    # base_model = ...
    # model = PeftModel.from_pretrained(...).merge_and_unload()
    
    model.eval()
    print("모델 로드 완료!")
    return tokenizer, model


def predict(texts, tokenizer, model):
    """
    입력된 텍스트 리스트에 대해 악성 여부를 예측합니다.
    (이 함수는 변경할 필요가 없습니다.)
    """
    if isinstance(texts, str):
        texts = [texts]

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    batch = {
        'input_ids': encodings['input_ids'].to(model.device),
        'attention_mask': encodings['attention_mask'].to(model.device)
    }

    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    probs = logits.softmax(dim=-1).cpu()
    labels = probs.argmax(dim=-1).tolist()

    results = []
    for i, text in enumerate(texts):
        label_id = labels[i]
        results.append({
            "text": text,
            "label_id": label_id,
            "label_name": LABEL_MAP.get(label_id, "알 수 없음"),
            "probability": float(probs[i, label_id])
        })
        
    return results


def main():
    """
    메인 실행 함수 (이 함수는 변경할 필요가 없습니다.)
    """
    parser = argparse.ArgumentParser(description="한국어 악성 댓글 분류 모델 추론 스크립트")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="분류할 단일 텍스트")
    group.add_argument("--file", type=str, help="분류할 텍스트가 담긴 파일 경로 (한 줄에 한 텍스트)")
    group.add_argument("--interactive", action="store_true", help="대화형 모드로 실행")

    args = parser.parse_args()

    tokenizer, model = load_model()

    if args.text:
        results = predict(args.text, tokenizer, model)
        for res in results:
            print(f"입력: \"{res['text']}\"")
            print(f"결과: {res['label_name']} (확률: {res['probability']:.2%})")

    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if not lines:
                print("파일이 비어있거나 유효한 텍스트가 없습니다.")
                return

            print(f"총 {len(lines)}개의 텍스트를 파일에서 읽었습니다. 분석을 시작합니다...")
            results = predict(lines, tokenizer, model)
            for res in results:
                print("-" * 30)
                print(f"입력: \"{res['text']}\"")
                print(f"결과: {res['label_name']} (확률: {res['probability']:.2%})")

        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다. -> {args.file}")
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {e}")

    elif args.interactive:
        print("\n대화형 모드를 시작합니다. 분석할 문장을 입력하세요. (종료: 'exit' 또는 'quit')")
        while True:
            try:
                user_input = input(">>> ")
                if user_input.lower() in ["exit", "quit"]:
                    print("프로그램을 종료합니다.")
                    break
                if not user_input.strip():
                    continue
                
                results = predict(user_input, tokenizer, model)
                for res in results:
                    print(f"결과: {res['label_name']} (확률: {res['probability']:.2%})\n")

            except (KeyboardInterrupt, EOFError):
                print("\n프로그램을 종료합니다.")
                break


if __name__ == "__main__":
    main()

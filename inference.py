"""
한국어 악성 댓글 분류 모델 추론 스크립트

이 스크립트는 학습된 KoBERT 모델을 사용하여 한국어 텍스트의 악성 여부를 분류합니다.
PEFT(LoRA)로 튜닝된 모델의 추론을 지원합니다.

사용법:
    python inference.py --text "분석할 텍스트"
    python inference.py --file input.txt
    python inference.py --interactive
"""
import os
# -- 환경 설정 --
# 특정 GPU만 사용하도록 설정 (예: 0번 GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel


# -- 모델 설정 --
CONFIG = {
    "base_model": "skt/kobert-base-v1",
    "lora_dir": "checkpoints/kobert-lora/checkpoint-700",
}

# -- 레이블 정의 --
# 예: 0: 악성, 1: 정상
LABEL_MAP = {
    0: "악성",
    1: "정상"
}


def load_model(model_dir: str = CONFIG["lora_dir"]):
    """
    사전 학습된 KoBERT 모델과 LoRA 어댑터를 로드하고 병합합니다.
    """
    print("모델과 토크나이저를 로드하는 중... 🐢")
    cfg = AutoConfig.from_pretrained(
        CONFIG["base_model"],
        num_labels=len(LABEL_MAP),
        problem_type="single_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    
    # 기본 모델 로드
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model"],
        config=cfg,
        device_map="auto"
    )
    
    # LoRA 가중치를 불러와 기본 모델과 병합
    model = PeftModel.from_pretrained(base_model, model_dir).merge_and_unload()
    model.eval()
    print("모델 로드 완료! ✨")
    return tokenizer, model


def predict(texts, tokenizer, model):
    """
    입력된 텍스트 리스트에 대해 악성 여부를 예측합니다.
    """
    if isinstance(texts, str):
        texts = [texts]

    # 토크나이징
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    # 모델이 기대하는 입력 형식으로 변환
    batch = {
        'input_ids': encodings['input_ids'].to(model.device),
        'attention_mask': encodings['attention_mask'].to(model.device)
    }

    # 예측 수행
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    probs = logits.softmax(dim=-1).cpu()
    labels = probs.argmax(dim=-1).tolist()

    # 결과 포맷팅
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
    메인 실행 함수: 커맨드 라인 인자를 파싱하고 적절한 모드를 실행합니다.
    """
    parser = argparse.ArgumentParser(description="한국어 악성 댓글 분류 모델 추론 스크립트")
    
    # 세 가지 모드는 동시에 사용할 수 없도록 설정
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="분류할 단일 텍스트")
    group.add_argument("--file", type=str, help="분류할 텍스트가 담긴 파일 경로 (한 줄에 한 텍스트)")
    group.add_argument("--interactive", action="store_true", help="대화형 모드로 실행")

    args = parser.parse_args()

    # 모델 로드 (한 번만 실행)
    tokenizer, model = load_model()

    if args.text:
        # -- 단일 텍스트 모드 --
        results = predict(args.text, tokenizer, model)
        for res in results:
            print(f"💬 입력: \"{res['text']}\"")
            print(f"✅ 결과: {res['label_name']} (확률: {res['probability']:.2%})")

    elif args.file:
        # -- 파일 모드 --
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
                print(f"💬 입력: \"{res['text']}\"")
                print(f"✅ 결과: {res['label_name']} (확률: {res['probability']:.2%})")

        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다. -> {args.file}")
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {e}")

    elif args.interactive:
        # -- 대화형 모드 --
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
                    print(f"✅ 결과: {res['label_name']} (확률: {res['probability']:.2%})\n")

            except (KeyboardInterrupt, EOFError):
                print("\n프로그램을 종료합니다.")
                break


if __name__ == "__main__":
    main()

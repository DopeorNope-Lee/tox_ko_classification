#!/usr/bin/env python3
"""
한국어 악성 댓글 분류 모델 학습 스크립트

이 스크립트는 한국어 댓글을 분석하여 악성 댓글과 일반 댓글을 분류하는 모델을 학습합니다.
KoBERT 기반 모델에 LoRA를 적용하여 효율적인 fine-tuning을 수행합니다.
Windows 11 환경의 GPU 내장 노트북에 최적화되어 있습니다.

학습을 시작하려면 이 파일 하단의 설정값들을 수정한 후 스크립트를 실행하세요.

사용 전에 setup.py를 먼저 실행하여 환경을 설정하세요.

주의사항:
    - 노트북 GPU 메모리 제약을 고려하여 기본 배치 크기가 작게 설정되어 있습니다.
    - GPU 메모리가 부족하면 BATCH_SIZE를 더 줄이세요.
    - 학습 중에는 다른 프로그램을 종료하여 GPU 메모리를 확보하세요.
"""

import numpy as np
import pandas as pd
import torch
import datasets
from pathlib import Path
from typing import Dict, Tuple

# Transformers 라이브러리 import
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    DataCollatorWithPadding,
    AutoConfig, 
    AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model
import evaluate


def load_and_preprocess_data(data_path: str = "./korean-malicious-comments-dataset/Dataset.csv"):
    """
    데이터 로딩 및 전처리
    
    Args:
        data_path (str): 데이터셋 파일 경로
        
    Returns:
        datasets.DatasetDict: 전처리된 데이터셋
    """
    print("=== 데이터 로딩 및 전처리 ===")
    
    # CSV 파일 로드 (Windows 경로 지원)
    try:
        df = pd.read_csv(data_path, sep='\t')
    except:
        df = pd.read_csv(data_path)
    
    print(f"원본 데이터셋 크기: {df.shape}")
    
    # 컬럼명 설정
    df.columns = ['text', 'label']
    
    # 결측값 제거
    df = df.dropna().reset_index(drop=True)
    print(f"전처리 후 데이터셋 크기: {df.shape}")
    
    # 라벨 분포 확인
    print("라벨 분포:")
    print(df['label'].value_counts())
    
    # 데이터 분할 (train: 9500, validation: 500) - 필요시 수정
    valid_df = df.sample(n=500, random_state=42)
    train_df = df.drop(valid_df.index)
    
    print(f"훈련 데이터: {len(train_df)}개")
    print(f"검증 데이터: {len(valid_df)}개")
    
    # HuggingFace Dataset 형식으로 변환
    data = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(train_df),
        'valid': datasets.Dataset.from_pandas(valid_df)
    })
    
    print("데이터 로딩 완료\n")
    return data


def build_prompt(sentence: str) -> str:
    """
    입력 문장을 프롬프트 형식으로 변환
    
    Args:
        sentence (str): 원본 문장
        
    Returns:
        str: 프롬프트 형식의 문장
    """
    return (
        "다음 문장이 긍정인지 부정인지 판단하세요.\n\n"
        "### 문장:\n"
        f"{sentence}"
    )


def get_dataset(data, tokenizer, max_len: int = 512):
    """
    데이터셋을 토크나이저를 사용하여 모델 입력 형식으로 변환
    
    Args:
        data: HuggingFace DatasetDict
        tokenizer: 토크나이저
        max_len (int): 최대 시퀀스 길이
        
    Returns:
        토크나이징된 데이터셋
    """
    def _encode(batch):
        # 텍스트를 토큰화
        enc = tokenizer(batch["text"],
                        truncation=True,
                        padding="max_length",
                        max_length=max_len)
        # 라벨을 int64 형식으로 변환
        enc["labels"] = np.array(batch["label"], dtype=np.int64)
        return enc

    # 배치 단위로 토크나이징 수행
    tokenised = data.map(_encode, batched=True, remove_columns=["text", "label"])
    # PyTorch 형식으로 설정
    tokenised.set_format("torch",
                         columns=["input_ids", "attention_mask", "labels"])
    return tokenised


def setup_model_and_tokenizer(model_path: str = "skt/kobert-base-v1"):
    """
    모델과 토크나이저 설정
    
    Args:
        model_path (str): 모델 경로
        
    Returns:
        tuple: (model, tokenizer)
    """
    print("=== 모델 및 토크나이저 설정 ===")
    
    # 모델 설정 (2개 클래스 분류)
    config = AutoConfig.from_pretrained(
        model_path, 
        num_labels=2, 
        problem_type="single_label_classification"
    )
    
    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    
    # 토크나이저 로드 (KoBERT는 fast tokenizer를 지원하지 않음)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # 라벨 매핑 설정
    model.config.id2label = {0: "toxic", 1: "none"}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    
    print(f"모델 로드 완료: {model_path}")
    print(f"라벨 매핑: 0 -> {model.config.id2label[0]}, 1 -> {model.config.id2label[1]}")
    
    return model, tokenizer


def setup_lora(model, r: int = 16, lora_alpha: int = 16, lora_dropout: float = 0.1):
    """
    LoRA 설정 및 적용
    
    Args:
        model: 기본 모델
        r (int): LoRA의 rank
        lora_alpha (int): LoRA의 alpha 값
        lora_dropout (float): Dropout 비율
        
    Returns:
        LoRA가 적용된 모델
    """
    print("=== LoRA 설정 ===")
    
    # KoBERT 모델의 LoRA 타겟 모듈 설정
    targets = ["query", "key", "value"]
    
    # LoRA 설정
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQUENCE_CLASSIFICATION"
    )
    
    # LoRA 적용
    model = get_peft_model(model, lora_cfg)
    
    # 학습 가능한 파라미터만 출력
    model.print_trainable_parameters()
    
    print("LoRA 설정 완료\n")
    return model


class SmartCollator(DataCollatorWithPadding):
    """스마트 데이터 콜레이터"""
    
    def __call__(self, features):
        batch = super().__call__(features)
        return batch


def compute_metrics(eval_pred):
    """
    평가 예측 결과를 바탕으로 메트릭을 계산
    
    Args:
        eval_pred: (logits, labels) 튜플
        
    Returns:
        Dict: 정확도와 F1 점수
    """
    logits, labels = eval_pred
    
    preds = logits.argmax(axis=-1)
    labels = labels.astype(np.int64)
    
    # 정확도 계산
    accuracy_metric = evaluate.load("accuracy")
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    
    # F1 점수 계산 (weighted average)
    f1_metric = evaluate.load("f1")
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    
    return {"accuracy": acc, "f1": f1}


def train_model(model, tokenizer, data, max_len: int = 512, batch_size: int = 32, 
                eval_batch_size: int = 8, epochs: int = 20, learning_rate: float = 5e-5,
                output_dir: str = "model-checkpoints/kobert"):
    """
    모델 학습 수행 (노트북 GPU 최적화)
    
    Args:
        model: 학습할 모델
        tokenizer: 토크나이저
        data: 데이터셋
        max_len (int): 최대 시퀀스 길이
        batch_size (int): 훈련 배치 크기 (노트북 GPU 메모리 제약 고려)
        eval_batch_size (int): 평가 배치 크기
        epochs (int): 훈련 에포크 수
        learning_rate (float): 학습률
        output_dir (str): 모델 저장 경로
        
    Returns:
        Trainer: 학습된 트레이너 객체
    """
    print("=== 모델 학습 시작 (노트북 GPU 최적화) ===")
    
    # 데이터셋 토크나이징
    ds = get_dataset(data, tokenizer, max_len)
    print("데이터셋 토크나이징 완료")
    
    # 토크나이저 임베딩 크기 조정
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # GPU 메모리 확인
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 메모리: {gpu_memory:.1f}GB")
        
        # GPU 메모리에 따른 배치 크기 조정 제안
        if gpu_memory < 4.0 and batch_size > 16:
            print(f"⚠️  GPU 메모리가 적습니다. 배치 크기를 16 이하로 줄이는 것을 권장합니다.")
        elif gpu_memory < 8.0 and batch_size > 32:
            print(f"⚠️  GPU 메모리가 제한적입니다. 배치 크기를 32 이하로 줄이는 것을 권장합니다.")
    
    # 학습 인자 설정 (노트북 최적화)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=True,  # 16비트 정밀도로 메모리 절약
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_steps=50,
        eval_steps=50,
        warmup_steps=100,
        weight_decay=0.01,
        # 노트북 최적화 설정
        dataloader_pin_memory=False,  # Windows에서 메모리 절약
        remove_unused_columns=False,  # Windows 호환성
        report_to=None,  # 로깅 비활성화로 메모리 절약
    )
    
    # 데이터 콜레이터 설정
    data_collator = SmartCollator(tokenizer, return_tensors="pt")
    
    # Trainer 객체 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 모델 학습 실행
    trainer.train()
    
    print("모델 학습 완료!")
    
    return trainer


def evaluate_model(trainer):
    """모델 평가 및 결과 출력"""
    print("=== 모델 평가 ===")
    
    # 최종 모델 평가
    final_results = trainer.evaluate()
    
    print("\n=== 최종 평가 결과 ===")
    print(f"평가 손실: {final_results['eval_loss']:.4f}")
    print(f"정확도: {final_results['eval_accuracy']:.4f}")
    print(f"F1 점수: {final_results['eval_f1']:.4f}")
    
    return final_results


def save_model(trainer, tokenizer, save_path: str = "final-model"):
    """모델 저장"""
    print(f"=== 모델 저장: {save_path} ===")
    
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"모델이 {save_path}에 저장되었습니다.")


def test_predictions(model, tokenizer, test_texts: list):
    """예측 테스트"""
    print("=== 예측 테스트 ===")
    
    def predict_text(text):
        # 프롬프트 적용
        prompt_text = build_prompt(text)
        
        # 토크나이징
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
        
        return model.config.id2label[predicted_class]
    
    for text in test_texts:
        result = predict_text(text)
        print(f"텍스트: {text}")
        print(f"예측: {result}\n")


# ============================================================================
# 학습 설정값 - 여기서 직접 수정하세요 (Windows 노트북 최적화)
# ============================================================================

# 데이터 설정
DATA_PATH = "./korean-malicious-comments-dataset/Dataset.csv"

# 모델 설정
MODEL_PATH = "skt/kobert-base-v1"
MAX_LEN = 512

# 학습 설정 (노트북 GPU 메모리 제약 고려)
BATCH_SIZE = 32      # 노트북 GPU 메모리 제약으로 작게 설정 (기존: 128)
EVAL_BATCH_SIZE = 8  # 평가 배치 크기도 작게 설정 (기존: 16)
EPOCHS = 20
LEARNING_RATE = 5e-5

# LoRA 설정
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# 저장 경로 설정 (Windows 경로)
OUTPUT_DIR = "model-checkpoints\\kobert"
SAVE_PATH = "final-model"

# 테스트 텍스트
TEST_TEXTS = [
    "안녕하세요! 좋은 하루 되세요.",
    "이런 말은 하면 안 됩니다.",
    "정말 멋진 프로젝트네요!"
]

# ============================================================================
# 메인 실행 코드
# ============================================================================

if __name__ == "__main__":
    print("🚀 한국어 악성 댓글 분류 모델 학습 (Windows 11 + GPU 노트북)")
    print("=" * 60)
    print()
    
    # 데이터 로딩 및 전처리
    data = load_and_preprocess_data(DATA_PATH)
    
    # 텍스트에 프롬프트 적용
    data = data.map(lambda x: {"text": build_prompt(x["text"])})
    
    # 모델 및 토크나이저 설정
    model, tokenizer = setup_model_and_tokenizer(MODEL_PATH)
    
    # LoRA 설정
    model = setup_lora(model, LORA_R, LORA_ALPHA, LORA_DROPOUT)
    
    # 모델 학습
    trainer = train_model(
        model, tokenizer, data, 
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        output_dir=OUTPUT_DIR
    )
    
    # 모델 평가
    results = evaluate_model(trainer)
    
    # 모델 저장
    save_model(trainer, tokenizer, SAVE_PATH)
    
    # 예측 테스트
    test_predictions(model, tokenizer, TEST_TEXTS)
    
    print("=" * 60)
    print("=== 학습 완료 ===")
    print(f"최종 정확도: {results['eval_accuracy']:.4f}")
    print(f"최종 F1 점수: {results['eval_f1']:.4f}")
    print()
    print("💡 노트북 최적화 팁:")
    print("   - GPU 메모리가 부족하면 BATCH_SIZE를 더 줄이세요 (16 또는 8)")
    print("   - 학습 중에는 다른 프로그램을 종료하여 GPU 메모리를 확보하세요")
    print("   - 양자화를 위해 quantization.py를 실행하세요") 
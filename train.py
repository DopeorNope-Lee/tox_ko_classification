#!/usr/bin/env python3
"""
한국어 악성 댓글 분류 모델 학습 스크립트

이 스크립트는 한국어 댓글을 분석하여 악성 댓글과 일반 댓글을 분류하는 모델을 학습합니다.
KoBERT 기반 모델에 LoRA를 적용하여 효율적인 fine-tuning을 수행합니다.

사용법:
    python train.py
"""
# train.py
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('./')
from pathlib import Path
import logging
from transformers import Trainer, TrainingArguments
from utils.data import build_dataset
from utils.modeling import build_lora_model
from utils.collator import SmartCollator
from utils.metric import compute_metrics

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)

# ──────────────────────────────
CONFIG = {
    "model_name":  "skt/kobert-base-v1",
    "data_dir":    "data",
    "csv_file":    "train.csv",
    "output_dir":  "checkpoints/kobert-lora",
    "epochs":      5,
    "batch_size":  32,
    "lr":          2e-5,
    "data_size": 100 # 학습에 활용할 데이터의 크기를 정할 수 있습니다.
}
# ──────────────────────────────

def train(cfg: dict = CONFIG):
    ds, tok = build_dataset(csv_path= Path(cfg["data_dir"]) / cfg["csv_file"], data_size=cfg["data_size"])
    model   = build_lora_model(cfg["model_name"], num_labels=2)

    tr_args = TrainingArguments(
        output_dir               = cfg["output_dir"],
        eval_strategy      = "steps",
        save_strategy      = "steps",
        per_device_train_batch_size = cfg["batch_size"],
        per_device_eval_batch_size  = cfg["batch_size"],
        learning_rate            = cfg["lr"],
        num_train_epochs         = cfg["epochs"],
        fp16                     = True,
        logging_steps            = 1,
        save_steps               = 50,
        eval_steps               = 50,
        # save_total_limit         = 3, # total 제한 없이 저장
        load_best_model_at_end   = True,
        metric_for_best_model    = "eval_loss",
    )

    trainer = Trainer(
        model           = model,
        args            = tr_args,
        train_dataset   = ds["train"],
        eval_dataset    = ds["valid"],
        data_collator   = SmartCollator(tok),
        compute_metrics = compute_metrics,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    logging.info("Finished training & saved to %s", cfg["output_dir"])

# 스크립트 직접 실행 시
if __name__ == "__main__":
    train()

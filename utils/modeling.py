from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, AutoConfig)
from peft import LoraConfig, get_peft_model
import torch

def build_lora_model(base_model:str, num_labels:int):
    cfg   = AutoConfig.from_pretrained(base_model,
                                       num_labels=num_labels,
                                       problem_type="single_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(base_model, config=cfg)
    targets = ["query", "key", "value"]
    lora   = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1,
                        bias="none", task_type="SEQ_CLS",
                        target_modules=targets)
    return get_peft_model(model, lora)

def get_tokenizer(model_name:str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=False)

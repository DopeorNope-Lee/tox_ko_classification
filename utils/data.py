# utils/data.py

from datasets import DatasetDict, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from utils.prompts import CLS_SYSTEM_PROMPT, CLS_USER_PROMPT  # prompts.py에서 프롬프트 임포트

def build_dataset(csv_path, text_col="text", label_col="label", valid_size=0.1, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[label_col])
    df = df.reset_index(drop=True)
    df[label_col] = df[label_col].astype(int)
    
    train_df, valid_df = train_test_split(df, test_size=valid_size, stratify=df[label_col], random_state=seed)

    dset = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "valid": Dataset.from_pandas(valid_df.reset_index(drop=True)),
    })
    # HuggingFace Trainer expects columns: input_ids, attention_mask, labels
    return add_encoding_columns(dset, text_col, label_col)

def add_encoding_columns(ds, text_col, label_col, model_name="skt/kobert-base-v1"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tokenize(batch):
        # 배치 데이터를 받아서 처리
        prompts = [f"{CLS_SYSTEM_PROMPT}{CLS_USER_PROMPT.format(text=text)}" for text in batch[text_col]]
        # 배치 데이터를 토큰화
        return tok(prompts, truncation=True, padding='max_length', max_length=512)

    # 'batched=True'로 배치 처리, 'remove_columns'로 텍스트 컬럼 제거
    ds = ds.map(tokenize, batched=True, remove_columns=[text_col])
    ds = ds.rename_column(label_col, "labels")
    return ds, tok

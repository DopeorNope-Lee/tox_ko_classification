from datasets import DatasetDict, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from utils.prompts import CLS_SYSTEM_PROMPT, CLS_USER_PROMPT  # prompts.py에서 프롬프트 임포트

def build_dataset(csv_path, data_size=None, text_col="document", label_col="label", valid_size=0.1, seed=42):
    """
    CSV 파일을 로드하고, 데이터셋 크기를 조절하며, HuggingFace Dataset으로 변환합니다.
    """
    df = pd.read_csv(csv_path)
    
    if data_size:
        print(f"데이터셋 크기를 {data_size}로 제한합니다.")
        df = df.head(data_size)

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
    """
    데이터셋에 토크나이징된 인코딩 컬럼을 추가합니다.
    """
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

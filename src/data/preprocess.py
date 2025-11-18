import re
from typing import Tuple, Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # assuming columns: "statement", "status"
    df = df[["statement", "status"]].dropna()
    df["clean_text"] = df["statement"].apply(clean_text)
    return df


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["status"])
    return df, le


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 911,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=df["label"],
    )
    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val_size,
        random_state=random_state,
        stratify=temp_df["label"],
    )
    return train_df, val_df, test_df


def build_vocab(
    texts: List[str],
    max_vocab_size: int = 20000,
    min_freq: int = 1,
) -> Dict[str, int]:
    from collections import Counter

    counter = Counter()
    for text in texts:
        tokens = text.split()
        counter.update(tokens)

    # reserve 0 for PAD, 1 for OOV
    word2idx = {"<PAD>": 0, "<OOV>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(word2idx) >= max_vocab_size:
            break
        word2idx[word] = len(word2idx)
    return word2idx


def texts_to_sequences(
    texts: List[str],
    word2idx: Dict[str, int],
    max_len: int = 70,
) -> List[List[int]]:
    seqs = []
    oov_idx = word2idx.get("<OOV>", 1)
    for text in texts:
        tokens = text.split()
        indices = [word2idx.get(tok, oov_idx) for tok in tokens][:max_len]
        # pad
        if len(indices) < max_len:
            indices = indices + [word2idx["<PAD>"]] * (max_len - len(indices))
        seqs.append(indices)
    return seqs

from typing import Dict

import torch

from src.data.preprocess import clean_text, texts_to_sequences
from src.models.rnn_models import LSTMClassifier, GRUClassifier
from src.utils import get_device


def load_rnn_for_inference(
    checkpoint: Dict,
    model_type: str = "lstm",
):
    model_state = checkpoint["model_state_dict"]
    word2idx = checkpoint["word2idx"]
    label_encoder = checkpoint["label_encoder"]
    vocab_size = len(word2idx)
    num_classes = len(label_encoder.classes_)

    if model_type == "lstm":
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=num_classes,
        )
    else:
        model = GRUClassifier(
            vocab_size=vocab_size,
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=num_classes,
        )

    model.load_state_dict(model_state)
    model.eval()
    device = get_device()
    model.to(device)
    return model, word2idx, label_encoder, device


def predict_text_rnn(
    text: str,
    model,
    word2idx,
    label_encoder,
    device,
    max_len: int = 70,
):
    cleaned = clean_text(text)
    seq = texts_to_sequences([cleaned], word2idx, max_len=max_len)[0]
    x = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = probs.argmax()
    label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])
    return label, confidence

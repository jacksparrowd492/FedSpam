"""
FedSpam-Shield — command-line spam prediction with the federated global model.

Usage:
    python predict.py --subject "You won!" --body "Claim your prize now"
    python predict.py --subject "..." --body "..." --velocity 12 --header-mismatch 1
    python predict.py            # interactive: prompts for subject and body

Expects (relative to this file):
    model/global_model.pt
    model/tfidf_vectorizer.joblib
    model/velocity_scaler.json

Feature pipeline is identical to app.py / server.py:
    [tfidf(subject + " " + body) | (log1p(velocity) - mean) / std | header_mismatch]
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "global_model.pt"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
SCALER_PATH = MODEL_DIR / "velocity_scaler.json"

THRESHOLD = 0.5


class SpamMLP(nn.Module):
    """Input -> Dense -> ReLU -> Dropout -> Dense(1); outputs logits. Must match server.py / clients."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_artifacts():
    missing = [p.name for p in (MODEL_PATH, TFIDF_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {MODEL_DIR}: {', '.join(missing)}")

    # Dims come from the saved weights (net.0.weight: [hidden, input])
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    hidden_dim, input_dim = state_dict["net.0.weight"].shape
    model = SpamMLP(input_dim, hidden_dim)
    model.load_state_dict(state_dict, strict=True)
    model.eval()  # disables dropout

    vectorizer = joblib.load(TFIDF_PATH)
    with open(SCALER_PATH) as f:
        scaler = json.load(f)
    mean, std = float(scaler["mean"]), float(scaler["std"]) or 1.0

    vocab_dim = len(vectorizer.vocabulary_)
    if input_dim != vocab_dim + 2:
        raise ValueError(f"Model expects {input_dim} features but TF-IDF + 2 gives {vocab_dim + 2}; "
                         "the vectorizer does not match the trained model")
    return model, vectorizer, mean, std


def build_features(subject, body, velocity, header_mismatch, vectorizer, mean, std):
    text = f"{subject} {body}".strip()
    tfidf = vectorizer.transform([text]).toarray().astype(np.float32)  # (1, V)
    vel_norm = (np.log1p(velocity) - mean) / std
    extra = np.array([[vel_norm, float(header_mismatch)]], dtype=np.float32)
    return np.hstack([tfidf, extra])  # (1, V + 2)


@torch.no_grad()
def spam_probability(model, features):
    logit = model(torch.from_numpy(features))
    return float(torch.sigmoid(logit)[0])  # model outputs logits


def predict(subject, body, velocity=1, header_mismatch=0, artifacts=None):
    """Return (label, spam_probability) where label is "Spam" or "Not Spam"."""
    model, vectorizer, mean, std = artifacts or load_artifacts()
    feats = build_features(subject, body, velocity, header_mismatch, vectorizer, mean, std)
    prob = spam_probability(model, feats)
    return ("Spam" if prob > THRESHOLD else "Not Spam"), prob


def main():
    ap = argparse.ArgumentParser(description="Classify an email as Spam or Not Spam.")
    ap.add_argument("--subject", help="email subject (prompted if neither subject nor body is given)")
    ap.add_argument("--body", help="email body")
    ap.add_argument("--velocity", type=float, default=1,
                    help="emails this sender sent in a short window (default: 1)")
    ap.add_argument("--header-mismatch", type=int, choices=[0, 1], default=0,
                    help="1 if From/Reply-To/Return-Path headers disagree (default: 0)")
    args = ap.parse_args()

    subject, body = args.subject, args.body
    if subject is None and body is None:
        subject = input("Subject: ")
        body = input("Body: ")
    subject, body = subject or "", body or ""
    if not subject.strip() and not body.strip():
        ap.error("enter a subject or body")

    label, prob = predict(subject, body, args.velocity, args.header_mismatch)
    print(f"Prediction:       {label}")
    print(f"Spam probability: {prob:.4f}")


if __name__ == "__main__":
    main()

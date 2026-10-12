import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from scipy import sparse

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TRAIN_FILE = ROOT / "processed_emails_v3_train.csv"
TEST_FILE = ROOT / "processed_emails_v3_test.csv"
GLOBAL_MODEL = MODEL_DIR / "global_model.pt"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
SCALER_FILE = MODEL_DIR / "velocity_scaler.json"

CENTRAL_MODEL = MODEL_DIR / "centralized_model.pt"
RESULT_FILE = MODEL_DIR / "centralized_vs_federated.json"

SEED = 42
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
THRESHOLD = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpamMLP(nn.Module):
    def __init__(self, input_dim=5002, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_frame(path):
    df = pd.read_csv(path)

    required = {
        "subject", "body", "velocity", "header_mismatch", "label"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["velocity"] = (
        pd.to_numeric(df["velocity"], errors="coerce")
        .fillna(0).clip(lower=0)
    )
    df["header_mismatch"] = (
        pd.to_numeric(df["header_mismatch"], errors="coerce")
        .fillna(0).gt(0).astype(int)
    )
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path.name}: label must be 0 or 1")

    return df


def make_features(df, vectorizer, scaler):
    text = (df["subject"] + " " + df["body"]).str.strip()
    tfidf = vectorizer.transform(text)

    mean = float(scaler["mean"])
    std = float(scaler["std"])
    if std <= 0:
        raise ValueError("Velocity scaler std must be positive")

    velocity = (
        np.log1p(df["velocity"].to_numpy(dtype=np.float64)) - mean
    ) / std

    numeric = sparse.csr_matrix(
        np.column_stack([
            velocity,
            df["header_mismatch"].to_numpy(dtype=np.float32),
        ]),
        dtype=np.float32,
    )

    features = sparse.hstack(
        [tfidf, numeric], format="csr", dtype=np.float32
    )

    if features.shape[1] != int(scaler["input_dim"]):
        raise ValueError(
            f"Feature dimension {features.shape[1]} does not match "
            f"scaler input_dim {scaler['input_dim']}"
        )

    return features


def load_weights(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return obj


def predict_probabilities(model, X):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for start in range(0, X.shape[0], BATCH_SIZE):
            xb = X[start:start + BATCH_SIZE].toarray()
            xb = torch.from_numpy(xb.astype(np.float32)).to(device)
            probs = torch.sigmoid(model(xb))
            all_probs.extend(probs.cpu().numpy().tolist())

    return np.asarray(all_probs)


def evaluate(name, model, X, y):
    probs = predict_probabilities(model, X)
    pred = (probs >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    result = {
        "model": name,
        "test_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": THRESHOLD,
    }

    print(f"\n=== {name} test results ===")
    for key in ("test_samples", "accuracy", "precision", "recall", "f1"):
        print(f"{key:15}: {result[key]}")
    print(f"TN={tn} | FP={fp} | FN={fn} | TP={tp}")

    print("\nClassification report:")
    print(classification_report(
        y, pred, target_names=["Not Spam", "Spam"], zero_division=0
    ))
    return result


def main():
    for path in (TRAIN_FILE, TEST_FILE, GLOBAL_MODEL,
                 VECTORIZER_FILE, SCALER_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Device:", device)
    print("Loading existing shared preprocessing artifacts...")
    vectorizer = joblib.load(VECTORIZER_FILE)
    scaler = json.loads(SCALER_FILE.read_text(encoding="utf-8"))

    if len(vectorizer.vocabulary_) != 5000:
        raise ValueError("Expected the existing 5000-feature vectorizer")

    print("Loading training and held-out test data...")
    train_df = load_frame(TRAIN_FILE)
    test_df = load_frame(TEST_FILE)

    print(f"Train rows: {len(train_df):,}; "
          f"spam ratio: {train_df['label'].mean():.2%}")
    print(f"Test rows: {len(test_df):,}; "
          f"spam ratio: {test_df['label'].mean():.2%}")

    print("Building features with the same vectorizer and scaler...")
    X_train = make_features(train_df, vectorizer, scaler)
    X_test = make_features(test_df, vectorizer, scaler)
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=int)

    if not np.isin(y_test, [0, 1]).all():
        raise ValueError("Test labels must be binary")

    # Evaluate the existing federated model first.
    fed_state = load_weights(GLOBAL_MODEL)
    federated_model = SpamMLP(
        input_dim=X_test.shape[1],
        hidden_dim=int(scaler.get("hidden_dim", 128)),
        dropout=0.3,
    ).to(device)
    federated_model.load_state_dict(fed_state)

    federated_result = evaluate(
        "Federated FedAvg", federated_model, X_test, y_test
    )

    # Train a separate centralized baseline on the training CSV.
    print("\nTraining centralized baseline...")
    centralized_model = SpamMLP(
        input_dim=X_train.shape[1], hidden_dim=128, dropout=0.3
    ).to(device)

    optimizer = torch.optim.Adam(
        centralized_model.parameters(), lr=LEARNING_RATE
    )
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        centralized_model.train()
        order = rng.permutation(len(y_train))
        total_loss = 0.0
        total_seen = 0

        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            xb = X_train[ids].toarray().astype(np.float32)
            xb = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(y_train[ids]).to(device)

            optimizer.zero_grad()
            loss = criterion(centralized_model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(ids)
            total_seen += len(ids)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"training loss={total_loss / total_seen:.4f}"
        )

    training_seconds = time.perf_counter() - start_time

    # Save only to a new filename; do not replace the federated model.
    torch.save(centralized_model.state_dict(), CENTRAL_MODEL)

    centralized_result = evaluate(
        "Centralized", centralized_model, X_test, y_test
    )
    centralized_result["training_seconds"] = round(training_seconds, 2)
    centralized_result["epochs"] = EPOCHS
    centralized_result["batch_size"] = BATCH_SIZE
    centralized_result["learning_rate"] = LEARNING_RATE

    results = {
        "dataset": {
            "train_file": TRAIN_FILE.name,
            "test_file": TEST_FILE.name,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        "preprocessing": "Existing shared TF-IDF vectorizer and velocity scaler",
        "federated": federated_result,
        "centralized": centralized_result,
        "notes": [
            "Both models were evaluated on the same held-out test CSV.",
            "Centralized training uses the full training CSV.",
            "The centralized optimizer and training procedure are a baseline, "
            "not necessarily identical to the clients' local optimization.",
            "This script does not modify global_model.pt.",
        ],
    }

    RESULT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Comparison summary ===")
    print(f"{'Metric':<15}{'Federated':>15}{'Centralized':>15}")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(
            f"{key:<15}"
            f"{federated_result[key]:>15.4f}"
            f"{centralized_result[key]:>15.4f}"
        )

    print(f"\nCentralized model saved: {CENTRAL_MODEL}")
    print(f"Comparison saved: {RESULT_FILE}")


if __name__ == "__main__":
    main()

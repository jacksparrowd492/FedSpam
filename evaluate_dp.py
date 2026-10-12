"""
FedSpam-Shield — compare the FedAvg baseline with the DP-SGD global models on the held-out test set.

Evaluates model/global_model.pt (baseline) and every model/global_model_<run>.pt produced by
`server.py --fresh --run-name <run>`, using exactly the same test CSV, features and metrics as
centralized_train.py. The privacy budget of each run is read from its metadata file, which the
server fills from the clients' reports.

Usage:
    python evaluate_dp.py                       # all runs found in model/
    python evaluate_dp.py dp_eps1 dp_eps3       # only these runs (plus the baseline)
    python evaluate_dp.py demo_dp --out model/dp_comparison_demo.json

Writes model/dp_comparison.json (or --out). Refuses to replace an existing results file unless
--overwrite is given. Does not modify any model.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib

from centralized_train import (
    MODEL_DIR,
    SCALER_FILE,
    TEST_FILE,
    VECTORIZER_FILE,
    SpamMLP,
    device,
    evaluate,
    load_frame,
    load_weights,
    make_features,
)

BASELINE = ("baseline (no DP)", MODEL_DIR / "global_model.pt", MODEL_DIR / "global_metadata.json")
RESULT_FILE = MODEL_DIR / "dp_comparison.json"


def find_runs(names):
    runs = [BASELINE]
    if names:
        paths = [MODEL_DIR / f"global_model_{n}.pt" for n in names]
    else:
        paths = sorted(MODEL_DIR.glob("global_model_*.pt"))
    for path in paths:
        run = path.stem[len("global_model_"):]
        runs.append((run, path, MODEL_DIR / f"global_metadata_{run}.json"))
    missing = [str(p) for _, p, _ in runs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Model file(s) not found: {', '.join(missing)}")
    return runs


def read_metadata(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def output_path(args, default):
    """Never silently replace a recorded results file: require --out or --overwrite."""
    out = Path(args.out) if args.out else default
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} already exists (it is part of the experiment record). "
                 "Pass --out <new file> to write elsewhere, or --overwrite to replace it.")
    return out


def main(argv):
    ap = argparse.ArgumentParser(description="Evaluate baseline and DP global models on the held-out test set.")
    ap.add_argument("runs", nargs="*", help="run names (default: every model/global_model_<run>.pt)")
    ap.add_argument("--out", help=f"results file (default {RESULT_FILE})")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing results file")
    args = ap.parse_args(argv)
    result_file = output_path(args, RESULT_FILE)
    runs = find_runs(args.runs)

    vectorizer = joblib.load(VECTORIZER_FILE)
    scaler = json.loads(SCALER_FILE.read_text(encoding="utf-8"))
    test_df = load_frame(TEST_FILE)
    X_test = make_features(test_df, vectorizer, scaler)
    y_test = test_df["label"].to_numpy(dtype=int)
    print(f"Held-out test set: {TEST_FILE.name}, {len(y_test):,} rows, spam ratio {y_test.mean():.2%}")

    results = []
    for name, model_path, meta_path in runs:
        state = load_weights(model_path)
        hidden_dim, input_dim = state["net.0.weight"].shape
        model = SpamMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.3).to(device)
        model.load_state_dict(state)

        meta = read_metadata(meta_path)
        fit = meta.get("client_fit_metrics", {})
        result = evaluate(name, model, X_test, y_test)
        result.update(
            model_file=model_path.name,
            rounds=meta.get("round"),
            dp_epsilon=fit.get("dp_epsilon"),
            dp_delta=fit.get("dp_delta"),
            dp_noise_multiplier=fit.get("dp_noise_multiplier"),
            dp_max_grad_norm=fit.get("dp_max_grad_norm"),
        )
        results.append(result)

    result_file.write_text(json.dumps({
        "test_file": TEST_FILE.name,
        "test_rows": int(len(y_test)),
        "notes": [
            "dp_epsilon is the largest epsilon spent by any client over all rounds (per-email DP-SGD).",
            "A missing dp_epsilon means the clients trained without differential privacy.",
        ],
        "results": results,
    }, indent=2), encoding="utf-8")

    print("\n=== Differential privacy vs. baseline (held-out test set) ===")
    print(f"{'Run':<22}{'epsilon':>9}{'sigma':>8}{'rounds':>8}{'accuracy':>10}{'precision':>11}{'recall':>9}{'F1':>8}")
    for r in results:
        eps = f"{r['dp_epsilon']:.2f}" if r["dp_epsilon"] is not None else "inf"
        sigma = f"{r['dp_noise_multiplier']:.3f}" if r["dp_noise_multiplier"] is not None else "-"
        print(f"{r['model']:<22}{eps:>9}{sigma:>8}{str(r['rounds'] or '-'):>8}"
              f"{r['accuracy']:>10.4f}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}")
    print(f"\nSaved: {result_file}")


if __name__ == "__main__":
    main(sys.argv[1:])

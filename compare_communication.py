"""
FedSpam-Shield — compare communication cost and training time across federated runs.

Reads the model/communication_<run>.json files that server.py writes during training (bytes
exchanged with each client, round durations, client training times) and the privacy budget from
model/global_metadata_<run>.json.

Usage:
    python compare_communication.py                         # every run found in model/
    python compare_communication.py comm_nodp comm_dp_eps3  # only these runs
    python compare_communication.py demo_dp --out model/communication_comparison_demo.json

Writes model/communication_comparison.json (or --out). Refuses to replace an existing results file
unless --overwrite is given. Never modifies models or other results.
"""

import argparse
import json
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "model"
RESULT_FILE = MODEL_DIR / "communication_comparison.json"
MB = 1e6


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def find_runs(names):
    if names:
        paths = [MODEL_DIR / f"communication_{n}.json" for n in names]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Communication log(s) not found: {', '.join(missing)}")
        return paths
    return sorted(p for p in MODEL_DIR.glob("communication_*.json")
                  if not p.name.startswith("communication_comparison"))


def summarize(path):
    comm = load_json(path)
    run = path.stem[len("communication_"):]
    fit = load_json(MODEL_DIR / f"global_metadata_{run}.json").get("client_fit_metrics", {})
    totals, rounds = comm["totals"], comm["rounds"]
    n = max(len(rounds), 1)
    clients = []
    for cid, c in sorted(comm["clients"].items(), key=lambda kv: kv[1].get("num_examples") or 0):
        clients.append({
            "client_id": cid,
            "num_examples": c["num_examples"],
            "mb_received_from_server": c["bytes_received_from_server"] / MB,
            "mb_sent_to_server": c["bytes_sent_to_server"] / MB,
            "mb_total": (c["bytes_received_from_server"] + c["bytes_sent_to_server"]) / MB,
            "messages_received": c["messages_received"],
            "messages_sent": c["messages_sent"],
            "total_train_time_s": c["train_time_s"],
            "avg_train_time_per_round_s": c["train_time_s"] / n,
        })
    return {
        "run": run,
        "dp_epsilon": fit.get("dp_epsilon"),
        "dp_noise_multiplier": fit.get("dp_noise_multiplier"),
        "rounds": totals["rounds_completed"],
        "mb_total_all_clients": totals["bytes_total"] / MB,
        "mb_per_round_all_clients": totals["bytes_total"] / MB / n,
        "training_time_s": totals["training_time_s"],
        "avg_round_s": totals["avg_round_s"],
        "min_round_s": totals["min_round_s"],
        "max_round_s": totals["max_round_s"],
        "non_training_time_s": totals["non_training_time_s"],
        "wall_clock_s": totals.get("wall_clock_s"),
        "client_failures": totals["client_failures"],
        "clients": clients,
    }


def output_path(args, default):
    """Never silently replace a recorded results file: require --out or --overwrite."""
    out = Path(args.out) if args.out else default
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} already exists (it is part of the experiment record). "
                 "Pass --out <new file> to write elsewhere, or --overwrite to replace it.")
    return out


def main(argv):
    ap = argparse.ArgumentParser(description="Compare communication cost and training time across runs.")
    ap.add_argument("runs", nargs="*", help="run names (default: every model/communication_<run>.json)")
    ap.add_argument("--out", help=f"results file (default {RESULT_FILE})")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing results file")
    args = ap.parse_args(argv)
    result_file = output_path(args, RESULT_FILE)
    runs = [summarize(p) for p in find_runs(args.runs)]
    if not runs:
        print(f"No communication logs in {MODEL_DIR}. Run server.py --run-name <name> first.")
        return

    print("=== Per-run cost (all clients together) ===")
    print(f"{'Run':<16}{'epsilon':>8}{'rounds':>7}{'total MB':>10}{'MB/round':>10}"
          f"{'train time s':>14}{'avg round s':>13}{'non-train s':>13}")
    for r in runs:
        eps = f"{r['dp_epsilon']:.2f}" if r["dp_epsilon"] is not None else "inf"
        print(f"{r['run']:<16}{eps:>8}{r['rounds']:>7}{r['mb_total_all_clients']:>10.2f}"
              f"{r['mb_per_round_all_clients']:>10.2f}{r['training_time_s']:>14.1f}{r['avg_round_s']:>13.2f}"
              f"{r['non_training_time_s']:>13.1f}")

    print("\n=== Per client ===")
    print(f"{'Run':<16}{'client':<26}{'samples':>8}{'MB down':>9}{'MB up':>8}{'MB total':>10}"
          f"{'msgs down/up':>14}{'train s/round':>15}")
    for r in runs:
        for c in r["clients"]:
            print(f"{r['run']:<16}{c['client_id'][:25]:<26}{c['num_examples'] or 0:>8}"
                  f"{c['mb_received_from_server']:>9.2f}{c['mb_sent_to_server']:>8.2f}{c['mb_total']:>10.2f}"
                  f"{c['messages_received']:>8}/{c['messages_sent']:<5}{c['avg_train_time_per_round_s']:>15.2f}")

    result_file.write_text(json.dumps({
        "notes": [
            "Bytes are serialized Flower protobuf messages measured on the server; gRPC/TCP framing adds <1%.",
            "MB = 10^6 bytes. Each round sends the global model twice per client (fit + evaluate) and "
            "receives one model update plus a small evaluation result.",
            "training_time_s runs from the first fit instruction to the last evaluation result; "
            "non_training_time_s is that minus the slowest client's local training time in each round.",
            "On localhost the network transfer itself is near-instant; run across laptops for real LAN timings.",
        ],
        "runs": runs,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {result_file}")


if __name__ == "__main__":
    main(sys.argv[1:])

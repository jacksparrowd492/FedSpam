"""
FedSpam-Shield — collect submission evidence into evidence/.

Reads the saved models, result JSONs and run logs (evidence/logs/) and writes:
    evidence/evidence.json      every number below, machine-readable
    evidence/SUMMARY.md         accuracy, privacy, communication and reproducibility tables
    evidence/checksums.sha256   SHA-256 of every model, result, code and log file

Read-only on model/ and the code; safe to rerun at any time.

Usage:
    python collect_evidence.py
"""

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
EVIDENCE = ROOT / "evidence"
LOGS = EVIDENCE / "logs"
CLIENT_DIRS = {"client1": ROOT.parent / "Client-1", "client2": ROOT.parent / "Client-2"}

ACCURACY_RUNS = ["nodp", "dp_eps8", "dp_eps3", "dp_eps1"]
COMM_RUNS = ["comm_nodp", "comm_dp_eps8", "comm_dp_eps3", "comm_dp_eps1"]
SERVER_CODE = ["server.py", "predict.py", "flask_app.py", "centralized_train.py", "evaluate_dp.py",
               "compare_communication.py", "collect_evidence.py", "requirements.txt"]
CLIENT_CODE = {"client1": ["client1.py", "fedspam_common.py", "requirements.txt"],
               "client2": ["client2.py", "fedspam_common.py", "requirements.txt"]}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_log(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def last_float(pattern, text):
    found = re.findall(pattern, text)
    return float(found[-1]) if found else None


# ──────────────────────────────────────────────────────────────
# Evidence sections
# ──────────────────────────────────────────────────────────────
def client_setup(log_text):
    """Data split and DP configuration as printed by client1.py / client2.py at start-up."""
    info = {}
    m = re.search(r"Loaded (\S+): (\d+) rows -> (\d+) usable \| spam=(\d+) legit=(\d+)", log_text)
    if m:
        info.update(data_file=m[1], rows=int(m[2]), usable_rows=int(m[3]), spam=int(m[4]), legit=int(m[5]))
    m = re.search(r"Split: train (\d+) \(spam (\d+)\) \| validation (\d+) \(spam (\d+)\)", log_text)
    if m:
        info.update(train=int(m[1]), train_spam=int(m[2]), validation=int(m[3]), validation_spam=int(m[4]))
    m = re.search(r"Vectorizer SHA-256 fingerprint: (\w+)", log_text)
    if m:
        info["vectorizer_fingerprint"] = m[1]
    m = re.search(r"Differential privacy: ON \(DP-SGD\) \| noise multiplier ([\d.]+) \| clip norm ([\d.]+) \| "
                  r"delta ([\deE.+-]+) \| sample rate ([\d.]+) \| planned steps (\d+) \((\d+) rounds x (\d+) epochs\)"
                  r"(?: \| target epsilon ([\d.]+))?", log_text)
    if m:
        info["dp"] = {
            "noise_multiplier": float(m[1]), "max_grad_norm": float(m[2]), "delta": float(m[3]),
            "sample_rate": float(m[4]), "planned_steps": int(m[5]), "rounds": int(m[6]),
            "local_epochs": int(m[7]), "target_epsilon": float(m[8]) if m[8] else None,
            "epsilon_spent": last_float(r"DP budget spent so far: epsilon ([\d.]+)", log_text),
        }
    elif "Differential privacy: OFF" in log_text:
        info["dp"] = None
    return info


def accuracy_section():
    comparison = load_json(MODEL_DIR / "dp_comparison.json")
    rows = []
    for r in comparison["results"]:
        rows.append({k: r.get(k) for k in ("model", "model_file", "dp_epsilon", "dp_delta", "dp_noise_multiplier",
                                           "dp_max_grad_norm", "rounds", "accuracy", "precision", "recall", "f1",
                                           "tn", "fp", "fn", "tp", "test_samples")})
    return {"test_file": comparison["test_file"], "test_rows": comparison["test_rows"], "results": rows}


def privacy_section():
    runs = {}
    for run in ACCURACY_RUNS:
        clients = {name: client_setup(read_log(LOGS / "accuracy_runs" / run / f"{name}.log"))
                   for name in ("client1", "client2")}
        runs[run] = clients
    return {
        "mechanism": "DP-SGD on each client: per-email gradient clipping (L2 norm C) + Gaussian noise "
                     "N(0, (sigma*C)^2) on the summed gradient, divided by the expected batch size; Adam optimizer",
        "privacy_unit": "one email (one row of a client's training split)",
        "guarantee": "(epsilon, delta)-DP for every model update a client sends, composed over all rounds",
        "sampling": "Poisson sampling: each email joins each step independently with probability q = batch/n",
        "accountant": "Renyi DP (RDP) accountant from Opacus 1.6.0, orders 1.1-512, converted to (epsilon, delta)",
        "calibration": "sigma chosen as the smallest value whose RDP epsilon after rounds x epochs x round(n/batch) "
                       "steps is at most the target epsilon",
        "randomness": "DP noise and Poisson sampling use unseeded OS entropy (a known seed would let the noise be "
                      "removed); dropout and initial weights stay seeded",
        "not_covered": [
            "validation loss/accuracy/precision/recall/F1 each client reports to the server every round "
            "(computed on the client's held-out validation split, which is never trained on)",
            "the client sample counts sent with each update",
        ],
        "runs": runs,
    }


def communication_section():
    comparison = load_json(MODEL_DIR / "communication_comparison.json")
    per_round = {}
    for run in COMM_RUNS:
        log = load_json(MODEL_DIR / f"communication_{run}.json")
        per_round[run] = [{k: r[k] for k in ("round", "bytes_down", "bytes_up", "duration_s", "fit_phase_s",
                                             "evaluate_phase_s", "max_client_train_time_s")}
                          for r in log["rounds"]]
    return {"notes": comparison["notes"], "runs": comparison["runs"], "per_round": per_round}


def model_file(name: str) -> Path:
    """A model in model/, or its copy moved to model/archive/ during clean-up."""
    path = MODEL_DIR / name
    if path.exists():
        return path
    archived = sorted((MODEL_DIR / "archive").rglob(name))
    if not archived:
        raise FileNotFoundError(f"{name} not found in {MODEL_DIR} or {MODEL_DIR / 'archive'}")
    return archived[0]


def reproducibility_section():
    def weights(name):
        return torch.load(model_file(name), map_location="cpu")

    def identical(a, b):
        wa, wb = weights(a), weights(b)
        return wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)

    common = [CLIENT_DIRS[c] / "fedspam_common.py" for c in ("client1", "client2")]
    return {
        "bit_identical_checks": {
            "global_model.pt == global_model_nodp.pt (baseline reproduced by a fresh 10-round run)":
                identical("global_model.pt", "global_model_nodp.pt"),
            "global_model_nodp.pt == global_model_comm_nodp.pt (communication logging does not change training)":
                identical("global_model_nodp.pt", "global_model_comm_nodp.pt"),
            "Client-1 and Client-2 fedspam_common.py are byte-identical":
                all(p.exists() for p in common) and sha256(common[0]) == sha256(common[1]),
        },
        "seeds": {
            "server initial weights": "torch.manual_seed(42), random init (--fresh)",
            "client train/validation split": "train_test_split(test_size=0.2, random_state=42, stratified)",
            "client training": "seed 42 + round number (shuffling, dropout)",
            "DP noise and Poisson sampling": "deliberately unseeded, so DP runs are not bit-reproducible",
        },
        "configuration": {
            "clients": 2, "rounds": 10, "local_epochs": 2, "batch_size": 64, "learning_rate": 1e-4,
            "optimizer": "Adam", "hidden_dim": 128, "dropout": 0.3, "aggregation": "FedAvg (Flower)",
            "dp_max_grad_norm": 1.0, "dp_delta": 1e-5, "dp_target_epsilons": [8, 3, 1],
            "execution": "server and both clients on one laptop over 127.0.0.1",
        },
        "test_split_overlap": "0 of the test emails appear in client1.csv, client2.csv or the training CSV "
                              "(exact subject+body match); vectorizer fitted on client1.csv only",
    }


def environment_section():
    import flwr
    import numpy
    import sklearn

    env = {
        "python": sys.version.split()[0],
        # platform.platform() reports Windows 11 as "Windows-10-10.0.<build>"; builds >= 22000 are Windows 11
        "os": (f"Windows 11 (build {platform.version().split('.')[-1]})"
               if sys.platform == "win32" and int(platform.version().split(".")[-1]) >= 22000
               else platform.platform()),
        "cpu": platform.processor(),
        "logical_cpus": __import__("os").cpu_count(),
        "packages": {"torch": torch.__version__, "flwr": flwr.__version__, "scikit-learn": sklearn.__version__,
                     "numpy": numpy.__version__},
    }
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            env["ram_gb"] = round(status.ullTotalPhys / 2**30, 1)
    req = CLIENT_DIRS["client1"] / "requirements.txt"
    if req.exists():
        pin = re.search(r"^opacus==(\S+)", req.read_text(encoding="utf-8"), re.M)
        env["packages"]["opacus (clients)"] = pin[1] if pin else None
    return env


def write_checksums():
    files = sorted(p for p in MODEL_DIR.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    files += [ROOT / f for f in SERVER_CODE if (ROOT / f).exists()]
    for client, names in CLIENT_CODE.items():
        files += [CLIENT_DIRS[client] / n for n in names if (CLIENT_DIRS[client] / n).exists()]
    files += sorted(p for p in EVIDENCE.rglob("*") if p.is_file() and p.parent != EVIDENCE)
    lines = [f"{sha256(p)}  {p.relative_to(ROOT.parent).as_posix()}" for p in files]
    # LF line endings: `sha256sum -c` treats a trailing \r as part of the file name
    (EVIDENCE / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)


# ──────────────────────────────────────────────────────────────
# Markdown summary
# ──────────────────────────────────────────────────────────────
def fmt_eps(v):
    return "∞ (no DP)" if v is None else f"{v:.2f}"


def summary_markdown(ev):
    acc, priv, comm, rep, env = (ev[k] for k in ("accuracy", "privacy", "communication", "reproducibility",
                                                 "environment"))
    out = [f"# FedSpam-Shield — experiment evidence",
           "",
           f"Generated {ev['generated_utc']} by `collect_evidence.py` from the files in `model/` and `evidence/logs/`.",
           "",
           "## 1. Accuracy (held-out test set)",
           "",
           f"Test file `{acc['test_file']}`, {acc['test_rows']:,} emails. None of them appears in the client or "
           "training CSVs, and the TF-IDF vectorizer and velocity scaler were fitted on client1.csv only. "
           "ε is the larger of the two clients' budgets; per-client σ is in section 2.",
           "",
           "| Model | ε | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in acc["results"]:
        out.append(f"| {r['model']} | {fmt_eps(r['dp_epsilon'])} | {r['accuracy']:.4f} | "
                   f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['tn']} | {r['fp']} | "
                   f"{r['fn']} | {r['tp']} |")

    out += ["", "## 2. Privacy", "",
            f"- **Mechanism:** {priv['mechanism']}",
            f"- **Privacy unit:** {priv['privacy_unit']}",
            f"- **Guarantee:** {priv['guarantee']}",
            f"- **Sampling:** {priv['sampling']}",
            f"- **Accounting:** {priv['accountant']}",
            f"- **Calibration:** {priv['calibration']}",
            f"- **Randomness:** {priv['randomness']}",
            "- **Not covered by ε:** " + "; ".join(priv["not_covered"]),
            "",
            "| Run | Client | Train emails | Target ε | ε spent | δ | σ | Clip C | q | Steps |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for run, clients in priv["runs"].items():
        for name, c in clients.items():
            dp = c.get("dp")
            if dp:
                out.append(f"| {run} | {name} | {c.get('train', '?')} | {dp['target_epsilon']} | "
                           f"{dp['epsilon_spent']:.3f} | {dp['delta']:g} | {dp['noise_multiplier']:.4f} | "
                           f"{dp['max_grad_norm']} | {dp['sample_rate']:.5f} | {dp['planned_steps']} |")
            else:
                out.append(f"| {run} | {name} | {c.get('train', '?')} | – | – | – | – | – | – | – |")

    out += ["", "## 3. Communication and training time", "",
            "| Run | ε | Rounds | Total MB | MB/round | Training time (s) | Avg round (s) | Non-training (s) |",
            "|---|---|---|---|---|---|---|---|"]
    for r in comm["runs"]:
        out.append(f"| {r['run']} | {fmt_eps(r['dp_epsilon'])} | {r['rounds']} | {r['mb_total_all_clients']:.2f} | "
                   f"{r['mb_per_round_all_clients']:.2f} | {r['training_time_s']:.1f} | {r['avg_round_s']:.2f} | "
                   f"{r['non_training_time_s']:.1f} |")
    out += ["", "| Run | Client samples | MB received | MB sent | Messages down/up | Training s/round |",
            "|---|---|---|---|---|---|"]
    for r in comm["runs"]:
        for c in r["clients"]:
            out.append(f"| {r['run']} | {c['num_examples']} | {c['mb_received_from_server']:.2f} | "
                       f"{c['mb_sent_to_server']:.2f} | {c['messages_received']}/{c['messages_sent']} | "
                       f"{c['avg_train_time_per_round_s']:.2f} |")
    out += [""] + [f"- {n}" for n in comm["notes"]]
    out += ["- Per-round values are in `evidence.json` → `communication.per_round` and in "
            "`model/communication_<run>.json`."]

    out += ["", "## 4. Reproducibility", ""]
    for check, ok in rep["bit_identical_checks"].items():
        out.append(f"- {'PASS' if ok else 'FAIL'}: {check}")
    out += [f"- Test/train overlap: {rep['test_split_overlap']}", "", "**Seeds**", ""]
    out += [f"- {k}: {v}" for k, v in rep["seeds"].items()]
    out += ["", "**Configuration**", ""]
    out += [f"- {k}: {v}" for k, v in rep["configuration"].items()]
    out += ["", "**Environment**", ""]
    out += [f"- {k}: {v}" for k, v in env.items() if k != "packages"]
    out += [f"- {k}: {v}" for k, v in env["packages"].items()]
    out += ["", f"File integrity: `checksums.sha256` lists {ev['checksum_count']} files "
                "(verify with `sha256sum -c` from the CAT-2 folder).",
            "", "**Run logs** (`evidence/logs/`)", "",
            "- `accuracy_runs/`: the four runs scored in section 1 (server + both clients).",
            "- `communication_runs/`: the four runs measured in section 3.",
            "- `failed_attempts/`: a DP ε=1 communication run in which client 2 ran out of memory "
            "(machine-wide commit limit); it was discarded and rerun.",
            "- `../scripts/`: the sweep scripts used to launch the runs and the DP-SGD gradient check."]
    return "\n".join(out) + "\n"


def main():
    if not LOGS.is_dir():
        sys.exit(f"{LOGS} not found: copy the run logs there first")
    evidence = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "accuracy": accuracy_section(),
        "privacy": privacy_section(),
        "communication": communication_section(),
        "reproducibility": reproducibility_section(),
        "environment": environment_section(),
    }
    evidence["checksum_count"] = write_checksums()
    (EVIDENCE / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (EVIDENCE / "SUMMARY.md").write_text(summary_markdown(evidence), encoding="utf-8")

    failed = [k for k, ok in evidence["reproducibility"]["bit_identical_checks"].items() if not ok]
    print(f"Wrote {EVIDENCE / 'SUMMARY.md'}, evidence.json and checksums.sha256 "
          f"({evidence['checksum_count']} files)")
    print("Reproducibility checks: " + ("all PASS" if not failed else "FAIL: " + "; ".join(failed)))


if __name__ == "__main__":
    main()

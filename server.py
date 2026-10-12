"""Flower federated learning server for the spam classifier.

Runs FedAvg with exactly 2 clients for 10 rounds, then exits. After every
round the aggregated weights are converted back into a PyTorch state_dict and
saved to model/global_model.pt (plus model/global_metadata.json).

Clients must send/receive parameters as a list of numpy arrays in the order of
SpamMLP.state_dict(), i.e. [net.0.weight, net.0.bias, net.3.weight, net.3.bias].

Usage:
    python server_fl.py [--address 0.0.0.0:8080] [--rounds 10]
    python server_fl.py --fresh --run-name dp_eps3     # DP experiment (clients run with --dp-epsilon 3)

--run-name NAME saves to model/global_model_NAME.pt + model/global_metadata_NAME.json instead, so
experiments never overwrite the baseline model used by predict.py / flask_app.py; it refuses to reuse
an existing run name unless --overwrite is given. --fresh starts from random weights; DP runs need it,
because warm-starting from a model trained without DP on the same client data would void the clients'
privacy guarantee.

Communication cost (bytes per client, round durations, total training time) is written after every
round to model/communication.json, or model/communication_NAME.json with --run-name.
"""
import argparse
import json
import logging
import os
import time
from collections import OrderedDict

import flwr as fl
import joblib
import numpy as np
import torch
import torch.nn as nn
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, serde

MODEL_DIR = "model"
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "global_model.pt")
GLOBAL_META_PATH = os.path.join(MODEL_DIR, "global_metadata.json")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
NUM_CLIENTS = 2

# Own handler (not the root logger) so Flower's log lines are not printed twice
log = logging.getLogger("fl-server")
log.setLevel(logging.INFO)
log.propagate = False
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_handler)


class SpamMLP(nn.Module):
    """Input -> Dense -> ReLU -> Dropout -> Dense(1); outputs logits. Must match the clients."""

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


def weighted_average(metrics):
    """Average client metrics weighted by each client's number of examples.

    Privacy parameters (dp_*) are not averaged: the largest client value is reported. For dp_epsilon
    that is the worst case; for dp_noise_multiplier it is the most private client, not the least.
    """
    total = sum(n for n, _ in metrics)
    if not total:
        return {}
    keys = {k for _, m in metrics for k, v in m.items() if isinstance(v, (int, float))}
    return {k: (max(m[k] for _, m in metrics if k in m) if k.startswith("dp_")
                else sum(n * m.get(k, 0.0) for n, m in metrics) / total) for k in keys}


def output_paths(run_name):
    """(model, metadata, communication log) paths for a run."""
    if not run_name:
        return GLOBAL_MODEL_PATH, GLOBAL_META_PATH, os.path.join(MODEL_DIR, "communication.json")
    return (os.path.join(MODEL_DIR, f"global_model_{run_name}.pt"),
            os.path.join(MODEL_DIR, f"global_metadata_{run_name}.json"),
            os.path.join(MODEL_DIR, f"communication_{run_name}.json"))


_PROTO = {"fit_ins": serde.fit_ins_to_proto, "fit_res": serde.fit_res_to_proto,
          "evaluate_ins": serde.evaluate_ins_to_proto, "evaluate_res": serde.evaluate_res_to_proto}


def message_bytes(kind, msg):
    """Serialized protobuf size of a Flower message (weights + config/metrics), as sent over gRPC."""
    try:
        return _PROTO[kind](msg).ByteSize()
    except Exception:
        params = getattr(msg, "parameters", None)
        return sum(len(t) for t in params.tensors) if params is not None else 0


class CommunicationLog:
    """Bytes exchanged with each client and per-round timings, measured on the server.

    Byte counts are the serialized Flower messages; gRPC/HTTP2/TCP framing adds well under 1%.
    Round time = server sending fit instructions -> receiving the last evaluation result, so it includes
    local training; the clients' own reported train_time_s lets that be separated out.
    """

    def __init__(self, path, run_name, num_rounds):
        self.path = path
        self.server_start = time.time()
        self.data = {"run": run_name or "default", "num_rounds": num_rounds, "rounds": [], "clients": {}, "totals": {}}
        self.current = None

    def _client(self, proxy):
        return self.data["clients"].setdefault(proxy.cid, {
            "num_examples": None, "bytes_received_from_server": 0, "bytes_sent_to_server": 0,
            "messages_received": 0, "messages_sent": 0, "train_time_s": 0.0})

    def _add(self, proxy, kind, msg, direction):
        n = message_bytes(kind, msg)
        c = self._client(proxy)
        rnd = self.current["clients"].setdefault(proxy.cid, {"bytes_down": 0, "bytes_up": 0})
        if direction == "down":
            c["bytes_received_from_server"] += n
            c["messages_received"] += 1
            rnd["bytes_down"] += n
            self.current["bytes_down"] += n
        else:
            c["bytes_sent_to_server"] += n
            c["messages_sent"] += 1
            rnd["bytes_up"] += n
            self.current["bytes_up"] += n

    def fit_sent(self, server_round, instructions):
        self.current = {"round": server_round, "start": time.time(), "bytes_down": 0, "bytes_up": 0,
                        "clients": {}, "failures": 0}
        for proxy, ins in instructions:
            self._add(proxy, "fit_ins", ins, "down")

    def fit_received(self, results, failures):
        if self.current is None:  # no fit instructions were sent this round
            return
        self.current["fit_end"] = time.time()
        self.current["failures"] += len(failures)
        for proxy, res in results:
            self._add(proxy, "fit_res", res, "up")
            c = self._client(proxy)
            c["num_examples"] = res.num_examples
            train_s = float((res.metrics or {}).get("train_time_s", 0.0))
            c["train_time_s"] += train_s
            self.current["clients"][proxy.cid]["train_time_s"] = train_s

    def evaluate_sent(self, instructions):
        if self.current is None:  # no fit instructions were sent this round
            return
        self.current["evaluate_start"] = time.time()
        for proxy, ins in instructions:
            self._add(proxy, "evaluate_ins", ins, "down")

    def evaluate_received(self, results, failures):
        if self.current is None:  # no fit instructions were sent this round
            return
        end = time.time()
        r = self.current
        r["failures"] += len(failures)
        for proxy, res in results:
            self._add(proxy, "evaluate_res", res, "up")
        fit_end = r.get("fit_end", end)
        r.update(duration_s=end - r["start"], fit_phase_s=fit_end - r["start"],
                 evaluate_phase_s=end - r.get("evaluate_start", fit_end),
                 max_client_train_time_s=max((c.get("train_time_s", 0.0) for c in r["clients"].values()), default=0.0))
        r["start_offset_s"] = r.pop("start") - self.server_start
        r.pop("fit_end", None)
        r.pop("evaluate_start", None)
        self.data["rounds"].append(r)
        self._update_totals()
        self.write()
        log.info(f"Round {r['round']}: communication {r['bytes_down'] / 1e6:.2f} MB down + "
                 f"{r['bytes_up'] / 1e6:.2f} MB up (all clients) | round {r['duration_s']:.1f}s "
                 f"(fit {r['fit_phase_s']:.1f}s, of which client training <= {r['max_client_train_time_s']:.1f}s; "
                 f"evaluate {r['evaluate_phase_s']:.1f}s)")
        self.current = None

    def _update_totals(self):
        rounds = self.data["rounds"]
        durations = [r["duration_s"] for r in rounds]
        self.data["totals"] = {
            "rounds_completed": len(rounds),
            "bytes_down": sum(r["bytes_down"] for r in rounds),
            "bytes_up": sum(r["bytes_up"] for r in rounds),
            "bytes_total": sum(r["bytes_down"] + r["bytes_up"] for r in rounds),
            "training_time_s": sum(durations),  # first fit sent -> last evaluation received
            "avg_round_s": sum(durations) / len(durations),
            "min_round_s": min(durations),
            "max_round_s": max(durations),
            # Round time not spent in the slowest client's local training: transfer, (de)serialization,
            # aggregation, evaluation and waiting
            "non_training_time_s": sum(r["duration_s"] - r["max_client_train_time_s"] for r in rounds),
            "client_failures": sum(r["failures"] for r in rounds),
        }

    def finish(self):
        self.data["totals"]["wall_clock_s"] = time.time() - self.server_start  # includes waiting for clients
        self.write()

    def write(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that uses exactly NUM_CLIENTS per round, logs progress and saves the global model."""

    def __init__(self, model: SpamMLP, num_rounds: int, model_path=GLOBAL_MODEL_PATH, meta_path=GLOBAL_META_PATH,
                 comm_log: CommunicationLog = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.num_rounds = num_rounds
        self.model_path, self.meta_path = model_path, meta_path
        self.comm = comm_log

    # Always sample exactly NUM_CLIENTS (waits until that many are connected)
    def num_fit_clients(self, num_available_clients):
        return NUM_CLIENTS, NUM_CLIENTS

    def num_evaluation_clients(self, num_available_clients):
        return NUM_CLIENTS, NUM_CLIENTS

    def configure_fit(self, server_round, parameters, client_manager):
        connected = client_manager.num_available()
        log.info("=" * 60)
        log.info(f"Round {server_round}/{self.num_rounds} | clients connected: {connected}/{NUM_CLIENTS}")
        if connected < NUM_CLIENTS:
            log.info(f"Waiting for {NUM_CLIENTS - connected} more client(s) to connect...")
        instructions = super().configure_fit(server_round, parameters, client_manager)
        if self.comm:
            self.comm.fit_sent(server_round, instructions)
        return instructions

    def configure_evaluate(self, server_round, parameters, client_manager):
        instructions = super().configure_evaluate(server_round, parameters, client_manager)
        if self.comm:
            self.comm.evaluate_sent(instructions)
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        if self.comm:
            self.comm.fit_received(results, failures)
        log.info(f"Round {server_round}: received {len(results)} result(s), {len(failures)} failure(s)")
        for failure in failures:
            log.warning(f"Round {server_round}: client failure: {failure!r}")
        if not results:
            log.error(f"Round {server_round}: no client results, keeping previous global model")
            return None, {}

        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is None:
            log.error(f"Round {server_round}: aggregation failed, keeping previous global model")
            return None, {}

        sample_counts = [fit_res.num_examples for _, fit_res in results]
        for i, (_, fit_res) in enumerate(results, 1):
            m = fit_res.metrics or {}
            if "dp_epsilon" in m:
                log.info(f"Round {server_round}: client {i} ({fit_res.num_examples} samples) trained with DP-SGD | "
                         f"epsilon so far {m['dp_epsilon']:.3f} | delta {m.get('dp_delta', float('nan')):g} | "
                         f"noise multiplier {m.get('dp_noise_multiplier', float('nan')):.4f}")
            else:
                log.info(f"Round {server_round}: client {i} ({fit_res.num_examples} samples) trained WITHOUT DP")
        try:
            self.save_global_model(parameters, server_round, sample_counts, metrics)
        except Exception:
            # Saving must never take the server down; training continues next round
            log.exception(f"Round {server_round}: failed to save global model")
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if self.comm:
            try:
                self.comm.evaluate_received(results, failures)
            except Exception:
                log.exception(f"Round {server_round}: failed to record communication metrics")
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if loss is not None:
            shown = " | ".join(f"{k} {v:.4f}" for k, v in sorted(metrics.items()))
            log.info(f"Round {server_round}: eval loss {loss:.4f}" + (f" | {shown}" if shown else ""))
        return loss, metrics

    def save_global_model(self, parameters, server_round, sample_counts, fit_metrics=None):
        """Convert Flower parameters into a PyTorch state_dict and save it atomically."""
        ndarrays = parameters_to_ndarrays(parameters)
        keys = list(self.model.state_dict().keys())
        if len(ndarrays) != len(keys):
            raise ValueError(f"Expected {len(keys)} tensors, got {len(ndarrays)}")
        state_dict = OrderedDict((k, torch.tensor(np.asarray(v))) for k, v in zip(keys, ndarrays))
        self.model.load_state_dict(state_dict, strict=True)  # validates shapes

        tmp_path = self.model_path + ".tmp"
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, self.model_path)

        layer1 = self.model.net[0]
        with open(self.meta_path, "w") as f:
            json.dump({
                "aggregation": "FedAvg (Flower)",
                "round": server_round,
                "num_rounds": self.num_rounds,
                "client_sample_counts": sample_counts,
                "input_dim": layer1.in_features,
                "hidden_dim": layer1.out_features,
                "dropout": self.model.net[2].p,
                "model_outputs": "logits (apply sigmoid for probabilities)",
                # dp_* values are the largest across clients (see weighted_average); absent without DP
                "client_fit_metrics": dict(fit_metrics or {}),
            }, f, indent=2)
        log.info(f"Round {server_round}: global model saved to {self.model_path}")


def build_initial_model(input_dim, hidden_dim, dropout, model_path=GLOBAL_MODEL_PATH, fresh=False):
    """Fresh model, warm-started from an existing global model if its shape matches (unless fresh)."""
    if input_dim is None:
        input_dim = len(joblib.load(VECTORIZER_PATH).vocabulary_) + 2  # TF-IDF + velocity + header_mismatch
    torch.manual_seed(42)
    model = SpamMLP(input_dim, hidden_dim, dropout)

    if not fresh and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            log.info(f"Initial weights: loaded existing {model_path}")
            return model
        except Exception as e:
            log.warning(f"Could not load {model_path} ({e}); using random initial weights")
    log.info("Initial weights: random (seed 42)")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", default="0.0.0.0:8080")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--input-dim", type=int, default=None, help="default: vectorizer vocabulary size + 2")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--round-timeout", type=float, default=600.0,
                    help="seconds to wait for client results per round before treating them as failures")
    ap.add_argument("--fresh", action="store_true",
                    help="start from random weights instead of warm-starting (required for DP runs)")
    ap.add_argument("--run-name", default=None,
                    help="save to model/global_model_<NAME>.pt (+ metadata) instead of model/global_model.pt")
    ap.add_argument("--overwrite", action="store_true", help="allow --run-name to replace an existing run")
    args = ap.parse_args()
    if args.run_name and not all(c.isalnum() or c in "-_." for c in args.run_name):
        ap.error("--run-name may only contain letters, digits, '-', '_' and '.'")
    model_path, meta_path, comm_path = output_paths(args.run_name)
    if args.run_name and not args.overwrite:
        existing = [p for p in (model_path, meta_path, comm_path) if os.path.exists(p)]
        if existing:
            ap.error(f"run '{args.run_name}' already exists ({', '.join(existing)}); "
                     "choose a new --run-name or pass --overwrite")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model = build_initial_model(args.input_dim, args.hidden, args.dropout, model_path, args.fresh)
    initial = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    log.info(f"Model: input_dim={model.net[0].in_features}, hidden_dim={args.hidden}, "
             f"{sum(a.size for a in initial):,} parameters")

    comm = CommunicationLog(comm_path, args.run_name, args.rounds)
    strategy = SaveModelFedAvg(
        model=model,
        num_rounds=args.rounds,
        model_path=model_path,
        meta_path=meta_path,
        comm_log=comm,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        accept_failures=True,
        initial_parameters=ndarrays_to_parameters(initial),
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    log.info(f"Starting Flower server on {args.address} | {args.rounds} rounds | "
             f"waiting for {NUM_CLIENTS} clients")
    try:
        fl.server.start_server(
            server_address=args.address,
            config=fl.server.ServerConfig(num_rounds=args.rounds, round_timeout=args.round_timeout),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        log.warning("Interrupted by user; last saved global model is in model/")
        return
    finally:
        if comm.data["rounds"]:
            comm.finish()
    t = comm.data["totals"]
    if t:
        log.info(f"Communication: {t['bytes_total'] / 1e6:.2f} MB total over {t['rounds_completed']} rounds "
                 f"({t['bytes_down'] / 1e6:.2f} MB down, {t['bytes_up'] / 1e6:.2f} MB up) | training time "
                 f"{t['training_time_s']:.1f}s, avg round {t['avg_round_s']:.1f}s | saved to {comm_path}")
    log.info(f"Federated training finished. Final global model: {model_path}")


if __name__ == "__main__":
    main()

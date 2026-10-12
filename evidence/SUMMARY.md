# FedSpam-Shield — experiment evidence

Generated 2026-10-12 16:50 UTC by `collect_evidence.py` from the files in `model/` and `evidence/logs/`.

## 1. Accuracy (held-out test set)

Test file `processed_emails_v3_test.csv`, 16,570 emails. None of them appears in the client or training CSVs, and the TF-IDF vectorizer and velocity scaler were fitted on client1.csv only. ε is the larger of the two clients' budgets; per-client σ is in section 2.

| Model | ε | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| baseline (no DP) | ∞ (no DP) | 0.9940 | 0.9934 | 0.9950 | 0.9942 | 7986 | 56 | 43 | 8485 |
| nodp | ∞ (no DP) | 0.9940 | 0.9934 | 0.9950 | 0.9942 | 7986 | 56 | 43 | 8485 |
| dp_eps8 | 8.00 | 0.9124 | 0.9129 | 0.9173 | 0.9151 | 7296 | 746 | 705 | 7823 |
| dp_eps3 | 3.00 | 0.9111 | 0.9127 | 0.9148 | 0.9137 | 7296 | 746 | 727 | 7801 |
| dp_eps1 | 1.00 | 0.9094 | 0.9127 | 0.9110 | 0.9119 | 7299 | 743 | 759 | 7769 |

## 2. Privacy

- **Mechanism:** DP-SGD on each client: per-email gradient clipping (L2 norm C) + Gaussian noise N(0, (sigma*C)^2) on the summed gradient, divided by the expected batch size; Adam optimizer
- **Privacy unit:** one email (one row of a client's training split)
- **Guarantee:** (epsilon, delta)-DP for every model update a client sends, composed over all rounds
- **Sampling:** Poisson sampling: each email joins each step independently with probability q = batch/n
- **Accounting:** Renyi DP (RDP) accountant from Opacus 1.6.0, orders 1.1-512, converted to (epsilon, delta)
- **Calibration:** sigma chosen as the smallest value whose RDP epsilon after rounds x epochs x round(n/batch) steps is at most the target epsilon
- **Randomness:** DP noise and Poisson sampling use unseeded OS entropy (a known seed would let the noise be removed); dropout and initial weights stay seeded
- **Not covered by ε:** validation loss/accuracy/precision/recall/F1 each client reports to the server every round (computed on the client's held-out validation split, which is never trained on); the client sample counts sent with each update

| Run | Client | Train emails | Target ε | ε spent | δ | σ | Clip C | q | Steps |
|---|---|---|---|---|---|---|---|---|---|
| nodp | client1 | 25960 | – | – | – | – | – | – | – |
| nodp | client2 | 27061 | – | – | – | – | – | – | – |
| dp_eps8 | client1 | 25960 | 8.0 | 7.997 | 1e-05 | 0.5482 | 1.0 | 0.00247 | 8120 |
| dp_eps8 | client2 | 27061 | 8.0 | 7.998 | 1e-05 | 0.5455 | 1.0 | 0.00237 | 8460 |
| dp_eps3 | client1 | 25960 | 3.0 | 2.995 | 1e-05 | 0.7300 | 1.0 | 0.00247 | 8120 |
| dp_eps3 | client2 | 27061 | 3.0 | 2.997 | 1e-05 | 0.7251 | 1.0 | 0.00237 | 8460 |
| dp_eps1 | client1 | 25960 | 1.0 | 0.990 | 1e-05 | 1.1621 | 1.0 | 0.00247 | 8120 |
| dp_eps1 | client2 | 27061 | 1.0 | 0.997 | 1e-05 | 1.1475 | 1.0 | 0.00237 | 8460 |

## 3. Communication and training time

| Run | ε | Rounds | Total MB | MB/round | Training time (s) | Avg round (s) | Non-training (s) |
|---|---|---|---|---|---|---|---|
| comm_nodp | ∞ (no DP) | 10 | 153.76 | 15.38 | 67.4 | 6.74 | 3.5 |
| comm_dp_eps8 | 8.00 | 10 | 153.76 | 15.38 | 140.5 | 14.05 | 3.5 |
| comm_dp_eps3 | 3.00 | 10 | 153.76 | 15.38 | 136.9 | 13.69 | 3.3 |
| comm_dp_eps1 | 1.00 | 10 | 153.76 | 15.38 | 139.7 | 13.97 | 3.3 |

| Run | Client samples | MB received | MB sent | Messages down/up | Training s/round |
|---|---|---|---|---|---|
| comm_nodp | 25960 | 51.25 | 25.63 | 20/20 | 6.26 |
| comm_nodp | 27061 | 51.25 | 25.63 | 20/20 | 6.38 |
| comm_dp_eps8 | 25960 | 51.25 | 25.63 | 20/20 | 13.32 |
| comm_dp_eps8 | 27061 | 51.25 | 25.63 | 20/20 | 13.70 |
| comm_dp_eps3 | 25960 | 51.25 | 25.63 | 20/20 | 12.97 |
| comm_dp_eps3 | 27061 | 51.25 | 25.63 | 20/20 | 13.35 |
| comm_dp_eps1 | 25960 | 51.25 | 25.63 | 20/20 | 13.20 |
| comm_dp_eps1 | 27061 | 51.25 | 25.63 | 20/20 | 13.63 |

- Bytes are serialized Flower protobuf messages measured on the server; gRPC/TCP framing adds <1%.
- MB = 10^6 bytes. Each round sends the global model twice per client (fit + evaluate) and receives one model update plus a small evaluation result.
- training_time_s runs from the first fit instruction to the last evaluation result; non_training_time_s is that minus the slowest client's local training time in each round.
- On localhost the network transfer itself is near-instant; run across laptops for real LAN timings.
- Per-round values are in `evidence.json` → `communication.per_round` and in `model/communication_<run>.json`.

## 4. Reproducibility

- PASS: global_model.pt == global_model_nodp.pt (baseline reproduced by a fresh 10-round run)
- PASS: global_model_nodp.pt == global_model_comm_nodp.pt (communication logging does not change training)
- PASS: Client-1 and Client-2 fedspam_common.py are byte-identical
- Test/train overlap: 0 of the test emails appear in client1.csv, client2.csv or the training CSV (exact subject+body match); vectorizer fitted on client1.csv only

**Seeds**

- server initial weights: torch.manual_seed(42), random init (--fresh)
- client train/validation split: train_test_split(test_size=0.2, random_state=42, stratified)
- client training: seed 42 + round number (shuffling, dropout)
- DP noise and Poisson sampling: deliberately unseeded, so DP runs are not bit-reproducible

**Configuration**

- clients: 2
- rounds: 10
- local_epochs: 2
- batch_size: 64
- learning_rate: 0.0001
- optimizer: Adam
- hidden_dim: 128
- dropout: 0.3
- aggregation: FedAvg (Flower)
- dp_max_grad_norm: 1.0
- dp_delta: 1e-05
- dp_target_epsilons: [8, 3, 1]
- execution: server and both clients on one laptop over 127.0.0.1

**Environment**

- python: 3.10.0
- os: Windows 11 (build 26200)
- cpu: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD
- logical_cpus: 16
- ram_gb: 15.7
- torch: 2.14.1+cpu
- flwr: 1.30.0
- scikit-learn: 1.7.2
- numpy: 2.2.6
- opacus (clients): 1.6.0

File integrity: `checksums.sha256` lists 79 files (verify with `sha256sum -c` from the CAT-2 folder).

**Run logs** (`evidence/logs/`)

- `accuracy_runs/`: the four runs scored in section 1 (server + both clients).
- `communication_runs/`: the four runs measured in section 3.
- `failed_attempts/`: a DP ε=1 communication run in which client 2 ran out of memory (machine-wide commit limit); it was discarded and rerun.
- `../scripts/`: the sweep scripts used to launch the runs and the DP-SGD gradient check.

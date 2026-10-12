# FedSpam-Shield — demo runbook

The demo trains a new DP model under a new run name, so the recorded results in `model/` and `evidence/` are not touched. A 3-round run takes about 1–2 minutes once the clients have loaded their data.

## Before the demo

- Copy the current `client1.py`/`client2.py`, `fedspam_common.py` and `requirements.txt` to Laptops 1 and 2, then run `pip install -r requirements.txt` on each.
- On Laptop 3, run `ipconfig` and note the IPv4 address. Allow inbound TCP 8080 in Windows Firewall.
- If everything runs on one laptop: close memory-heavy apps first (for example `wsl --shutdown`), and start client 2 only after client 1 prints `Connecting to Flower server`.

## 1. Start the Flower server (Laptop 3, `Spam Mail` folder)

```powershell
.\.venv\Scripts\Activate.ps1
python server.py --fresh --run-name demo_dp_eps3 --rounds 3
```

`--fresh` starts from random weights, which the DP guarantee needs. `--run-name` writes `model/global_model_demo_dp_eps3.pt` and its metadata and communication log. It refuses to reuse an existing name.

## 2. Start the two clients with DP-SGD

Laptop 1 (`Client-1` folder):

```powershell
python client1.py --server-address <LAPTOP_3_IP>:8080 --rounds 3 --dp-epsilon 3
```

Laptop 2 (`Client-2` folder):

```powershell
python client2.py --server-address <LAPTOP_3_IP>:8080 --rounds 3 --dp-epsilon 3
```

What to point out while it runs:

- **Clients:** the calibrated noise multiplier, the ε spent after each round, and that only weights leave the laptop.
- **Server:** each client's ε per round, the MB sent and received per round, and the global model being saved after every round.

## 3. Show the results

```powershell
python evaluate_dp.py demo_dp_eps3 --out model\dp_comparison_demo.json
python compare_communication.py demo_dp_eps3 --out model\communication_comparison_demo.json
```

`--out` is required because the default result files belong to the experiment record. To show the full recorded results, open `evidence\SUMMARY.md`.

## 4. Show the spam checker

```powershell
python flask_app.py
```

Open http://127.0.0.1:5000. The UI uses `model/global_model.pt`, the no-DP FedAvg baseline.

## 5. Verify the record is intact (optional)

From the `CAT-2` folder in Git Bash:

```bash
sha256sum -c --quiet "Spam Mail/evidence/checksums.sha256" && echo "all files verified"
```

A demo run adds new files but changes none of the listed ones, so this still passes after the demo.

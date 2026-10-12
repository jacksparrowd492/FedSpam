#!/usr/bin/env bash
# Runs fresh 10-round federations on localhost: no DP, then DP at epsilon 8, 3, 1.
B="/f/Files/DPSA LAB/CAT-2"; S="$1"; PORT=8099
run() {  # $1 = run name, rest = extra client args
  name=$1; shift
  echo "=== $name start $(date +%T)"
  (cd "$B/Spam Mail" && .venv/Scripts/python.exe server.py --address 127.0.0.1:$PORT --rounds 10 --fresh --run-name "$name" > "$S/sweep_${name}_server.log" 2>&1) &
  srv=$!
  sleep 8
  (cd "$B/Client-1" && .venv/Scripts/python.exe client1.py --server-address 127.0.0.1:$PORT --rounds 10 "$@" > "$S/sweep_${name}_c1.log" 2>&1) &
  (cd "$B/Client-2" && .venv/Scripts/python.exe client2.py --server-address 127.0.0.1:$PORT --rounds 10 "$@" > "$S/sweep_${name}_c2.log" 2>&1)
  wait
  grep -h "Run finished\|eval loss" "$S/sweep_${name}_server.log" | tail -2
  echo "=== $name end $(date +%T)"
}
run nodp
run dp_eps8 --dp-epsilon 8
run dp_eps3 --dp-epsilon 3
run dp_eps1 --dp-epsilon 1
echo SWEEP DONE

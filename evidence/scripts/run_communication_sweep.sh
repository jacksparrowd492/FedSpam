#!/usr/bin/env bash
# Communication-measurement runs, one at a time. Clients start staggered (client2 waits until client1
# has loaded its data) to keep peak RAM down; any client failure stops that run instead of hanging.
B="/f/Files/DPSA LAB/CAT-2"; S="$1"; shift; PORT=8099

cleanup() {
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '127\.0\.0\.1:$PORT' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" >/dev/null 2>&1
}

wait_for() {  # $1 = log file, $2 = pattern, $3 = (unused), $4 = timeout s; fails early on a logged error
  local t=0
  until grep -q "$2" "$1" 2>/dev/null; do
    # kill -0 cannot see native Windows processes, so detect a crash from the log instead
    grep -qF -e "Traceback" -e "| ERROR |" "$1" 2>/dev/null && return 1
    sleep 2; t=$((t + 2))
    [ "$t" -ge "$4" ] && return 1
  done
  return 0  # otherwise the function returns the loop body's last (false) test
}

run() {  # $1 = run name, rest = extra client args
  local name=$1; shift
  local slog="$S/comm_${name}_server.log" c1log="$S/comm_${name}_c1.log" c2log="$S/comm_${name}_c2.log"
  echo "=== $name start $(date +%T)"
  cd "$B/Spam Mail" && .venv/Scripts/python.exe server.py --address 127.0.0.1:$PORT --rounds 10 --fresh --run-name "$name" > "$slog" 2>&1 &
  local srv=$!
  sleep 8
  cd "$B/Client-1" && .venv/Scripts/python.exe client1.py --server-address 127.0.0.1:$PORT --rounds 10 "$@" > "$c1log" 2>&1 &
  local c1=$!
  if ! wait_for "$c1log" "Connecting to Flower server" "$c1" 300; then
    echo "=== $name FAILED: client1 did not finish loading"; tail -5 "$c1log"; cleanup; wait; return 1
  fi
  cd "$B/Client-2" && .venv/Scripts/python.exe client2.py --server-address 127.0.0.1:$PORT --rounds 10 "$@" > "$c2log" 2>&1 &
  local c2=$!
  wait "$c2"; local rc2=$?
  if [ "$rc2" -ne 0 ]; then
    echo "=== $name FAILED: client2 exit code $rc2"; tail -5 "$c2log"; cleanup; wait; return 1
  fi
  wait "$c1"; local rc1=$?
  wait "$srv"; local rcs=$?
  grep -h "Communication:" "$slog" | sed 's/.*INFO | //'
  echo "=== $name end $(date +%T) (exit codes server $rcs, client1 $rc1, client2 $rc2)"
}

for spec in "$@"; do  # each spec: name[:epsilon]
  name=${spec%%:*}; eps=${spec#*:}
  if [ "$eps" = "$spec" ]; then run "$name"; else run "$name" --dp-epsilon "$eps"; fi || { echo "SWEEP ABORTED"; exit 1; }
done
echo "SWEEP DONE"

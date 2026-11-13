#!/usr/bin/env bash
# Load test for --enable-kda-hicache-node-aggregate on the K3.1 P. Run on CPU002.
set -uo pipefail

P_POD=${P_POD:-job-ceba47c9efb8-20261113222136-master-0}
P_URL=${P_URL:-http://10.51.1.1:37878}
ROUTER=${ROUTER:-http://10.51.6.44:38848}
MODEL=${MODEL:-k31-lvning-kda-on}
MASTER=${MASTER:-http://mooncake-master-online.mooncake.svc.cluster.local:9003}
N=${N:-300}
CONC=${CONC:-16}
DATASET=${DATASET:-/root/fjm/dataset_demo_1500.jsonl}
BENCH=${BENCH:-/root/lvning/builds/kda-k31-v1-src/benchmark.py}
OUT=${OUT:-/root/lvning/kda-k31-test/load-$(date +%m%d%H%M)}
NS=qiji
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
log() { echo "[$(date '+%F %T')] $*"; }
die() { log "ABORT: $*"; exit 1; }
restarts() {
  kubectl get pod -n "$NS" "$P_POD" \
    -o jsonpath='{.status.containerStatuses[?(@.name=="master")].restartCount}'
}
metrics() {
  curl -s -m 30 "$P_URL/metrics" > "$OUT/metrics$1.txt"
  log "metrics #$1: $(grep -c '^sglang:' "$OUT/metrics$1.txt") sglang lines"
}
scan() {
  log "scan #$1"
  kubectl exec -i -n "$NS" "$P_POD" -c master -- \
    python3 - "$MASTER" "$PREFIX" - < "$HERE/l3_key_scan.py" > "$OUT/scan$1.txt"
  local rc=$?
  sed '/__JSON_BEGIN__/,/__JSON_END__/d' "$OUT/scan$1.txt"
  sed -n '/__JSON_BEGIN__/,/__JSON_END__/p' "$OUT/scan$1.txt" | sed '1d;$d' > "$OUT/scan$1.json"
  [ -s "$OUT/scan$1.json" ] || log "scan #$1 produced no JSON (rc=$rc)"
  return $rc
}
bench() {
  log "pass $1: N=$N CONC=$CONC"
  python3 "$BENCH" --mode sequential --concurrency "$CONC" --num-requests "$N" \
    --dataset "$DATASET" --api-url "$ROUTER/v1/chat/completions" --model-override "$MODEL" \
    --request-timeout 1800 --output "$OUT/pass$1.md" --errors-file "$OUT/pass$1_errors.jsonl"
  log "pass $1 benchmark exit=$?"
}

log "OUT=$OUT P_POD=$P_POD START=$START"
PREFIX=$(kubectl logs -n "$NS" "$P_POD" -c master | grep -m1 'Using Mooncake config prefix' \
  | sed 's/.*Using Mooncake config prefix: *//' | awk '{print $1}')
[ -n "$PREFIX" ] || die "PREFIX not found in P log"
log "PREFIX=$PREFIX"
R0=$(restarts); log "restartCount=$R0"

scan 0 || die "scan #0 failed"
metrics 0

bench 1
log "sleep 120 for write-through drain"; sleep 120
metrics 1
scan 1

ok=
for i in 1 2 3 4 5; do
  body=$(curl -s -m 60 -X POST "$P_URL/flush_cache")
  log "flush_cache try $i: ${body:0:80}"
  case "$body" in "Cache flushed"*) ok=1; break ;; esac
  sleep 5
done
[ -n "$ok" ] || die "flush_cache never succeeded"

bench 2
log "sleep 60"; sleep 60
metrics 2
scan 2

kubectl logs -n "$NS" "$P_POD" -c master --since-time="$START" > "$OUT/p_log.txt"
log "P log since start: $(wc -l < "$OUT/p_log.txt") lines"
log "Traceback|Error lines (excl. jsonschema_rs): $(grep -E 'Traceback|Error' "$OUT/p_log.txt" | grep -vc jsonschema_rs)"
log "Traceback lines (all): $(grep -c Traceback "$OUT/p_log.txt")"
R1=$(restarts); log "restartCount=$R1 (start $R0)"

python3 - "$OUT" "$R0" "$R1" <<'PY'
import collections, json, os, re, sys

out, r0, r1 = sys.argv[1], sys.argv[2], sys.argv[3]
AGG_SIZE = 232316928
LBL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
NAMES = {"objects": "sglang:hicache_storage_objects_total",
         "pages": "sglang:hicache_storage_pages_total",
         "bytes": "sglang:hicache_storage_bytes_total"}
FAILED = "sglang:hicache_storage_read_failed_keys_total"
CACHED = "sglang:cached_tokens_total"
res = []

def verdict(level, msg):
    res.append(level)
    print("%-4s %s" % (level, msg))

def load_metrics(i):
    m = collections.Counter()
    p = os.path.join(out, "metrics%d.txt" % i)
    if not os.path.exists(p):
        return m
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        head, _, val = line.rstrip().rpartition(" ")
        name, _, labels = head.partition("{")
        lab = dict(LBL.findall(labels))
        try:
            v = float(val)
        except ValueError:
            continue
        for short, full in NAMES.items():
            if name == full:
                m[(short, lab.get("pool", "?"), lab.get("op", "?"))] += v
        if name == FAILED:
            m[("failed", lab.get("pool", "?"), lab.get("code", "?"))] += v
        if name == CACHED:
            m[("cached", lab.get("cache_source", "all"), "")] += v
    return m

def load_scan(i):
    p = os.path.join(out, "scan%d.json" % i)
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None

def bench_errors(i):
    p = os.path.join(out, "pass%d_errors.jsonl" % i)
    return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0

M = [load_metrics(i) for i in range(3)]
S = [load_scan(i) for i in range(3)]
D = [None] + [collections.Counter({k: M[i][k] - M[i - 1].get(k, 0) for k in M[i]}) for i in (1, 2)]

def g(d, kind, pool, op):
    return int(d.get((kind, pool, op), 0))

for i in (1, 2):
    d = D[i]
    print("\n== pass %d ==" % i)
    print("benchmark errors:", bench_errors(i))
    for pool in ("mamba", "kv", "indexer"):
        for op in ("exists", "get", "put"):
            print("  %-7s %-6s objects=%-8d pages=%-8d bytes=%d" % (
                pool, op, g(d, "objects", pool, op), g(d, "pages", pool, op), g(d, "bytes", pool, op)))
    for (kind, a, b), v in sorted(d.items()):
        if kind == "failed" and v:
            print("  read_failed pool=%s code=%s +%d" % (a, b, v))
        if kind == "cached":
            print("  cached_tokens source=%s +%d" % (a, v))
    if S[i] and S[i - 1]:
        new_agg = len(set(S[i]["agg_hashes"]) - set(S[i - 1]["agg_hashes"]))
        new_pr = len(set(S[i]["per_rank_kda_hashes"]) - set(S[i - 1]["per_rank_kda_hashes"]))
        dmla = S[i]["counts"].get("mla", 0) - S[i - 1]["counts"].get("mla", 0)
        print("  keys: new agg hashes=%d new per_rank_kda hashes=%d mla key count delta=%d"
              % (new_agg, new_pr, dmla))

print("\n== verdicts ==")
d1, d2 = D[1], D[2]
put_o, put_p = g(d1, "objects", "mamba", "put"), g(d1, "pages", "mamba", "put")
verdict("PASS" if put_o == put_p > 0 else "FAIL",
        "pass1 mamba put objects=%d pages=%d (equal, >0)" % (put_o, put_p))

if S[0] and S[1]:
    new_agg = len(set(S[1]["agg_hashes"]) - set(S[0]["agg_hashes"]))
    ok = put_o > 0 and 0.9 * put_o <= new_agg <= put_o
    verdict("PASS" if ok else "WARN", "pass1 new agg keys=%d vs mamba put objects=%d%s" % (
        new_agg, put_o, "" if ok else " (shared-pool eviction can drop keys)"))
else:
    verdict("WARN", "scan0/scan1 missing, agg key delta not checked")

for i in (1, 2):
    if S[i] and S[i - 1]:
        new_pr = len(set(S[i]["per_rank_kda_hashes"]) - set(S[i - 1]["per_rank_kda_hashes"]))
        verdict("PASS" if new_pr == 0 else "FAIL", "pass%d new per_rank_kda hashes=%d (expect 0)" % (i, new_pr))
    else:
        verdict("WARN", "pass%d per_rank_kda not checked (scan missing)" % i)

sizes = sorted({s for sc in S[1:] if sc for s in sc["sizes"]})
if not sizes:
    verdict("WARN", "no agg key sampled with a memory replica")
else:
    verdict("PASS" if sizes == [AGG_SIZE] else "FAIL", "agg sizes=%s (expect [%d])" % (sizes, AGG_SIZE))

get_o, get_p = g(d2, "objects", "mamba", "get"), g(d2, "pages", "mamba", "get")
verdict("PASS" if get_o == get_p > 0 else "FAIL",
        "pass2 mamba get objects=%d pages=%d (equal, >0)" % (get_o, get_p))
failed2 = int(sum(v for k, v in d2.items() if k[0] == "failed"))
verdict("PASS" if failed2 == 0 else "FAIL", "pass2 read_failed delta=%d" % failed2)
kv_get = g(d2, "pages", "kv", "get")
verdict("PASS" if kv_get > 0 else "FAIL", "pass2 kv get pages=%d (>0)" % kv_get)

for i in (1, 2):
    e = bench_errors(i)
    verdict("PASS" if e == 0 else "WARN", "pass%d benchmark errors=%d" % (i, e))

lines = open(os.path.join(out, "p_log.txt"), errors="replace").read().splitlines()
bad = 0
for n, l in enumerate(lines):
    if "Traceback" in l and not any("jsonschema_rs" in x for x in lines[n:n + 40]):
        bad += 1
verdict("PASS" if bad == 0 else "FAIL", "new Tracebacks (excl. jsonschema_rs)=%d" % bad)
verdict("PASS" if r0 == r1 else "FAIL", "restartCount %s -> %s" % (r0, r1))

overall = "FAIL" if "FAIL" in res else "WARN" if "WARN" in res else "PASS"
print("\nOVERALL: %s" % overall)
PY

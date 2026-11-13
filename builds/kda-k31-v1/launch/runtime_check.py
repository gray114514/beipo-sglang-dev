"""K3.1 runtime check for --enable-kda-hicache-node-aggregate on P.

Flow: R1 cold -> R2 (L1 hit) -> wait for write-through -> flush P L1/L2 -> R3 (L3 restore).
Usage: python3 runtime_check.py --router http://IP:PORT --p http://IP:PORT [--wait 90 --lines 1600]
"""
import argparse, json, re, sys, time, urllib.error, urllib.request

METRIC_RE = re.compile(r"^sglang:hicache_storage_(objects|pages|bytes|read_failed_keys|hit_limited)_total(?:\{(.*)\})?\s+(\S+)")
LABEL_RE = re.compile(r'(\w+)="([^"]*)"')
results = []


def http(url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def record(name, status, detail):
    results.append((name, status, detail))
    print(f"[{status}] {name}: {detail}")


def die(msg):
    print(f"ERROR: {msg}")
    sys.exit(2)


def build_prompt(n_lines):
    body = "\n".join(f"Record {i:05d}: value {(i * 7919) % 10007}." for i in range(n_lines))
    return body + "\n\nWhat are the values of Record 00007 and of the last record? Answer briefly."


def ask(a, salt, text, tag):
    t = time.time()
    try:
        code, raw = http(f"{a.router}/v1/chat/completions", {
            "model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "cache_salt": salt,
            "messages": [{"role": "user", "content": text}]}, timeout=a.req_timeout)
    except Exception as e:
        die(f"{tag} request failed: {e!r}")
    if code != 200:
        die(f"{tag} HTTP {code}: {raw[:300]}")
    resp = json.loads(raw)
    m = resp["choices"][0]["message"]
    r = {"text": (m.get("reasoning_content") or "") + "\n<content>\n" + (m.get("content") or ""),
         "usage": resp.get("usage") or {}, "secs": round(time.time() - t, 1)}
    print(f"{tag}: {r['secs']}s usage={json.dumps(r['usage'])} content={(m.get('content') or '')[:120]!r}")
    return r


def metrics(a):
    try:
        code, raw = http(f"{a.p}/metrics", timeout=30)
    except Exception as e:
        die(f"GET /metrics failed: {e!r}")
    if code != 200:
        die(f"GET /metrics HTTP {code}")
    out = {}
    for line in raw.splitlines():
        mt = METRIC_RE.match(line)
        if not mt:
            continue
        labels = dict(LABEL_RE.findall(mt.group(2) or ""))
        key = (mt.group(1), labels.get("pool", ""), labels.get("op") or labels.get("code", ""))
        out[key] = out.get(key, 0.0) + float(mt.group(3))
    return out


def delta(before, after):
    return {k: after[k] - before.get(k, 0.0) for k in after if after[k] - before.get(k, 0.0)}


def show(title, d):
    print(f"--- {title} ---")
    for (name, pool, op), v in sorted(d.items()):
        print(f"  {name:18s} pool={pool:8s} {op:8s} +{v:g}")
    if not d:
        print("  (no change)")


def flush(a):
    for i in range(a.flush_retries):
        try:
            code, raw = http(f"{a.p}/flush_cache", {}, timeout=60)
        except Exception as e:
            code, raw = -1, repr(e)
        print(f"flush_cache attempt {i + 1}: HTTP {code} {raw.strip()[:100]!r}")
        if code == 200 and "not" not in raw.lower():
            return True
        time.sleep(3)
    return False


ap = argparse.ArgumentParser()
ap.add_argument("--router", required=True, help="router base URL")
ap.add_argument("--p", required=True, help="prefill server base URL (TP0, metrics enabled)")
ap.add_argument("--model", default="k31-lvning-kda-on")
ap.add_argument("--lines", type=int, default=1600, help="prompt lines (~16 tokens each)")
ap.add_argument("--max-tokens", type=int, default=200)
ap.add_argument("--wait", type=int, default=90, help="seconds to wait for write-through to L3")
ap.add_argument("--req-timeout", type=int, default=600)
ap.add_argument("--flush-retries", type=int, default=5)
a = ap.parse_args()
a.router, a.p = a.router.rstrip("/"), a.p.rstrip("/")

try:
    code, _ = http(f"{a.p}/health", timeout=30)
except Exception as e:
    die(f"GET {a.p}/health failed: {e!r}")
record("P health", "PASS" if code == 200 else "FAIL", f"HTTP {code}")
code, raw = http(f"{a.p}/get_server_info", timeout=30)
flag = json.loads(raw).get("enable_kda_hicache_node_aggregate") if code == 200 else None
if flag is None:
    record("node_aggregate flag", "WARN", f"key missing in get_server_info (HTTP {code})")
else:
    record("node_aggregate flag", "PASS" if flag else "FAIL", f"enable_kda_hicache_node_aggregate={flag}")

salt = f"kda-k31-check-{int(time.time() * 1000)}"
text = build_prompt(a.lines)
print(f"cache_salt={salt} prompt_lines={a.lines}")
m0 = metrics(a)
r1 = ask(a, salt, text, "R1 cold")
r2 = ask(a, salt, text, "R2 L1  ")
print(f"waiting {a.wait}s for write-through to L3 ...")
time.sleep(a.wait)
m1 = metrics(a)
if not flush(a):
    die("flush_cache did not succeed; R3 would not exercise L3")
time.sleep(2)
r3 = ask(a, salt, text, "R3 L3  ")
m2 = metrics(a)

if not any(k[0] == "objects" for k in m2):
    record("metrics present", "FAIL", "no sglang:hicache_storage_* counters on P /metrics (metrics not exported?)")
w, r = delta(m0, m1), delta(m1, m2)
show("L3 delta: write phase (R1, R2, wait)", w)
show("L3 delta: R3 restore", r)

put_o, put_p = w.get(("objects", "mamba", "put"), 0), w.get(("pages", "mamba", "put"), 0)
record("mamba put aggregated", "PASS" if put_o > 0 and put_o == put_p else "FAIL",
       f"objects={put_o:g} pages={put_p:g} (expect >0 and equal)")
get_o, get_p = r.get(("objects", "mamba", "get"), 0), r.get(("pages", "mamba", "get"), 0)
get_b = r.get(("bytes", "mamba", "get"), 0)
record("mamba get aggregated", "PASS" if get_o > 0 and get_o == get_p and get_b > 0 else "FAIL",
       f"objects={get_o:g} pages={get_p:g} bytes={get_b:g} (expect >0, objects==pages)")
fails = {k: v for k, v in delta(m0, m2).items() if k[0] == "read_failed_keys"}
record("no L3 read failures", "FAIL" if fails else "PASS", str(fails) if fails else "none")
prompt_tokens = r1["usage"].get("prompt_tokens") or 0
kv_put = w.get(("pages", "kv", "put"), 0)
kv_get = r.get(("pages", "kv", "get"), 0)
ok = kv_put > 0 and kv_get >= 0.8 * kv_put
record("kv L3 hit on R3", "PASS" if ok else "WARN",
       f"R3 kv get pages={kv_get:g} vs write-phase kv put pages={kv_put:g} (prompt_tokens={prompt_tokens})")
record("R3 == R2 (full text)", "PASS" if r3["text"] == r2["text"] else "FAIL", "L3 restore vs L1 hit")
print(f"[INFO] R1 == R3 (full text): {r1['text'] == r3['text']} (cold vs prefix hit may differ)")

print("\n===== SUMMARY =====")
for name, status, detail in results:
    print(f"{status:4s}  {name}: {detail}")
failed = any(s == "FAIL" for _, s, _ in results)
print("OVERALL:", "FAIL" if failed else "PASS")
sys.exit(1 if failed else 0)

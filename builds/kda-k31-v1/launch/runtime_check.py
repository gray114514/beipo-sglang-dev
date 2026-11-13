"""K3.1 KDA node aggregation runtime check: cold -> L1 -> flush -> L3 restore.

Usage: python3 runtime_check.py --router http://IP:38848 --p http://IP:37878
"""
import argparse, json, re, time, urllib.request

def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()

def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode()

def prompt(n_lines):
    body = "\n".join(f"Record {i:05d}: value {(i * 7919) % 10007}." for i in range(n_lines))
    return body + "\n\nWhat are the values of Record 00007 and of the last record? Answer briefly."

def ask(a, salt, text):
    t = time.time()
    resp = json.loads(post(f"{a.router}/v1/chat/completions", {
        "model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "cache_salt": salt,
        "messages": [{"role": "user", "content": text}]}))
    m = resp["choices"][0]["message"]
    return {"text": (m.get("reasoning_content") or "") + "\n<content>\n" + (m.get("content") or ""),
            "content": m.get("content"), "usage": resp.get("usage"), "secs": round(time.time() - t, 1)}

def l3_metrics(a):
    out = {}
    for line in get(f"{a.p}/metrics").splitlines():
        if line.startswith("#"):
            continue
        if re.match(r"sglang:hicache_storage_(objects|pages|bytes|read_failed_keys|hit_limited)_total", line):
            name = line.split("{")[0].replace("sglang:hicache_storage_", "")
            labels = dict(re.findall(r'(pool|op|code)="([^"]*)"', line))
            key = f"{name}[{labels.get('pool')},{labels.get('op') or labels.get('code', '')}]"
            out[key] = out.get(key, 0) + float(line.rsplit(" ", 1)[1])
    return out

def delta(before, after):
    return {k: after[k] - before.get(k, 0) for k in sorted(after) if after[k] - before.get(k, 0)}

ap = argparse.ArgumentParser()
ap.add_argument("--router", required=True); ap.add_argument("--p", required=True)
ap.add_argument("--model", default="k31-lvning-kda-on"); ap.add_argument("--lines", type=int, default=1600)
ap.add_argument("--max-tokens", type=int, default=200); ap.add_argument("--wait", type=int, default=90)
a = ap.parse_args()
salt = f"kda-k31-{int(time.time())}"
text = prompt(a.lines)
print("salt", salt)
m0 = l3_metrics(a)
r1 = ask(a, salt, text); print("R1 cold ", r1["secs"], "s", r1["usage"], repr(r1["content"])[:120])
r2 = ask(a, salt, text); print("R2 L1   ", r2["secs"], "s", r2["usage"], repr(r2["content"])[:120])
print(f"wait {a.wait}s for write-through to L3"); time.sleep(a.wait)
m1 = l3_metrics(a); print("L3 io after R1/R2 (delta):", json.dumps(delta(m0, m1)))
print("flush P L1/L2:", post(f"{a.p}/flush_cache", {}, timeout=60).strip()[:80]); time.sleep(5)
r3 = ask(a, salt, text); print("R3 L3   ", r3["secs"], "s", r3["usage"], repr(r3["content"])[:120])
m2 = l3_metrics(a); print("L3 io during R3 (delta):", json.dumps(delta(m1, m2)))
print("R2 == R3 (full text):", r2["text"] == r3["text"]); print("R1 == R3 (full text):", r1["text"] == r3["text"])

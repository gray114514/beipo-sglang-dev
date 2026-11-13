"""Scan Mooncake master keys for one prefix. Runs inside the P pod via stdin:
kubectl exec -i ... -- python3 - MASTER_URL PREFIX - [--max-keys N] < l3_key_scan.py
"""
import argparse
import collections
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

AGG_SIZE = 232316928
PER_RANK_RE = re.compile(r"^\d+_(temporal|conv_\d+)$")


def http_get(url, timeout=60):
    return urllib.request.urlopen(url, timeout=timeout)


def key_count(base):
    with http_get(base + "/metrics") as r:
        for line in r.read().decode().splitlines():
            if line.startswith("master_key_count"):
                return int(float(line.split()[-1]))
    return None


def classify(suffix):
    if suffix.startswith("kdanode"):
        return "agg"
    if PER_RANK_RE.match(suffix):
        return "per_rank_kda"
    if "indexer" in suffix:
        return "indexer"
    if suffix.endswith("_k") or suffix == "k":
        return "mla"
    return "other"


def find_size(obj):
    if isinstance(obj, dict):
        if "size_" in obj:
            return obj["size_"]
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            s = find_size(v)
            if s is not None:
                return s
    return None


def query_size(base, key):
    url = base + "/query_key?key=" + urllib.parse.quote(key, safe="")
    try:
        with http_get(url, timeout=30) as r:
            body = r.read().decode()
    except urllib.error.HTTPError:
        return None
    try:
        data = json.loads(body).get("data") or []
    except (ValueError, AttributeError):
        return None
    sizes = {find_size(rep) for rep in data} - {None}
    return sorted(sizes) if sizes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master")
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--max-keys", type=int, default=3000000)
    a = ap.parse_args()
    base = a.master.rstrip("/")
    head = a.prefix + "_"

    n = key_count(base)
    print("master_key_count:", n)
    if n is not None and n > a.max_keys:
        print("STOP: master_key_count %d > --max-keys %d, not listing" % (n, a.max_keys))
        sys.exit(3)

    counts = collections.Counter()
    hashes = collections.defaultdict(set)
    other_samples = []
    recent_agg = collections.deque(maxlen=20)
    total = 0
    with http_get(base + "/get_all_keys", timeout=600) as r:
        for raw in r:
            key = raw.decode(errors="replace").strip()
            if not key:
                continue
            total += 1
            if not key.startswith(head):
                continue
            h, _, suffix = key[len(head):].partition("_")
            cls = classify(suffix)
            counts[cls] += 1
            hashes[cls].add(h)
            if cls == "agg":
                recent_agg.append(key)
            elif cls == "other" and len(other_samples) < 5:
                other_samples.append(key)

    print("listed keys: %d, with prefix %r: %d" % (total, a.prefix, sum(counts.values())))
    for cls in ("agg", "per_rank_kda", "mla", "indexer", "other"):
        print("  %-13s keys=%-9d hashes=%d" % (cls, counts[cls], len(hashes[cls])))
    for k in other_samples:
        print("  other sample:", k)

    agg = hashes["agg"]
    agg_with_mla = len(agg & hashes["mla"])
    print("agg hashes: %d, with mla key of same hash: %d (expect all)" % (len(agg), agg_with_mla))
    print("per_rank_kda keys: %d (expect 0)" % counts["per_rank_kda"])

    sizes, no_mem = [], 0
    for key in recent_agg:
        s = query_size(base, key)
        if s is None:
            no_mem += 1
        else:
            sizes.extend(s)
    distinct = sorted(set(sizes))
    print("agg size sample: queried=%d no_memory_replica=%d distinct_sizes=%s (expect [%d])"
          % (len(recent_agg), no_mem, distinct, AGG_SIZE))

    summary = {
        "prefix": a.prefix,
        "master_key_count": n,
        "listed_total": total,
        "counts": dict(counts),
        "hash_counts": {k: len(v) for k, v in hashes.items()},
        "agg_with_mla": agg_with_mla,
        "size_queried": len(recent_agg),
        "size_no_memory": no_mem,
        "sizes": distinct,
        "agg_hashes": sorted(agg),
        "per_rank_kda_hashes": sorted(hashes["per_rank_kda"]),
    }
    text = json.dumps(summary)
    if a.out != "-":
        with open(a.out, "w") as f:
            f.write(text)
    print("__JSON_BEGIN__")
    print(text)
    print("__JSON_END__")


if __name__ == "__main__":
    main()

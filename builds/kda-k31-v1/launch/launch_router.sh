#!/bin/bash
# Baseline router (user, 2026-11-13); changes: names only.
set -e; source "$(dirname "$0")/common.sh"; cd "$NS_DIR"
./northserve launch \
  --app-name "$APP" --model-name "$APP" --served-model-name "$APP" \
  --model-path /data/Kimi-K3 --backend sglang --namespace qiji --queue qiji \
  --minilb-replicas 1 --priority-class-name higher-priority-job \
  --image harbor.local.clusters/bp/lmsysorg/sglang:v0.5.21-infrawaves-dev-router-bodylimit-responses-prefill4xx \
  --use-host-network --component router --server-port 37878 --discovery-port 37878 \
  --router-engine sgl-router --server-discover \
  --extra-cmds "--tokenizer-path /data/Kimi-K3/tiktoken.model --policy cache_aware --cache-prefix-provider radix_tree --worker-queue-limit 4 --cache-affinity-min-matched-tokens 1024 --cache-switch-margin-tokens 4096 --pressure-abs-threshold-tokens 1024 --pressure-rel-threshold 1.25 --min-load-choices 3 --log-level info --log-format json --disable-input-ids-forwarding" \
  --standalone -y

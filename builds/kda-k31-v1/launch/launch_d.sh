#!/bin/bash
# Baseline K3.1 decode (user, 2026-11-13); changes: names, --dcp-size 8 (P DCP8 -> D DCP1 is rejected by requires_dcp_relayout). D keeps the unmodified image.
set -e; source "$(dirname "$0")/common.sh"; cd "$NS_DIR"
./northserve launch \
  --app-name "$APP" --model-name "$APP" --served-model-name "$APP" \
  --model-path /data/Kimi-K3.1-Pro --backend sglang --namespace qiji --queue qiji --replicas 1 \
  --server-port 37878 --discovery-port 37878 --server-discover \
  --pd-config-file configs/sglang_openai_pd.yaml --decode-nodes 1 --gpus-per-pod 8 \
  --image "$D_IMAGE" --priority-class-name higher-priority-job --gpu-type b300 --use-host-network \
  --component decode --drop_cache true --autotune-copy false --server_cpu_mem 600 \
  --hicachel3 true --mooncake_master online --L3cpumem 900 --ssd-size 8 \
  --extra-envs "SGLANG_TIMEOUT_KEEP_ALIVE=100 SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1 GLOO_SOCKET_IFNAME=bond0 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 UCX_TLS=rc,cuda_copy,cuda_ipc,self,sm GLOG_minloglevel=1 MC_LOG_LEVEL=WARNING SGLANG_K3_VIDEO_MAX_NUM_FRAMES=32 SGLANG_K3_VIDEO_MAX_INFLIGHT_FRAMES=512 SGLANG_K3_VIDEO_DEBUG_LOG=true OMP_NUM_THREADS=4 SGLANG_K3_VIDEO_DECODE_THREADS=4 SGLANG_K3_VIDEO_MAX_QUEUE_DEPTH=8 SGLANG_K3_IMAGE_PREPROCESS_MODE=auto SGLANG_K3_IMAGE_PREPROCESS_GPU_BUDGET_MB=16384 SGLANG_K3_IMAGE_PREPROCESS_OOM_FALLBACK=true SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=1920 " \
  --extra-cmds "--enable-cache-report --trust-remote-code --tp-size 8 --dcp-size 8 --ep-size 8 --moe-a2a-backend megamoe  --mem-fraction-static 0.92 --max-running-requests 64 --max-mamba-cache-size 150 --disaggregation-decode-extra-slots 16 --weight-loader-prefetch-checkpoints --model-loader-extra-config '{\"enable_multithread_load\": true}' --disaggregation-transfer-backend mooncake --mamba-full-memory-ratio 0.062 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --disable-radix-cache --cuda-graph-max-bs-decode 64 --context-length 1048576 --disaggregation-ib-device $IB --speculative-algorithm DFLASH --speculative-draft-model-path /data/Kimi-K3.1-Dflash2-Preview --speculative-draft-attention-backend cutedsl_mla --speculative-draft-kv-cache-dtype fp8_e4m3 --speculative-num-draft-tokens 5 --speculative-use-block-verification --enable-linear-replayssm-spec --mm-io-worker-num 16 --dsa-prefill-backend prims_ts --dsa-decode-backend prims_ts " \
  -y

FROM harbor.local.clusters/bp/lmsysorg/sglang:dev-v0.5.23-k3.1-fix
# Base sglang is editable at /root/sglang/python (commit 44bf3993c6); the 7 files
# below were byte-identical to Infrawaves infrawaves/dev/v0.5.23-k31 15a7502e24.
COPY python/ /root/sglang/python/
COPY MD5SUMS /root/kda-k31-overlay.MD5SUMS
RUN cd /root/sglang && sed 's# \./# #' /root/kda-k31-overlay.MD5SUMS | md5sum -c - \
 && python3 -c "import sglang.srt.mem_cache.mla_host_node_leader as m, sglang.srt.mem_cache.storage.mooncake_store.mooncake_store as s, sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler as a; assert hasattr(m, 'maybe_create_kda_node_aggregate_context'); assert hasattr(s.MooncakeStore, 'batch_io_node_aggregated_mamba'); print('kda overlay ok')"
LABEL ai.lvning.kda.source="Infrawaves/sglang kda-l3-node-aggregate-k31 (uncommitted) on 15a7502e24"

"""Local multi-process check of KDA node aggregation (gloo, CPU, 4 ranks = 2 nodes x 2).

Real: NodeShardedHostTensorAllocator, maybe_create_kda_node_aggregate_context,
MambaPoolHost (node_aggregate), MooncakeStore aggregated paths,
HybridCacheController slot exchange / MIN-reduce.
Faked: Mooncake client (in-process dict with real memcpy), memfd (tmp file),
MADV_POPULATE_WRITE, ranks_per_host, get_memory.
"""

import ctypes
import importlib.util
import os
import sys
import tempfile
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

REPO = os.environ["KDA_REPO"]
WORLD = 4
RANKS_PER_NODE = 2
SLOT_OF = lambda r: 2 + r  # deliberately different per rank
LAYERS, CAP = 2, 8
TEMPORAL_TAIL = (4, 8)  # float32
CONV_TAILS = [(5, 3), (7, 2)]  # bfloat16


class RealCopyStore:
    instances = []

    def __init__(self):
        self.data = {}
        self.exist_calls = 0
        self.get_calls = 0
        self.put_calls = 0
        type(self).instances.append(self)

    def setup(self, *a, **k):
        return 0

    def register_buffer(self, *a, **k):
        return 0

    def put(self, key, value, *args):
        self.data[key] = value
        return 0

    def is_exist(self, key):
        return 1 if key in self.data else 0

    def get(self, key):
        return self.data.get(key)

    def batch_is_exist(self, keys):
        self.exist_calls += 1
        return [1 if k in self.data else 0 for k in keys]

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, *args):
        self.put_calls += 1
        for k, pl, sl in zip(keys, ptrs, sizes):
            self.data[k] = b"".join(ctypes.string_at(p, s) for p, s in zip(pl, sl))
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
        self.get_calls += 1
        out = []
        for k, pl, sl in zip(keys, ptrs, sizes):
            blob = self.data.get(k)
            if blob is None:
                out.append(-1)
                continue
            off = 0
            for p, s in zip(pl, sl):
                chunk = blob[off : off + s]
                ctypes.memmove(p, chunk, len(chunk))
                off += s
            out.append(len(blob))
        return out


def slot_tensors(rank):
    g = torch.Generator().manual_seed(1000 + rank)
    t = torch.randn((LAYERS, 1) + TEMPORAL_TAIL, generator=g)
    convs = [
        torch.randn((LAYERS, 1) + tail, generator=g).to(torch.bfloat16)
        for tail in CONV_TAILS
    ]
    return t, convs


def slot_bytes(rank):
    t, convs = slot_tensors(rank)
    return [x.contiguous().view(torch.uint8).flatten().numpy().tobytes() for x in (t, *convs)]


def write_slot(pool, rank, slot):
    t, convs = slot_tensors(rank)
    pool.temporal_buffer[slot].copy_(t)
    for buf, c in zip(pool.conv_buffer, convs):
        buf[slot].copy_(c)


def read_slot_bytes(pool, slot):
    arrs = [pool.temporal_buffer[slot], *[b[slot] for b in pool.conv_buffer]]
    return [a.contiguous().view(torch.uint8).flatten().numpy().tobytes() for a in arrs]


def load_test_helpers():
    path = os.path.join(
        REPO, "test/registered/unit/mem_cache/test_mooncake_group_semantics.py"
    )
    spec = importlib.util.spec_from_file_location("kda_group_sem", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker(rank, shm_dir, store_file):
    sys.path.insert(0, os.path.join(REPO, "python"))
    sys.path.insert(0, REPO)
    dist.init_process_group(
        "gloo", init_method=f"file://{store_file}", rank=rank, world_size=WORLD
    )
    try:
        run(rank, shm_dir)
        print(f"[rank {rank}] ALL PASS", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"[rank {rank}] FAILED", flush=True)
        os._exit(1)
    dist.barrier()


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(rank, shm_dir):
    import sglang.srt.mem_cache.mla_host_node_leader as nl
    import sglang.srt.mem_cache.storage.mmap.mmap_allocator as mm
    from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
    from sglang.srt.mem_cache.hicache_storage import (
        PoolHitPolicy,
        PoolName,
        PoolTransfer,
    )
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    import sglang.srt.mem_cache.pool_host.mamba as mamba_mod

    def fake_create_shm(dims, dtype):
        n = 1
        for d in dims:
            n *= d
        n *= torch.empty([], dtype=dtype).element_size()
        tmp = os.path.join(shm_dir, f"tmp_{os.getpid()}")
        fd = os.open(tmp, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        os.ftruncate(fd, -(-n // mmap_pagesize) * mmap_pagesize)
        os.rename(tmp, os.path.join(shm_dir, f"shm_{os.getpid()}_{fd}"))
        return fd

    import mmap

    mmap_pagesize = mmap.PAGESIZE

    def fake_open(self, owner, stride):
        pid, owner_fd, leader_stride = owner
        check(leader_stride == stride, "stride mismatch")
        return os.open(os.path.join(shm_dir, f"shm_{pid}_{owner_fd}"), os.O_RDWR)

    patches = [
        patch.object(nl, "ranks_per_host", lambda: RANKS_PER_NODE),
        patch.object(
            nl,
            "get_memory",
            lambda: SimpleNamespace(enable_kda_hicache_node_aggregate=True),
        ),
        patch.object(mamba_mod, "host_memory_budget_bytes", lambda r, **k: 1 << 40),
    ]
    if sys.platform != "linux":
        # macOS has no memfd, /proc or MADV_POPULATE_WRITE.
        patches += [
            patch.object(nl, "create_shm", fake_create_shm),
            patch.object(
                nl.NodeShardedHostTensorAllocator, "_open_shared_fd", fake_open
            ),
            patch.object(mm, "_prefault_shm_shard", lambda *a, **k: None),
        ]
    if rank == 0:
        print(f"platform={sys.platform} real_memfd={sys.platform == 'linux'}", flush=True)
    for p in patches:
        p.start()

    # ---- context + real host pool on node-sharded memfd
    ctx = nl.maybe_create_kda_node_aggregate_context(
        tp_cache_group=dist.group.WORLD
    )
    check(ctx is not None, "context not created")
    node_idx, node_rank = divmod(rank, RANKS_PER_NODE)
    check(ctx.node_index == node_idx and ctx.node_rank == node_rank, "node ids")
    check(ctx.is_leader == (node_rank == 0), "leader flag")
    check(ctx.num_nodes == 2 and ctx.node_size == 2, "node counts")

    info = MambaStateBufferInfo(
        temporal=torch.zeros((LAYERS, CAP) + TEMPORAL_TAIL, dtype=torch.float32),
        conv=tuple(
            torch.zeros((LAYERS, CAP) + t, dtype=torch.bfloat16) for t in CONV_TAILS
        ),
    )
    pool = mamba_mod.MambaPoolHost(
        info,
        1.5,
        0,
        device_capacity=CAP,
        pin_memory=False,
        layout="page_first_direct",
        node_aggregate=ctx,
    )
    check(pool.allocator is ctx.allocator, "allocator not node-sharded")
    check(len(ctx.allocator.full_buffers()) == 3, "expected 3 shared buffers")
    own_ptrs = ctx.allocator.peer_base_ptrs(pool.temporal_buffer)
    check(own_ptrs[node_rank] == pool.temporal_buffer.data_ptr(), "own share ptr")
    print(
        f"[rank {rank}] pool size={pool.size} stride="
        f"{own_ptrs[1] - own_ptrs[0]} node {node_idx}/{ctx.num_nodes} "
        f"{'leader' if ctx.is_leader else 'follower'}",
        flush=True,
    )

    # ---- store with a fake client that really copies bytes
    helpers = load_test_helpers()
    with patch.dict(
        "sys.modules",
        helpers._fake_mooncake_modules(RealCopyStore, helpers.ReplicateConfigWithGroupIds),
    ):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        cfg = helpers._make_config(is_mla_model=True, tp_rank=rank, tp_size=WORLD)
        store = MooncakeStore(cfg)
    backend = RealCopyStore.instances[-1]
    store.mem_pool_host = SimpleNamespace(kv_buffer=None, page_size=1)
    store.register_mem_host_pool_v2(pool, PoolName.MAMBA)
    check(store.kda_node_aggregate is ctx, "store did not pick up context")

    ctrl = object.__new__(HybridCacheController)
    ctrl.kda_node_aggregate = ctx
    ctrl.storage_backend = store
    op = lambda: SimpleNamespace(skip_reason=None, request_id="r")

    def transfer(key, slot):
        return PoolTransfer(
            name=PoolName.MAMBA,
            keys=[key],
            host_indices=torch.tensor([slot]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

    my_slot = SLOT_OF(rank)
    node_ranks = [node_idx * RANKS_PER_NODE + i for i in range(RANKS_PER_NODE)]
    key_of = lambda page: f"{page}_kdanode{node_idx}of2_tp{WORLD}_v1"

    # ---- S1: backup, every rank in its own (different) slot
    write_slot(pool, rank, my_slot)
    dist.barrier()
    results = {}
    rest = ctrl._set_node_aggregated_mamba(op(), [transfer("pageA", my_slot)], results)
    check(rest == [] and results[PoolName.MAMBA] == [True], f"S1 results {results}")
    dist.barrier()
    if ctx.is_leader:
        expected = b"".join(b"".join(slot_bytes(r)) for r in node_ranks)
        page_keys = {k for k in backend.data if k.startswith("page")}
        check(page_keys == {key_of("pageA")}, f"S1 keys {page_keys}")
        check(backend.data[key_of("pageA")] == expected, "S1 object bytes differ")
        print(f"[rank {rank}] S1 ok: one object, {len(expected)} B = sum of {len(node_ranks)} rank slots", flush=True)
    else:
        check(backend.put_calls == 0 and backend.exist_calls == 0, "follower touched Mooncake")

    # ---- S2: wipe host slots, restore through the leader
    for t in (pool.temporal_buffer, *pool.conv_buffer):
        t.zero_()
    dist.barrier()
    results = {}
    ctrl._get_node_aggregated_mamba(op(), [transfer("pageA", my_slot)], results)
    check(results[PoolName.MAMBA] == [True], f"S2 results {results}")
    dist.barrier()
    check(read_slot_bytes(pool, my_slot) == slot_bytes(rank), "S2 restored bytes differ")
    check(backend.get_calls == (1 if ctx.is_leader else 0), "S2 get_calls")
    print(f"[rank {rank}] S2 ok: slot {my_slot} restored byte-exact", flush=True)

    # ---- S3: rank 1 has no Mamba transfer in this backup -> node 0 fails together
    results = {}
    transfers = [] if rank == 1 else [transfer("pageB", my_slot)]
    ctrl._set_node_aggregated_mamba(op(), transfers, results)
    if node_idx == 0:
        if rank == 0:
            check(results[PoolName.MAMBA] == [False], f"S3 leader must fail {results}")
        else:
            check(PoolName.MAMBA not in results, f"S3 skipped rank {results}")
        check(key_of("pageB") not in backend.data, "S3 node0 wrote despite peer skip")
    else:
        check(results[PoolName.MAMBA] == [True], f"S3 node1 results {results}")
    dist.barrier()
    if rank == 2:
        check(key_of("pageB") in backend.data, "S3 node1 leader should have written")
    print(f"[rank {rank}] S3 ok", flush=True)

    # ---- S4: restore with rank 3 skipping -> node 1 leader must not read
    gets_before = backend.get_calls
    results = {}
    transfers = [] if rank == 3 else [transfer("pageB", my_slot)]
    ctrl._get_node_aggregated_mamba(op(), transfers, results)
    if rank == 2:
        check(results[PoolName.MAMBA] == [False], f"S4 leader {results}")
        check(backend.get_calls == gets_before, "S4 leader read despite peer skip")
    if rank == 3:
        check(PoolName.MAMBA not in results, "S4 skipped rank")
    if rank == 0:
        # no skip on node 0: the leader reads, but pageB was never stored there
        check(results[PoolName.MAMBA] == [False], f"S4 rank0 {results}")
        check(backend.get_calls == gets_before + 1, "S4 node0 leader should read")
    dist.barrier()
    print(f"[rank {rank}] S4 ok", flush=True)

    # ---- S5: existence. node0 has pageA only, node1 has pageA+pageB
    res = store.batch_exists_v2(
        ["pageA", "pageB", "pageC"],
        [
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["pageC"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ],
    )
    # trailing=1: restore point p works when page p's checkpoint exists
    if not ctx.is_leader:
        check(res.restorable_prefix_pages == [1, 2, 3], f"S5 follower {res}")
    elif node_idx == 0:
        check(res.restorable_prefix_pages == [1], f"S5 node0 leader {res}")
    else:
        check(res.restorable_prefix_pages == [1, 2], f"S5 node1 leader {res}")
    exist_before = backend.exist_calls
    if not ctx.is_leader:
        check(exist_before == 0, "S5 follower queried Mooncake")
    # cross-rank intersection as the controller does it (MIN over flags)
    flags = torch.zeros(4, dtype=torch.int)
    flags[res.restorable_prefix_pages] = 1
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    check(flags.tolist() == [0, 1, 0, 0], f"S5 intersection {flags.tolist()}")
    print(f"[rank {rank}] S5 ok: restorable={res.restorable_prefix_pages}, group MIN={flags.tolist()}", flush=True)

    # ---- S6: short object on node 0 leader -> read fails on the whole node
    if rank == 0:
        backend.data[key_of("pageA")] = backend.data[key_of("pageA")][:-1]
    dist.barrier()
    for t in (pool.temporal_buffer, *pool.conv_buffer):
        t.zero_()
    dist.barrier()
    results = {}
    ctrl._get_node_aggregated_mamba(op(), [transfer("pageA", my_slot)], results)
    if rank == 0:
        check(results[PoolName.MAMBA] == [False], f"S6 short read must fail: {results}")
    else:
        check(results[PoolName.MAMBA] == [True], f"S6 {rank} {results}")
    print(f"[rank {rank}] S6 ok", flush=True)

    # ---- S7: real _page_transfer_sidecar; rank 1's KV read was short (local skip)
    for t in (pool.temporal_buffer, *pool.conv_buffer):
        t.zero_()
    dist.barrier()
    ctrl.storage_page_group = 1
    ctrl.mem_pool_host = SimpleNamespace(
        entry_map={PoolName.MAMBA: SimpleNamespace(host_pool=SimpleNamespace(page_size=1))}
    )
    operation = SimpleNamespace(
        request_id="r7",
        is_terminated=lambda: False,
        hash_value=["pageA"],
        skip_reason=None,
        pool_transfers=[transfer("pageA", my_slot)],
        prefix_keys=None,
        sidecar_hash_values=None,
        sidecar_hit_pages=0,
        sidecar_ts=None,
        start_page=0,
        storage_plans=None,
    )
    pool_hits = ctrl._page_transfer_sidecar(operation, 0 if rank == 1 else 1)
    dist.barrier()
    if rank == 1:
        check(pool_hits == {} and operation.skip_reason == "kv_short", f"S7 r1 {pool_hits}")
    elif rank == 0:
        check(pool_hits == {PoolName.MAMBA: 0}, f"S7 r0 {pool_hits}")
        check(operation.skip_reason == "node_peer_skip", "S7 r0 skip reason")
    else:
        check(pool_hits == {PoolName.MAMBA: 1}, f"S7 node1 {pool_hits}")
        check(read_slot_bytes(pool, my_slot) == slot_bytes(rank), "S7 node1 bytes")
    if node_idx == 0:
        check(read_slot_bytes(pool, my_slot) != slot_bytes(rank), "S7 node0 restored despite skip")
    print(f"[rank {rank}] S7 ok: {pool_hits}", flush=True)

    # ---- S8: real _page_backup_with_stable_layout (backup_skip ranks, Mamba only)
    ctrl.storage_config = SimpleNamespace(is_storage_writer=False)
    ctrl.storage_backend_type = "mooncake"
    ctrl.draft_node_merge = None
    write_slot(pool, rank, my_slot)
    dist.barrier()
    hits_seen = {}
    op8 = SimpleNamespace(
        id=8,
        pool_transfers=[transfer("pageD", my_slot)],
        prefix_keys=None,
        hash_value=["pageD"],
        start_page=0,
        completed_tokens=0,
        pool_storage_result=SimpleNamespace(
            update_extra_pool_hit_pages=lambda h: hits_seen.update(h)
        ),
    )
    ctrl.page_size = 1
    ctrl._page_backup_with_stable_layout(op8)
    dist.barrier()
    check(op8.completed_tokens == 1, f"S8 completed_tokens {op8.completed_tokens}")
    if ctx.is_leader:
        check(key_of("pageD") in backend.data, "S8 leader did not write")
    # an op with no Mamba on rank 1: whole node reports failure, others fine
    op9 = SimpleNamespace(
        id=9,
        pool_transfers=[] if rank == 1 else [transfer("pageE", my_slot)],
        prefix_keys=None,
        hash_value=["pageE"],
        start_page=0,
        completed_tokens=0,
        pool_storage_result=SimpleNamespace(update_extra_pool_hit_pages=lambda h: None),
    )
    ctrl._page_backup_with_stable_layout(op9)
    dist.barrier()
    expect = {0: 0, 1: 0, 2: 1, 3: 1}[rank]
    check(op9.completed_tokens == expect, f"S8b rank {rank} completed {op9.completed_tokens}")
    print(f"[rank {rank}] S8 ok", flush=True)

    for p in patches:
        p.stop()


if __name__ == "__main__":
    import torch.multiprocessing as tmp

    shm_dir = tempfile.mkdtemp(prefix="kda_shm_")
    store_file = os.path.join(tempfile.mkdtemp(prefix="kda_store_"), "store")
    tmp.spawn(worker, args=(shm_dir, store_file), nprocs=WORLD, join=True)
    print("DONE")

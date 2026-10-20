#!/usr/bin/env python3
"""
ANN Index Maintenance Scheduler  (v12)
======================================
Two-stage pipeline:
  1. Single-pass greedy heuristic  → candidate batch
  2. NetworkX network-simplex foreground flow

New in v12
----------
* Consecutive-skip early termination in the greedy loop.
  A counter tracks how many jobs in a row have been skipped because the
  target node had insufficient capacity even after rerouting.  When the
  counter reaches `max_consecutive_skips` (default 10, --max-skips flag),
  the job selection loop exits early.  The counter resets to zero whenever
  a job is successfully scheduled, so isolated failures do not accumulate.

  Rationale: once the scheduler has failed to place N jobs in a row it is
  a strong signal that all remaining capacity has been exhausted and further
  iteration is wasteful.  k-capped rejections do NOT count toward the
  consecutive-skip total because they are policy gates, not capacity failures.

All output is written to a log file (default: scheduler.log).
Nothing is printed to the terminal.  Use --log to override.

Usage:
    python scheduler.py                               # reads cluster_config.json
    python scheduler.py cluster_config.json
    python scheduler.py cluster_config.json --k 2
    python scheduler.py cluster_config.json --max-skips 5
    python scheduler.py cluster_config.json --log /var/log/scheduler.log
"""

import json
import logging
import math
import sys
import time
import argparse
from typing import Dict, List, Tuple

import networkx as nx

DEFAULT_CONFIG    = "cluster_config.json"
DEFAULT_LOG       = "scheduler.log"
DEFAULT_MAX_SKIPS = 10

FLOW_SCALE   = 1_000
WEIGHT_SCALE = 1_000


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: str) -> None:
    """
    Configure the root logger to write everything to `log_path`.
    No StreamHandler is added, so nothing reaches the terminal.
    """
    logging.basicConfig(
        filename=log_path,
        filemode="a",                          # append — safe for repeated runs
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Scaling helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_flow(qps: float) -> int:
    return max(1, round(qps * FLOW_SCALE))

def to_weight(value: float) -> int:
    return -max(1, round(value * WEIGHT_SCALE))


# ─────────────────────────────────────────────────────────────────────────────
# Blackbox: reindex reward for (shard, server)
# ─────────────────────────────────────────────────────────────────────────────

def reindex_value(shard_id: int, server_id: int,
                  maintenance_cost: float, is_hot: bool) -> float:
    """
    Placeholder reward signal — replace with real staleness / priority logic.
    Deterministic and hash-derived for reproducible results across runs.
    Contract: always returns a strictly positive float.
    """
    h    = math.sin(shard_id * 2654.435769 + server_id * 1234.567) * 0.5 + 0.5
    base = maintenance_cost * (0.8 + h)
    return base * (1.5 if is_hot else 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# State initialisation  (greedy stage)
# ─────────────────────────────────────────────────────────────────────────────

def build_initial_state(
    config: dict,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, Dict[int, float]]]:
    meta   = config["meta"]
    nodes  = config["nodes"]
    shards = config["shards"]

    node_capacity = meta["node_capacity"]

    node_qps: Dict[int, float] = {
        int(nid): nd["total_load_qps"] for nid, nd in nodes.items()
    }

    node_available_bg: Dict[int, float] = {
        nid: max(0.0, node_capacity - qps)
        for nid, qps in node_qps.items()
    }

    shard_routing_qps: Dict[int, Dict[int, float]] = {}
    for sid, sd in shards.items():
        sid = int(sid)
        shard_routing_qps[sid] = {
            int(nid): frac * sd["qps"]
            for nid, frac in sd["routing"].items()
        }

    return node_qps, node_available_bg, shard_routing_qps


# ─────────────────────────────────────────────────────────────────────────────
# Job generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_jobs(config: dict) -> List[dict]:
    jobs: List[dict] = []
    for sid, sd in config["shards"].items():
        sid    = int(sid)
        is_hot = sd["is_hot"]
        for nid in sd["replicas"]:
            cost  = sd["maintenance_cost"][str(nid)]
            value = reindex_value(
                shard_id         = sid,
                server_id        = nid,
                maintenance_cost = cost,
                is_hot           = is_hot,
            )
            jobs.append({
                "shard_id": sid,
                "node_id":  nid,
                "cost":     cost,
                "value":    value,
                "is_hot":   is_hot,
            })

    jobs.sort(key=lambda j: j["value"], reverse=True)
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Rerouting helpers  (greedy stage)
# ─────────────────────────────────────────────────────────────────────────────

def try_reroute(
    target_node: int,
    needed_bg: float,
    node_qps: Dict[int, float],
    node_available_bg: Dict[int, float],
    shard_routing_qps: Dict[int, Dict[int, float]],
    config: dict,
) -> Tuple[bool, List[Tuple[int, int, int, float]]]:
    if needed_bg <= 1e-9:
        return True, []

    shards_on_target = [
        (sid, routing[target_node])
        for sid, routing in shard_routing_qps.items()
        if routing.get(target_node, 0.0) > 1e-9
    ]

    t_available_bg = dict(node_available_bg)
    t_node_qps     = dict(node_qps)
    t_routing      = {sid: dict(r) for sid, r in shard_routing_qps.items()}

    tentative_reroutes: List[Tuple[int, int, int, float]] = []
    freed_bg = 0.0

    for sid, qps_on_target in shards_on_target:
        if freed_bg >= needed_bg - 1e-9:
            break

        replicas       = config["shards"][str(sid)]["replicas"]
        other_replicas = [r for r in replicas if r != target_node]

        for replica in other_replicas:
            if freed_bg >= needed_bg - 1e-9:
                break

            replica_headroom = t_available_bg.get(replica, 0.0)
            if replica_headroom <= 1e-9:
                continue

            current_on_target = t_routing[sid].get(target_node, 0.0)
            delta_qps         = min(current_on_target, replica_headroom)

            if delta_qps <= 1e-9:
                continue

            t_routing[sid][target_node] = current_on_target - delta_qps
            t_routing[sid][replica]     = t_routing[sid].get(replica, 0.0) + delta_qps
            t_node_qps[target_node]    -= delta_qps
            t_node_qps[replica]        += delta_qps
            t_available_bg[target_node] += delta_qps
            t_available_bg[replica]     -= delta_qps

            freed_bg += delta_qps
            tentative_reroutes.append((sid, target_node, replica, delta_qps))

    if freed_bg >= needed_bg - 1e-9:
        return True, tentative_reroutes
    return False, []


def apply_reroutes(
    reroutes: List[Tuple[int, int, int, float]],
    node_qps: Dict[int, float],
    node_available_bg: Dict[int, float],
    shard_routing_qps: Dict[int, Dict[int, float]],
) -> None:
    for sid, from_node, to_node, delta_qps in reroutes:
        shard_routing_qps[sid][from_node] = (
            shard_routing_qps[sid].get(from_node, 0.0) - delta_qps
        )
        shard_routing_qps[sid][to_node] = (
            shard_routing_qps[sid].get(to_node, 0.0) + delta_qps
        )
        node_qps[from_node]           -= delta_qps
        node_qps[to_node]             += delta_qps
        node_available_bg[from_node]  += delta_qps
        node_available_bg[to_node]    -= delta_qps


# ─────────────────────────────────────────────────────────────────────────────
# Greedy scheduler  (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduler(config: dict, k: int = 1,
                  max_consecutive_skips: int = DEFAULT_MAX_SKIPS) -> dict:
    """
    Single-pass greedy scheduler with consecutive-skip early termination.

    Parameters
    ----------
    config : dict
        Parsed cluster_config.json.
    k : int
        Maximum number of replicas from one replica set that may be scheduled
        in a single batch.  k=1 means at most one reindex per shard.
    max_consecutive_skips : int
        If this many jobs in a row are skipped due to insufficient capacity
        (even after rerouting), the selection loop exits early.  The counter
        resets to zero whenever a job is successfully scheduled.
        k-capped rejections do NOT count — they are policy gates, not
        capacity failures.

    Returns
    -------
    dict with keys:
      scheduled               – greedy candidate batch (input to FG stage)
      skipped                 – jobs that could not fit (capacity infeasible)
      k_capped                – jobs rejected by the k-quota gate
      reroute_log             – reroutes applied, keyed to the triggering job
      consecutive_skips_at_exit – counter value when the loop ended
      early_exit              – True if the loop exited due to the skip limit
      final_node_available_bg – remaining spare capacity after the pass
      final_node_qps          – QPS load per node after all reroutes
    """
    node_qps, node_available_bg, shard_routing_qps = build_initial_state(config)
    jobs = generate_jobs(config)

    replica_set_scheduled: Dict[int, int] = {}

    scheduled:   List[dict] = []
    skipped:     List[dict] = []
    k_capped:    List[dict] = []
    reroute_log: List[dict] = []

    consecutive_skips = 0   # counts back-to-back capacity failures
    early_exit        = False

    for job in jobs:

        # ── Consecutive-skip early termination ───────────────────────────────
        if consecutive_skips >= max_consecutive_skips:
            early_exit = True
            log.info(
                f"  [greedy] Early exit — {consecutive_skips} consecutive "
                f"capacity failures reached limit ({max_consecutive_skips})."
            )
            break

        shard_id    = job["shard_id"]
        target_node = job["node_id"]
        cost        = job["cost"]

        # ── k-cap check (policy gate — does NOT affect consecutive_skips) ────
        already_scheduled = replica_set_scheduled.get(shard_id, 0)
        if already_scheduled >= k:
            k_capped.append(job)
            continue

        available = node_available_bg[target_node]

        if available >= cost - 1e-9:
            # ── Direct schedule ───────────────────────────────────────────────
            node_available_bg[target_node]  -= cost
            replica_set_scheduled[shard_id]  = already_scheduled + 1
            scheduled.append(job)
            consecutive_skips = 0   # reset on success

        else:
            # ── Attempt rerouting to free spare capacity ──────────────────────
            needed  = cost - available
            success, reroutes = try_reroute(
                target_node       = target_node,
                needed_bg         = needed,
                node_qps          = node_qps,
                node_available_bg = node_available_bg,
                shard_routing_qps = shard_routing_qps,
                config            = config,
            )

            if success:
                apply_reroutes(
                    reroutes, node_qps, node_available_bg, shard_routing_qps,
                )
                node_available_bg[target_node]  -= cost
                replica_set_scheduled[shard_id]  = already_scheduled + 1
                scheduled.append(job)
                reroute_log.append({
                    "job": job,
                    "reroutes": [
                        {
                            "shard_id":  r[0],
                            "from_node": r[1],
                            "to_node":   r[2],
                            "delta_qps": round(r[3], 4),
                        }
                        for r in reroutes
                    ],
                })
                consecutive_skips = 0   # reset on success

            else:
                # Capacity failure — increment consecutive-skip counter
                skipped.append(job)
                consecutive_skips += 1
                log.info(
                    f"  [greedy] Skipped shard={shard_id:2d} node={target_node}"
                    f"  cost={cost:.4f}  consecutive_skips={consecutive_skips}"
                )

    return {
        "scheduled":               scheduled,
        "skipped":                 skipped,
        "k_capped":                k_capped,
        "reroute_log":             reroute_log,
        "consecutive_skips_at_exit": consecutive_skips,
        "early_exit":              early_exit,
        "final_node_available_bg": node_available_bg,
        "final_node_qps":          node_qps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Foreground graph  (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fg_caps(nodes: dict, node_capacity: float) -> Dict[int, int]:
    return {int(nid): to_flow(node_capacity) for nid in nodes}


def build_foreground_graph(
    shards: dict,
    fg_caps: Dict[int, int],
    maintenance_deductions: Dict[int, float],
) -> nx.DiGraph:
    G                = nx.DiGraph()
    total_qps_scaled = 0

    for sid_str, sdata in shards.items():
        sid        = int(sid_str)
        qps_scaled = to_flow(sdata["qps"])
        total_qps_scaled += qps_scaled
        shard_node = f"shard_{sid}"
        G.add_node(shard_node, demand=-qps_scaled)

        for server_id in sdata["replicas"]:
            server_node = f"server_{server_id}"
            if server_node not in G:
                G.add_node(server_node, demand=0)
            G.add_edge(shard_node, server_node,
                       capacity=qps_scaled, weight=0)

    G.add_node("SNK", demand=total_qps_scaled)

    for nid, base_cap in fg_caps.items():
        server_node = f"server_{nid}"
        if server_node not in G:
            continue
        deduction     = to_flow(maintenance_deductions.get(nid, 0.0))
        effective_cap = max(0, base_cap - deduction)
        G.add_edge(server_node, "SNK", capacity=effective_cap, weight=0)

    return G


def run_foreground_flow(
    batch: List[dict],
    shards: dict,
    fg_caps: Dict[int, int],
) -> Tuple[bool, nx.DiGraph, dict]:
    deductions: Dict[int, float] = {}
    for job in batch:
        nid = job["node_id"]
        deductions[nid] = deductions.get(nid, 0.0) + job["cost"]

    G = build_foreground_graph(shards, fg_caps, deductions)

    try:
        flow_dict = nx.min_cost_flow(G)
        return True, G, flow_dict
    except (nx.NetworkXUnfeasible, nx.NetworkXError):
        return False, G, {}


# ─────────────────────────────────────────────────────────────────────────────
# Cluster utilization metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_cluster_utilization(
    meta: dict,
    scheduled_jobs: List[dict],
    final_node_qps: Dict[int, float],
) -> Tuple[float, float, float, float, float]:
    node_capacity     = meta["node_capacity"]
    num_nodes         = meta["num_nodes"]
    total_capacity    = node_capacity * num_nodes
    total_qps         = meta["total_qps"]
    total_maintenance = sum(j["cost"] for j in scheduled_jobs)

    deductions: Dict[int, float] = {}
    for job in scheduled_jobs:
        nid = job["node_id"]
        deductions[nid] = deductions.get(nid, 0.0) + job["cost"]

    overall_util = (total_qps + total_maintenance) / total_capacity

    node_utils = [
        (final_node_qps.get(nid, 0.0) + deductions.get(nid, 0.0)) / node_capacity
        for nid in final_node_qps
    ]
    avg_node_util = sum(node_utils) / len(node_utils) if node_utils else 0.0

    return total_qps, total_maintenance, total_capacity, overall_util, avg_node_util


# ─────────────────────────────────────────────────────────────────────────────
# Log output
# ─────────────────────────────────────────────────────────────────────────────

def log_results(
    greedy_result: dict,
    fg_feasible: bool,
    fg_caps: Dict[int, int],
    meta: dict,
    elapsed_s: float,
    k: int,
    max_consecutive_skips: int,
) -> None:
    W = "-" * 66

    # ── Stage 1: greedy summary ───────────────────────────────────────────────
    log.info("=" * 66)
    log.info(f"[1/2] GREEDY HEURISTIC OUTPUT  (k={k})")
    log.info("=" * 66)
    for job in greedy_result["scheduled"]:
        hot_tag = " [HOT]" if job["is_hot"] else ""
        log.info(
            f"  shard={job['shard_id']:2d}  node={job['node_id']}"
            f"  cost={job['cost']:9.4f}  value={job['value']:10.4f}{hot_tag}"
        )
    log.info(f"  Greedy scheduled : {len(greedy_result['scheduled'])}")
    log.info(f"  Greedy skipped   : {len(greedy_result['skipped'])}")
    log.info(
        f"  Greedy k-capped  : {len(greedy_result['k_capped'])}"
        f"  (replica-set quota k={k} hit)"
    )

    # ── Consecutive-skip termination summary ──────────────────────────────────
    log.info(W)
    log.info("  CONSECUTIVE-SKIP TERMINATION")
    log.info(W)
    if greedy_result["early_exit"]:
        log.warning(
            f"  Loop exited EARLY — {max_consecutive_skips} consecutive "
            f"capacity failures.  Remaining jobs were not evaluated."
        )
    else:
        log.info(
            f"  Loop ran to completion — no early exit."
        )
    log.info(
        f"  {'Consecutive skips at exit':<40}: "
        f"{greedy_result['consecutive_skips_at_exit']:>4}"
        f"  (limit={max_consecutive_skips})"
    )

    if greedy_result["skipped"]:
        log.info(W)
        log.info("  SKIPPED  (spare capacity insufficient even after rerouting)")
        log.info(W)
        for job in greedy_result["skipped"]:
            log.info(
                f"  shard={job['shard_id']:2d}  node={job['node_id']}"
                f"  cost={job['cost']:9.4f}  value={job['value']:10.4f}"
            )

    if greedy_result["reroute_log"]:
        log.info(W)
        log.info("  REROUTES APPLIED  (greedy stage)")
        log.info(W)
        for entry in greedy_result["reroute_log"]:
            j = entry["job"]
            log.info(
                f"  > job  shard={j['shard_id']:2d}  node={j['node_id']}"
                f"  cost={j['cost']:.4f}  value={j['value']:.4f}"
            )
            for r in entry["reroutes"]:
                log.info(
                    f"      shard {r['shard_id']:2d}: {r['delta_qps']:8.2f} QPS"
                    f"  node {r['from_node']} -> node {r['to_node']}"
                )

    # ── Stage 2: foreground flow result ──────────────────────────────────────
    log.info("=" * 66)
    log.info("[2/2] FOREGROUND FLOW RESULT")
    log.info("=" * 66)
    log.info(f"  Batch size fed into FG graph : {len(greedy_result['scheduled'])}")

    if fg_feasible:
        log.info("  Result : FEASIBLE — all query load routable with batch deductions")
    else:
        log.warning("  Result : INFEASIBLE — batch exceeds foreground capacity")

    # ── Per-node deduction table ──────────────────────────────────────────────
    deductions: Dict[int, float] = {}
    for job in greedy_result["scheduled"]:
        nid = job["node_id"]
        deductions[nid] = deductions.get(nid, 0.0) + job["cost"]

    log.info(W)
    log.info("  PER-NODE FOREGROUND IMPACT")
    log.info(
        f"  {'Node':>4}  {'base_cap':>10}  {'deducted':>10}"
        f"  {'eff_cap':>10}  {'avail_bg':>10}  {'qps':>10}  {'node_util':>9}"
    )
    log.info(W)
    node_capacity = meta["node_capacity"]
    for nid in sorted(greedy_result["final_node_available_bg"]):
        base_cap  = fg_caps[nid] / FLOW_SCALE
        ded       = deductions.get(nid, 0.0)
        eff_cap   = max(0.0, base_cap - ded)
        avail_bg  = greedy_result["final_node_available_bg"][nid]
        qps       = greedy_result["final_node_qps"][nid]
        node_util = (qps + ded) / node_capacity
        log.info(
            f"  {nid:>4}  {base_cap:>10.4f}  {ded:>10.4f}"
            f"  {eff_cap:>10.4f}  {avail_bg:>10.4f}  {qps:>10.4f}  {node_util:>9.2%}"
        )

    # ── Committed batch summary ───────────────────────────────────────────────
    if greedy_result["scheduled"]:
        total_val  = sum(j["value"] for j in greedy_result["scheduled"])
        total_cost = sum(j["cost"]  for j in greedy_result["scheduled"])
        log.info(W)
        log.info("  COMMITTED BATCH SUMMARY")
        log.info(W)
        log.info(f"  Total committed value : {total_val:.4f}")
        log.info(f"  Total committed cost  : {total_cost:.4f}")

    # ── Cluster utilization ───────────────────────────────────────────────────
    (total_qps, total_maintenance,
     total_capacity, overall_util, avg_node_util) = compute_cluster_utilization(
        meta, greedy_result["scheduled"], greedy_result["final_node_qps"]
    )

    maint_fraction = total_maintenance / total_capacity

    log.info("=" * 66)
    log.info("  CLUSTER UTILIZATION")
    log.info("=" * 66)
    log.info(
        f"  {'Total cluster capacity':<44}: {total_capacity:>10.2f}"
        f"  ({meta['num_nodes']} nodes x {meta['node_capacity']:.2f})"
    )
    log.info(f"  {'QPS flowed  (query traffic)':<44}: {total_qps:>10.2f}")
    log.info(f"  {'Maintenance reserved  (committed jobs)':<44}: {total_maintenance:>10.4f}")
    log.info(f"  {'Combined load  (qps + maintenance)':<44}: {total_qps + total_maintenance:>10.4f}")
    log.info(W)
    log.info(
        f"  {'Overall cluster utilization':<44}: {overall_util:>10.2%}"
        f"  <- (qps + maintenance) / capacity"
    )
    log.info(
        f"  {'Average node utilization':<44}: {avg_node_util:>10.2%}"
        f"  <- mean per-node (qps[n] + ded[n]) / cap"
    )
    log.info(
        f"  {'  └─ maintenance overhead':<44}: {maint_fraction:>10.2%}"
        f"  <- total_maintenance / total_capacity"
    )

    # ── Runtime ──────────────────────────────────────────────────────────────
    log.info("=" * 66)
    log.info("  ALGORITHM RUNTIME")
    log.info("=" * 66)
    if elapsed_s < 1.0:
        log.info(f"  Total wall-clock time : {elapsed_s * 1000:.3f} ms")
    else:
        log.info(f"  Total wall-clock time : {elapsed_s:.4f} s")
    log.info("=" * 66)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ANN Index Maintenance Scheduler (greedy + FG flow)"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"Path to cluster config JSON (default: {DEFAULT_CONFIG})"
    )
    parser.add_argument(
        "--k", type=int, default=1,
        help="Max replicas per replica set allowed in one batch (default: 1)"
    )
    parser.add_argument(
        "--max-skips", type=int, default=DEFAULT_MAX_SKIPS,
        dest="max_skips",
        help=(
            f"Terminate greedy loop early after this many consecutive "
            f"capacity-failure skips (default: {DEFAULT_MAX_SKIPS}). "
            f"k-capped rejections do not count."
        )
    )
    parser.add_argument(
        "--log", default=DEFAULT_LOG, metavar="PATH",
        help=f"Path to log file (default: {DEFAULT_LOG})"
    )
    args = parser.parse_args()

    setup_logging(args.log)

    log.info("=" * 66)
    log.info("ANN Index Maintenance Scheduler  starting")
    log.info(f"  config     : {args.config}")
    log.info(f"  k          : {args.k}")
    log.info(f"  max-skips  : {args.max_skips}")
    log.info(f"  log        : {args.log}")
    log.info("=" * 66)

    config = load_config(args.config)
    meta   = config["meta"]

    # ── Start timer ───────────────────────────────────────────────────────────
    t_start = time.perf_counter()

    # ── Stage 1: greedy heuristic ─────────────────────────────────────────────
    log.info(f"[1/2] Running greedy heuristic (k={args.k}, max_skips={args.max_skips}) ...")
    greedy_result = run_scheduler(config, k=args.k,
                                  max_consecutive_skips=args.max_skips)
    log.info(f"      Greedy candidate batch : {len(greedy_result['scheduled'])} job(s)")
    if greedy_result["early_exit"]:
        log.warning(
            f"      Early exit triggered after {args.max_skips} "
            f"consecutive capacity failures."
        )

    # ── Stage 2: foreground flow ──────────────────────────────────────────────
    log.info("[2/2] Building foreground graph and running flow ...")
    fg_caps = compute_fg_caps(config["nodes"], meta["node_capacity"])

    feasible, _fg_graph, _fg_flow = run_foreground_flow(
        batch   = greedy_result["scheduled"],
        shards  = config["shards"],
        fg_caps = fg_caps,
    )

    # ── Stop timer ────────────────────────────────────────────────────────────
    t_end     = time.perf_counter()
    elapsed_s = t_end - t_start

    status = "FEASIBLE" if feasible else "INFEASIBLE"
    log.info(f"      Result : {status}")

    # ── Write full report to log ──────────────────────────────────────────────
    log_results(
        greedy_result         = greedy_result,
        fg_feasible           = feasible,
        fg_caps               = fg_caps,
        meta                  = meta,
        elapsed_s             = elapsed_s,
        k                     = args.k,
        max_consecutive_skips = args.max_skips,
    )


if __name__ == "__main__":
    main()

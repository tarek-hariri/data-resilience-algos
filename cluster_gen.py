#!/usr/bin/env python3
"""
cluster_gen.py

Two-phase generation for large-scale clusters:

  Phase 1 — Greedy placement  (no solver)
      Shards sorted by QPS descending. Each shard's k replicas are
      assigned to the k nodes with the lowest accumulated QPS weight.
      O(S × k × log N) — scales to thousands of shards and nodes.

  Phase 2 — Routing LP  (Gurobi, continuous variables only)
      With placement fixed, routing fractions r[s,n] are optimised.
      No binary variables. At 2500 shards × k=3 replicas this is
      ~7500 routing variables — solves in milliseconds.

Maintenance costs
  Each replica of a logical shard has its own independently sampled
  reindex cost. maintenance_cost in the JSON is a dict keyed by node ID
  (matching the replicas list), not a single scalar.

  COST_MU is derived so the expected total cost across all replicas is
  approximately BACKGROUND_OVERLOAD_FACTOR × background_capacity:

    COST_MU = log(overload × headroom / (NUM_SHARDS × k))

Output format:
  shards[s].maintenance_cost  →  {"node_id": cost, ...}  (one entry per replica)

Active constraints (routing LP only):
    C4  all traffic for each shard must be served
    C5  node load definition
    C6  peak load linearization (min-max objective)
    C7  load floor — prevents near-idle nodes
    C9  node capacity ceiling

Dependencies:
    pip install gurobipy numpy
    (requires a valid Gurobi licence — free academic licences at gurobi.com)
"""

import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict


# ── PARAMETERS ────────────────────────────────────────────────────────────────
NUM_NODES          = 1000
NUM_SHARDS         = 50000
REPLICATION_FACTOR = 3      # k replicas per logical shard

# Saturation & capacity
NODE_CAPACITY      = 1_500  # QPS ceiling per node
TARGET_SATURATION  = 0.80   # foreground serving load (0.0 – 1.0)

# ── QPS mixture model parameters ─────────────────────────────────────────────
HOT_SHARD_FRACTION  = 0.05   # fraction of shards drawn from the hot population
HOT_MULTIPLIER      = 8.0    # hot shard median is HOT_MULTIPLIER × cold median
COLD_SIGMA          = 0.5    # log-normal spread for cold shards
HOT_SIGMA           = 0.5    # log-normal spread for hot shards
QPS_CAP_MULTIPLIER  = 15.0   # no shard may exceed this × the per-shard mean
# ─────────────────────────────────────────────────────────────────────────────

DELTA              = 0.05   # C7 slack: load floor = (1 - delta) * L_bar
RANDOM_SEED        = 42
OUTPUT_FILE        = "cluster_config.json"
SOLVER_TIME_LIMIT  = 120    # seconds

# ── Maintenance cost parameters ───────────────────────────────────────────────
# Each replica is sampled independently. COST_MU is derived so that the
# expected sum across all NUM_SHARDS × k replicas equals:
#
#   BACKGROUND_OVERLOAD_FACTOR × background_capacity
#
# where background_capacity = (1 - TARGET_SATURATION) × NUM_NODES × NODE_CAPACITY
#
# Dividing by (NUM_SHARDS × REPLICATION_FACTOR) rather than just NUM_SHARDS
# keeps the total budget anchor correct now that there are k costs per shard.
BACKGROUND_OVERLOAD_FACTOR = 1.5
COST_SIGMA                 = 0.8

# Derived — do not set manually
COST_MU = np.log(
    BACKGROUND_OVERLOAD_FACTOR
    * (1 - TARGET_SATURATION)
    * NODE_CAPACITY * NUM_NODES
    / (NUM_SHARDS * REPLICATION_FACTOR)
)
# ─────────────────────────────────────────────────────────────────────────────


# ── JSON serializer ───────────────────────────────────────────────────────────
def convert_np(obj):
    """Convert numpy scalar/array types to native Python for json.dump."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. SAMPLE QPS (mixture model) ─────────────────────────────────────────────
def sample_qps(
    num_shards:     int,
    total_qps:      float,
    hot_fraction:   float,
    hot_multiplier: float,
    cold_sigma:     float,
    hot_sigma:      float,
    cap_multiplier: float,
) -> tuple[np.ndarray, set]:
    """
    Sample per-shard QPS from a two-component log-normal mixture.

    n_hot  shards drawn from LogNormal(log(hot_multiplier), hot_sigma)
    n_cold shards drawn from LogNormal(0,                   cold_sigma)

    Steps:
      1. Normalize assembled values to total_qps.
      2. Clip any shard exceeding cap_multiplier × mean_qps.
      3. Renormalize to restore the total.

    Returns
    -------
    qps        : ndarray of per-shard QPS values summing to total_qps
    hot_shards : set of shard indices drawn from the hot distribution
    """
    n_hot  = max(1, round(hot_fraction * num_shards))
    n_cold = num_shards - n_hot

    perm     = np.random.permutation(num_shards)
    hot_idx  = set(int(i) for i in perm[:n_hot])
    cold_idx = [int(i) for i in perm[n_hot:]]
    hot_list = [int(i) for i in perm[:n_hot]]

    cold_raw = np.random.lognormal(mean=0.0,                    sigma=cold_sigma, size=n_cold)
    hot_raw  = np.random.lognormal(mean=np.log(hot_multiplier), sigma=hot_sigma,  size=n_hot)

    qps = np.empty(num_shards)
    for i, idx in enumerate(cold_idx):
        qps[idx] = cold_raw[i]
    for i, idx in enumerate(hot_list):
        qps[idx] = hot_raw[i]

    qps = qps / qps.sum() * total_qps
    mean_qps = total_qps / num_shards
    qps = np.clip(qps, 0.0, cap_multiplier * mean_qps)
    qps = qps / qps.sum() * total_qps

    return qps, hot_idx


# ── 2. SAMPLE MAINTENANCE COSTS (per replica) ─────────────────────────────────
def sample_maintenance_costs(
    num_shards: int,
    k:          int,
    mu:         float,
    sigma:      float,
) -> np.ndarray:
    """
    Draw per-replica reindex costs from a log-normal distribution.

    Returns a 2D array of shape (num_shards, k). Each row is one logical
    shard; each column is one replica. Costs are independent across both
    shards and replicas — no two replicas are constrained to be equal.

    With COST_MU derived from cluster capacity parameters, the expected
    sum across all num_shards × k values is approximately:

        BACKGROUND_OVERLOAD_FACTOR × background_capacity
    """
    return np.random.lognormal(mean=mu, sigma=sigma, size=(num_shards, k))


def assign_replica_costs(
    costs_raw:  np.ndarray,
    placement:  dict,
    num_shards: int,
) -> dict:
    """
    Associate the (num_shards, k) cost array with actual node IDs from placement.

    Returns
    -------
    replica_costs : shard_id -> {node_id (int): cost (float)}
    """
    replica_costs = {}
    for s in range(num_shards):
        replica_costs[s] = {
            n: float(costs_raw[s, i])
            for i, n in enumerate(placement[s])
        }
    return replica_costs


# ── 3. PHASE 1 — GREEDY PLACEMENT ────────────────────────────────────────────
def greedy_placement(
    qps:        np.ndarray,
    num_shards: int,
    num_nodes:  int,
    k:          int,
) -> tuple[dict, dict]:
    """
    Assign k replicas per shard using a greedy load-balancing heuristic.

    Sort shards by QPS descending so hot shards get first pick of
    underloaded nodes. For each shard, choose the k nodes with the
    lowest accumulated QPS weight.

    Complexity: O(S × k × log N)

    Returns
    -------
    placement   : shard_id -> [node_id, ...]  (length k, native Python ints)
    node_shards : node_id  -> [shard_id, ...]
    """
    node_weight = np.zeros(num_nodes)
    placement   = {}
    node_shards = defaultdict(list)

    order = np.argsort(qps)[::-1]

    for s_np in order:
        s      = int(s_np)
        chosen = [int(n) for n in np.argsort(node_weight)[:k]]
        placement[s] = chosen
        for n in chosen:
            node_shards[n].append(s)
            node_weight[n] += qps[s] / k

    return placement, node_shards


# ── 4. PHASE 2 — ROUTING LP (Gurobi) ─────────────────────────────────────────
def solve_routing_lp(
    qps:           np.ndarray,
    placement:     dict,
    node_shards:   dict,
    num_nodes:     int,
    num_shards:    int,
    node_capacity: float,
    delta:         float,
) -> tuple:
    """
    Optimise routing fractions given a fixed placement.

    Variables  (all continuous — no binary variables)
    ---------
    r[s,n]  ∈ [0,1]  — fraction of shard s traffic routed to node n
                        only created for valid (s,n) replica pairs
    L[n]    ≥ 0      — total QPS load on node n
    L_max   ≥ 0      — peak load (minimised)

    Constraints
    -----------
    C4  all traffic for each shard must be served
    C5  node load = sum of routed QPS
    C6  L_max tracks the busiest node
    C7  load floor = (1 - delta) * L_bar
    C9  node capacity ceiling
    """
    L_bar = float(qps.sum()) / num_nodes

    model = gp.Model("RoutingLP")
    model.Params.TimeLimit  = SOLVER_TIME_LIMIT
    model.Params.OutputFlag = 1

    valid_pairs = [(s, n) for s, nodes in placement.items() for n in nodes]

    r     = model.addVars(valid_pairs, lb=0.0, ub=1.0, name="r")
    L     = model.addVars(num_nodes,   lb=0.0,          name="L")
    L_max = model.addVar(              lb=0.0,          name="L_max")

    model.setObjective(L_max, GRB.MINIMIZE)

    model.addConstrs(
        (gp.quicksum(r[s, n] for n in placement[s]) == 1
         for s in range(num_shards)),
        name="C4"
    )
    model.addConstrs(
        (L[n] == gp.quicksum(qps[s] * r[s, n] for s in node_shards[n])
         for n in range(num_nodes)),
        name="C5"
    )
    model.addConstrs(
        (L_max >= L[n] for n in range(num_nodes)),
        name="C6"
    )
    model.addConstrs(
        (L[n] >= (1 - delta) * L_bar for n in range(num_nodes)),
        name="C7"
    )
    model.addConstrs(
        (L[n] <= node_capacity for n in range(num_nodes)),
        name="C9"
    )

    model.optimize()

    return model, r, L, L_max, L_bar


# ── 5. EXTRACT RESULTS ────────────────────────────────────────────────────────
def extract_results(
    model, r, L, L_max,
    placement:   dict,
    node_shards: dict,
    num_nodes:   int,
    num_shards:  int,
) -> tuple:
    """Extract routing decisions from the solved LP. Requires model.SolCount > 0."""
    routing = {}
    for s in range(num_shards):
        routing[s] = {}
        for n in placement[s]:
            val = r[s, n].X
            if val > 1e-6:
                routing[s][n] = round(float(val), 6)

    node_loads = {n: round(float(L[n].X), 4) for n in range(num_nodes)}

    return routing, node_loads


# ── 6. BUILD CONFIG DICT ──────────────────────────────────────────────────────
def build_config(
    qps: np.ndarray,
    replica_costs: dict,
    placement, node_shards, routing, node_loads,
    hot_shards: set,
    num_nodes: int,
    num_shards: int,
    k: int,
    node_capacity: float,
    target_saturation: float,
    L_bar: float,
    L_max_val: float,
    solver_status: str,
) -> dict:
    total_qps         = float(qps.sum())
    actual_saturation = round(L_max_val / node_capacity, 6)
    background_cap    = (1 - target_saturation) * num_nodes * node_capacity
    total_maint_cost  = sum(
        cost
        for s_costs in replica_costs.values()
        for cost in s_costs.values()
    )

    return {
        "meta": {
            "num_nodes":                  num_nodes,
            "num_shards":                 num_shards,
            "replication_factor":         k,
            "total_qps":                  round(total_qps, 4),
            "node_capacity":              node_capacity,
            "target_saturation":          target_saturation,
            "actual_saturation":          actual_saturation,
            "solver_status":              solver_status,
            "L_bar":                      round(L_bar, 4),
            "L_max_achieved":             round(L_max_val, 4),
            "load_imbalance_ratio":       round(L_max_val / L_bar, 4),
            "background_capacity":        round(background_cap, 4),
            "background_overload_factor": BACKGROUND_OVERLOAD_FACTOR,
            "total_maintenance_cost":     round(total_maint_cost, 4),
        },
        "shards": {
            str(s): {
                "qps":              round(float(qps[s]), 4),
                "is_hot":           s in hot_shards,
                # maintenance_cost is now a dict: node_id (str) -> cost (float)
                # one independent cost per replica, not a single shared scalar
                "maintenance_cost": {
                    str(n): round(c, 4)
                    for n, c in replica_costs[s].items()
                },
                "replicas":         placement[s],
                "routing":          {str(n): v for n, v in routing[s].items()},
            }
            for s in range(num_shards)
        },
        "nodes": {
            str(n): {
                "shards_held":     node_shards[n],
                "hot_shards_held": [s for s in node_shards[n] if s in hot_shards],
                "total_load_qps":  node_loads[n],
                "load_fraction":   round(node_loads[n] / total_qps, 6),
                "utilisation":     round(node_loads[n] / node_capacity, 6),
            }
            for n in range(num_nodes)
        },
    }


# ── 7. MAIN ───────────────────────────────────────────────────────────────────
def main():
    assert 0.0 < TARGET_SATURATION < 1.0, (
        "TARGET_SATURATION must be in (0, 1) — "
        "saturation of 1.0 leaves no background headroom."
    )

    total_qps      = TARGET_SATURATION * NUM_NODES * NODE_CAPACITY
    background_cap = (1 - TARGET_SATURATION) * NUM_NODES * NODE_CAPACITY
    expected_maint = BACKGROUND_OVERLOAD_FACTOR * background_cap
    mean_qps       = total_qps / NUM_SHARDS

    print(f"Saturation target       : {TARGET_SATURATION * 100:.0f}%")
    print(f"Node capacity           : {NODE_CAPACITY} QPS/node")
    print(f"Derived total QPS       : {total_qps:.0f}")
    print(f"Mean QPS per shard      : {mean_qps:.1f}")
    print(f"QPS cap per shard       : {QPS_CAP_MULTIPLIER * mean_qps:.1f}  "
          f"({QPS_CAP_MULTIPLIER:.0f}× mean)")
    print(f"Background headroom     : {background_cap:.0f} QPS-equiv")
    print(f"Expected maint. work    : {expected_maint:.0f} QPS-equiv  "
          f"({BACKGROUND_OVERLOAD_FACTOR}× headroom)")
    print(f"Derived COST_MU         : {COST_MU:.4f}  "
          f"(log-normal median/replica ≈ {np.exp(COST_MU):.2f})")

    # ── Sample QPS ──────────────────────────────────────────────────────────
    print("\nSampling QPS distribution (mixture model)...")
    np.random.seed(RANDOM_SEED)
    qps, hot_shards = sample_qps(
        NUM_SHARDS, total_qps,
        HOT_SHARD_FRACTION, HOT_MULTIPLIER,
        COLD_SIGMA, HOT_SIGMA,
        QPS_CAP_MULTIPLIER,
    )
    n_hot    = len(hot_shards)
    n_cold   = NUM_SHARDS - n_hot
    hot_qps  = qps[list(hot_shards)]
    cold_qps = qps[[s for s in range(NUM_SHARDS) if s not in hot_shards]]
    print(f"  hot shards   : {n_hot}  "
          f"(QPS range [{hot_qps.min():.1f}, {hot_qps.max():.1f}], "
          f"mean {hot_qps.mean():.1f})")
    print(f"  cold shards  : {n_cold}  "
          f"(QPS range [{cold_qps.min():.1f}, {cold_qps.max():.1f}], "
          f"mean {cold_qps.mean():.1f})")
    print(f"  max/mean ratio: {qps.max() / mean_qps:.1f}×")

    # ── Phase 1: Greedy placement ────────────────────────────────────────────
    print("\nPhase 1 — greedy placement...")
    placement, node_shards = greedy_placement(
        qps, NUM_SHARDS, NUM_NODES, REPLICATION_FACTOR
    )
    shard_counts = [len(node_shards[n]) for n in range(NUM_NODES)]
    print(f"  shards/node range: [{min(shard_counts)}, {max(shard_counts)}]  "
          f"(mean {sum(shard_counts) / NUM_NODES:.1f})")

    # ── Sample maintenance costs (per replica, after placement) ──────────────
    print("\nSampling maintenance costs (per-replica, log-normal)...")
    costs_raw     = sample_maintenance_costs(
        NUM_SHARDS, REPLICATION_FACTOR, COST_MU, COST_SIGMA
    )
    replica_costs = assign_replica_costs(costs_raw, placement, NUM_SHARDS)
    total_maint   = costs_raw.sum()
    admissible    = background_cap / total_maint
    print(f"  cost range (per replica) = [{costs_raw.min():.2f}, {costs_raw.max():.2f}]")
    print(f"  cost mean / median       = {costs_raw.mean():.2f} / {np.median(costs_raw):.2f}")
    print(f"  total cost (all replicas)= {total_maint:.2f}  "
          f"(expected ≈ {expected_maint:.2f})")
    print(f"  ~admissible fraction     = {admissible * 100:.1f}%")

    # ── Phase 2: Routing LP ──────────────────────────────────────────────────
    print("\nPhase 2 — routing LP (Gurobi)...")
    model, r, L, L_max, L_bar = solve_routing_lp(
        qps, placement, node_shards,
        NUM_NODES, NUM_SHARDS,
        NODE_CAPACITY, DELTA
    )

    status_map = {
        GRB.OPTIMAL:     "Optimal",
        GRB.INFEASIBLE:  "Infeasible",
        GRB.INF_OR_UNBD: "InfOrUnbounded",
        GRB.TIME_LIMIT:  "TimeLimit",
        GRB.SUBOPTIMAL:  "Suboptimal",
    }
    solver_status = status_map.get(model.Status, f"GurobiStatus_{model.Status}")

    if model.SolCount == 0:
        print(f"\n  ✗ Routing LP found no solution (status: {solver_status}).")
        print("    Try increasing DELTA or SOLVER_TIME_LIMIT.")
        return

    L_max_val  = float(L_max.X)
    actual_sat = L_max_val / NODE_CAPACITY

    print(f"\n  LP status          : {solver_status}")
    print(f"  L_bar              = {L_bar:.2f} QPS  (ideal per-node load)")
    print(f"  L_max              = {L_max_val:.2f} QPS  (actual peak)")
    print(f"  Imbalance          = {L_max_val / L_bar:.3f}×  (1.0 = perfect)")
    print(f"  Target saturation  = {TARGET_SATURATION * 100:.0f}%")
    print(f"  Actual saturation  = {actual_sat * 100:.1f}%")

    # ── Extract ─────────────────────────────────────────────────────────────
    print("\nExtracting routing...")
    routing, node_loads = extract_results(
        model, r, L, L_max,
        placement, node_shards,
        NUM_NODES, NUM_SHARDS,
    )

    # ── Print node summary (sample first 10) ─────────────────────────────────
    print(f"\n  {'node':<12} {'load (QPS)':>10}  {'util':>6}  {'shards':>6}  {'hot':>4}")
    print("  " + "─" * 54)
    for n in range(min(10, NUM_NODES)):
        hot_n = len([s for s in node_shards[n] if s in hot_shards])
        util  = node_loads[n] / NODE_CAPACITY
        bar   = "█" * int(util * 20)
        print(f"  node_{n:04d}   {node_loads[n]:9.2f}   {util:5.1%}  "
              f"{len(node_shards[n]):6d}  {hot_n:4d}  {bar}")
    if NUM_NODES > 10:
        print(f"  ... ({NUM_NODES - 10} more nodes)")

    # ── Write config ─────────────────────────────────────────────────────────
    config = build_config(
        qps, replica_costs,
        placement, node_shards, routing, node_loads, hot_shards,
        NUM_NODES, NUM_SHARDS, REPLICATION_FACTOR,
        NODE_CAPACITY, TARGET_SATURATION,
        L_bar, L_max_val, solver_status
    )
    with open(OUTPUT_FILE, "w") as f:
        json.dump(config, f, indent=2, default=convert_np)
    print(f"\n  ✓ Config written → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

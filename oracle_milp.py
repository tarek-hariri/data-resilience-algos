import json
import math
import time
from gurobipy import Model, GRB, quicksum

# ── Reward function ────────────────────────────────────────────────────────────
def reindex_value(shard_id: int, server_id: int,
                  maintenance_cost: float, is_hot: bool) -> float:
    h    = math.sin(shard_id * 2654.435769 + server_id * 1234.567) * 0.5 + 0.5
    base = maintenance_cost * (0.8 + h)
    return base * (1.5 if is_hot else 1.0)


# ── Load cluster config ────────────────────────────────────────────────────────
with open("cluster_config.json") as fh:
    cluster = json.load(fh)

meta        = cluster["meta"]
shards_data = cluster["shards"]
nodes_data  = cluster["nodes"]

k             = 1
node_capacity = meta["node_capacity"]


# ── Sets ───────────────────────────────────────────────────────────────────────
S = [int(s) for s in shards_data]
V = [int(v) for v in nodes_data]

P = [
    (int(sid), int(srv))
    for sid, shard in shards_data.items()
    for srv in shard["replicas"]
]

P_i = {i: [j for (ii, j) in P if ii == i] for i in S}
P_j = {j: [i for (i, jj) in P if jj == j] for j in V}


# ── Parameters ─────────────────────────────────────────────────────────────────
D = {int(sid): shard["qps"] for sid, shard in shards_data.items()}

r = {
    (int(sid), int(srv)): cost
    for sid, shard in shards_data.items()
    for srv, cost in shard["maintenance_cost"].items()
}

w = {
    (int(sid), int(srv)): reindex_value(
        shard_id=int(sid),
        server_id=int(srv),
        maintenance_cost=cost,
        is_hot=shards_data[sid]["is_hot"],
    )
    for sid, shard in shards_data.items()
    for srv, cost in shard["maintenance_cost"].items()
}

C = {j: node_capacity for j in V}


# ── Build MILP ─────────────────────────────────────────────────────────────────
start_time = time.time()

m = Model("reindex_scheduling")
m.setParam("OutputFlag", 0)
m.setParam("MIPGap", 1e-4)

f = {
    (i, j): m.addVar(lb=0.0, ub=D[i], vtype=GRB.CONTINUOUS, name=f"f_{i}_{j}")
    for (i, j) in P
}
x = {
    (i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    for (i, j) in P
}

m.update()

m.setObjective(
    quicksum(w[i, j] * x[i, j] for (i, j) in P),
    GRB.MAXIMIZE,
)

# C1 — Foreground demand equality
for i in S:
    m.addConstr(
        quicksum(f[i, j] for j in P_i[i]) == D[i],
        name=f"demand_eq_{i}",
    )

# C3 — Server resource budget (coupling constraint)
for j in V:
    m.addConstr(
        quicksum(f[i, j] for i in P_j[j])
        + quicksum(r[i, j] * x[i, j] for i in P_j[j])
        <= C[j],
        name=f"server_budget_{j}",
    )

# C4 — Per-shard reindex parallelism limit
for i in S:
    m.addConstr(
        quicksum(x[i, j] for j in P_i[i]) <= k,
        name=f"reindex_limit_{i}",
    )

m.optimize()

elapsed_ms = (time.time() - start_time) * 1000.0


# ── Write results to oracle_log.txt ────────────────────────────────────────────
with open("oracle_log.txt", "w", encoding="utf-8") as log:

    def write(line=""):
        log.write(line + "\n")

    SEP  = "=" * 68
    SEP2 = "-" * 68

    write(SEP)
    write("  ORACLE LOG -- Joint Load Steering & Reindex Scheduling")
    write(f"  Cluster : {meta['num_nodes']} nodes  |  "
          f"{meta['num_shards']} shards  |  "
          f"replication factor {meta['replication_factor']}")
    write(f"  k       : {k}  (max concurrent reindex jobs per logical shard)")
    write(SEP)

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        write(f"\n  Solver returned status {m.Status} -- no feasible solution found.")
        write(f"  Total execution time : {elapsed_ms:.2f} ms")
    else:
        status_str = "OPTIMAL" if m.Status == GRB.OPTIMAL else "SUBOPTIMAL"

        write(f"\n  Solver status        : {status_str}")
        write(f"  Total reward         : {m.ObjVal:.4f}")
        write(f"  MIP gap              : {m.MIPGap * 100:.4f} %")
        write(f"  Total execution time : {elapsed_ms:.2f} ms")
        write()

        # ── Per-server utilisation data ────────────────────────────────────────
        server_stats = {}
        for j in sorted(V):
            fg  = sum(f[i, j].X for i in P_j[j])
            bg  = sum(r[i, j] * x[i, j].X for i in P_j[j])
            tot = fg + bg
            util_pct = 100.0 * tot / C[j]
            server_stats[j] = {
                "fg": fg, "bg": bg, "total": tot,
                "capacity": C[j], "util_pct": util_pct,
            }

        # ── Cluster-wide utilisation summary ───────────────────────────────────
        write(SEP2)
        write("  CLUSTER UTILISATION SUMMARY")
        write(SEP2)

        all_utils  = [s["util_pct"] for s in server_stats.values()]
        all_totals = [s["total"]    for s in server_stats.values()]
        all_fg     = [s["fg"]       for s in server_stats.values()]
        all_bg     = [s["bg"]       for s in server_stats.values()]
        total_cap  = sum(C[j] for j in V)

        avg_util     = sum(all_utils) / len(all_utils)
        min_util     = min(all_utils)
        max_util     = max(all_utils)
        min_srv      = min(server_stats, key=lambda j: server_stats[j]["util_pct"])
        max_srv      = max(server_stats, key=lambda j: server_stats[j]["util_pct"])
        cluster_util = 100.0 * sum(all_totals) / total_cap

        total_fg_cluster = sum(all_fg)
        total_bg_cluster = sum(all_bg)
        used_total       = sum(all_totals)
        fg_share = 100.0 * total_fg_cluster / total_cap
        bg_share = 100.0 * total_bg_cluster / total_cap
        util_std = (sum((u - avg_util) ** 2 for u in all_utils) / len(all_utils)) ** 0.5

        write(f"  Overall cluster utilisation   : {cluster_util:6.2f} %")
        write(f"  Average server utilisation    : {avg_util:6.2f} %")
        write(f"  Minimum server utilisation    : {min_util:6.2f} %  (Server {min_srv})")
        write(f"  Maximum server utilisation    : {max_util:6.2f} %  (Server {max_srv})")
        write(f"  Utilisation std deviation     : {util_std:6.2f} %")
        write()
        write(f"  Total cluster capacity        : {total_cap:10.2f} RU")
        write(f"  Total foreground (FG) flow    : {total_fg_cluster:10.2f} QPS"
              f"  ({fg_share:.1f} % of total cluster capacity)")
        write(f"  Total background (BG) flow    : {total_bg_cluster:10.2f} RU "
              f"  ({bg_share:.1f} % of total cluster capacity)")
        write(f"  Total used capacity           : {used_total:10.2f} RU")
        write(f"  Total spare capacity          : {total_cap - used_total:10.2f} RU")
        write()

        # ── Per-server utilisation breakdown ───────────────────────────────────
        write(SEP2)
        write("  PER-SERVER UTILISATION BREAKDOWN")
        write(SEP2)
        write(f"  {'Server':<10} {'FG (QPS)':>12} {'BG (RU)':>12} "
              f"{'Total (RU)':>12} {'Capacity':>10} {'Util %':>8}")
        write(f"  {'-'*8:<10} {'-'*10:>12} {'-'*10:>12} "
              f"{'-'*10:>12} {'-'*8:>10} {'-'*6:>8}")
        for j in sorted(V):
            s = server_stats[j]
            write(f"  {j:<10} {s['fg']:>12.2f} {s['bg']:>12.2f} "
                  f"{s['total']:>12.2f} {s['capacity']:>10.0f} {s['util_pct']:>7.2f}%")
        write()

        # ── Reindex schedule ───────────────────────────────────────────────────
        write(SEP2)
        write("  REINDEX SCHEDULE")
        write(SEP2)
        scheduled = sorted([(i, j) for (i, j) in P if x[i, j].X > 0.5])
        if scheduled:
            total_reward = sum(w[i, j] for (i, j) in scheduled)
            write(f"  {len(scheduled)} job(s) scheduled  |  "
                  f"total reward = {total_reward:.4f}\n")
            write(f"  {'Shard':<8} {'Server':<10} {'Cost (RU)':>12} "
                  f"{'Reward':>12} {'Hot?':>6}")
            write(f"  {'-'*6:<8} {'-'*8:<10} {'-'*10:>12} "
                  f"{'-'*10:>12} {'-'*4:>6}")
            for (i, j) in scheduled:
                hot_tag = "yes" if shards_data[str(i)]["is_hot"] else "no"
                write(f"  {i:<8} {j:<10} {r[i,j]:>12.4f} "
                      f"{w[i,j]:>12.4f} {hot_tag:>6}")
        else:
            write("  No reindex jobs selected -- insufficient server headroom.")
        write()

        # ── Rejected candidates ────────────────────────────────────────────────
        write(SEP2)
        write("  REJECTED / INACTIVE REINDEX CANDIDATES")
        write(SEP2)
        rejected = sorted([(i, j) for (i, j) in P if x[i, j].X <= 0.5])
        write(f"  {'Shard':<8} {'Server':<10} {'Cost (RU)':>12} "
              f"{'Reward':>12} {'Hot?':>6}")
        write(f"  {'-'*6:<8} {'-'*8:<10} {'-'*10:>12} "
              f"{'-'*10:>12} {'-'*4:>6}")
        for (i, j) in rejected:
            hot_tag = "yes" if shards_data[str(i)]["is_hot"] else "no"
            write(f"  {i:<8} {j:<10} {r[i,j]:>12.4f} "
                  f"{w[i,j]:>12.4f} {hot_tag:>6}")
        write()

        # ── Load steering rules ────────────────────────────────────────────────
        write(SEP2)
        write("  LOAD STEERING RULES  (original routing -> optimised routing)")
        write(SEP2)
        for i in sorted(S):
            orig_routing = shards_data[str(i)]["routing"]
            demand       = D[i]
            write(f"  Shard {i}  (demand = {demand:.2f} QPS"
                  f"{'  HOT' if shards_data[str(i)]['is_hot'] else ''})")
            for j in sorted(P_i[i]):
                orig_frac = float(orig_routing.get(str(j), 0.0))
                new_flow  = f[i, j].X
                new_frac  = new_flow / demand if demand > 0 else 0.0
                delta_pct = (new_frac - orig_frac) * 100.0
                if delta_pct >  0.01:
                    tag = "  ^ steered toward"
                elif delta_pct < -0.01:
                    tag = "  v steered away"
                else:
                    tag = "  - unchanged"
                write(f"    Server {j}:  {orig_frac*100:6.2f}%  ->  {new_frac*100:6.2f}%"
                      f"  ({new_flow:8.2f} QPS){tag}")
            write()

        write(SEP)
        write("  END OF LOG")
        write(SEP)

print("Done -- results written to oracle_log.txt")
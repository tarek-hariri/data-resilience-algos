#!/usr/bin/env python3
"""
ann_maintenance_scheduler.py

Two-graph network-simplex index maintenance scheduler for ANN query systems.

Algorithm
---------
1.  Parse cluster_config.json.
2.  Derive per-server FG capacity and BG budget directly from meta.node_capacity.
        fg_cap[n]    = node_capacity          (physical ceiling; uniform across nodes)
        bg_budget[n] = node_capacity × alpha  (max resource BG may spend on server n)
3.  Build and solve the background min-cost flow graph to maximise reindex reward.
4.  Extract fully-saturated shard→server edges as candidate jobs.
5.  Post-process: keep only the highest-value job per logical shard (fault tolerance).
6.  Iteratively test foreground feasibility; drop the lowest-value job on each
    infeasible iteration until the flow is solvable (or the batch empties).
7.  Print the committed batch and a run summary.

Visualization (--visualize 1)
------------------------------
  Opens a single window with clickable tabs, one per pipeline stage:
    · BG: Before       — background graph topology with capacities + weights
    · BG: After        — same graph with flow values; saturated edges highlighted
    · Post-Process     — table of KEEP / DROP decisions per candidate job
    · FG: Initial      — foreground graph with full-batch deductions, pre-solve
    · FG: Iter N       — each infeasible iteration; evicted edge shown in red
    · FG: ✅ Done      — final committed state; active job edges in green

Usage
-----
  python ann_maintenance_scheduler.py cluster_config.json
  python ann_maintenance_scheduler.py cluster_config.json --alpha 0.20
  python ann_maintenance_scheduler.py cluster_config.json --alpha 0.20 --visualize 1
"""

import argparse
import json
import math
import sys
from pathlib import Path

import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
# Tunable defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ALPHA = 0.30

FLOW_SCALE   = 1_000   # QPS / maintenance-cost  →  integer flow units
WEIGHT_SCALE = 1_000   # reindex value            →  integer edge weight


# ─────────────────────────────────────────────────────────────────────────────
# Scaling helpers
# ─────────────────────────────────────────────────────────────────────────────
def to_flow(qps: float) -> int:
    return max(1, round(qps * FLOW_SCALE))


def to_weight(value: float) -> int:
    """Negative so that min-cost flow ≡ max-reward."""
    return -max(1, round(value * WEIGHT_SCALE))


# ─────────────────────────────────────────────────────────────────────────────
# Blackbox: reindex reward for (shard, server)
# ─────────────────────────────────────────────────────────────────────────────
def reindex_value(shard_id: int, server_id: int,
                  maintenance_cost: float, is_hot: bool) -> float:
    """
    Placeholder reward signal — replace with real staleness / priority logic.
    Deterministic and hash-derived for reproducible results.
    Contract: always returns a strictly positive float.
    """
    h    = math.sin(shard_id * 2654.435769 + server_id * 1234.567) * 0.5 + 0.5
    base = maintenance_cost * (0.8 + h)
    return base * (1.5 if is_hot else 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Budget derivation
# ─────────────────────────────────────────────────────────────────────────────
def compute_budgets(nodes: dict, node_capacity: float,
                    alpha: float) -> tuple[dict[int, int], dict[int, int]]:
    """
    fg_caps[n]    = to_flow(node_capacity)          — physical ceiling
    bg_budgets[n] = to_flow(node_capacity × alpha)  — BG spend limit per epoch
    Both uniform across nodes (node_capacity is a cluster-wide scalar).
    """
    fg_caps:    dict[int, int] = {}
    bg_budgets: dict[int, int] = {}
    for nid_str in nodes:
        nid = int(nid_str)
        fg_caps[nid]    = to_flow(node_capacity)
        bg_budgets[nid] = to_flow(node_capacity * alpha)
    return fg_caps, bg_budgets


# ─────────────────────────────────────────────────────────────────────────────
# Background graph — build
# ─────────────────────────────────────────────────────────────────────────────
def build_background_graph(shards: dict,
                            bg_budgets: dict[int, int]
                            ) -> tuple[nx.DiGraph, dict[tuple[int, int], int]]:
    """
    Four-layer DAG:  SRC → [shards] → [servers] → SNK
    Slack SRC→SNK lets unused budget drain at zero cost (always feasible).
    Shard→server weight = -scaled_value  (min-cost ≡ max-reward).
    """
    G        = nx.DiGraph()
    total_bg = sum(bg_budgets.values())

    G.add_node("SRC", demand=-total_bg)
    G.add_node("SNK", demand= total_bg)
    G.add_edge("SRC", "SNK", capacity=total_bg, weight=0)

    job_caps: dict[tuple[int, int], int] = {}

    for sid_str, sdata in shards.items():
        sid        = int(sid_str)
        maint_cost = sdata["maintenance_cost"]
        is_hot     = sdata["is_hot"]
        replicas   = sdata["replicas"]
        shard_node = f"shard_{sid}"
        job_cap    = to_flow(maint_cost)

        G.add_node(shard_node, demand=0)
        G.add_edge("SRC", shard_node,
                   capacity=job_cap * len(replicas), weight=0)

        for server_id in replicas:
            server_node = f"server_{server_id}"
            if server_node not in G:
                G.add_node(server_node, demand=0)
                G.add_edge(server_node, "SNK",
                           capacity=bg_budgets.get(server_id, 0), weight=0)
            val = reindex_value(sid, server_id, maint_cost, is_hot)
            G.add_edge(shard_node, server_node,
                       capacity=job_cap, weight=to_weight(val))
            job_caps[(sid, server_id)] = job_cap

    return G, job_caps


# ─────────────────────────────────────────────────────────────────────────────
# Background graph — solve
# ─────────────────────────────────────────────────────────────────────────────
def solve_background(shards: dict,
                     bg_budgets: dict[int, int]
                     ) -> tuple[list[dict], nx.DiGraph, dict,
                                dict[tuple[int, int], int]]:
    G, job_caps = build_background_graph(shards, bg_budgets)
    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        print("[BG] Solver: infeasible.", file=sys.stderr)
        return [], G, {}, job_caps
    except nx.NetworkXError as exc:
        print(f"[BG] Solver error: {exc}", file=sys.stderr)
        return [], G, {}, job_caps

    # ── Slack edge diagnostic ─────────────────────────────────────────────
    # The SRC→SNK slack edge drains background budget that could NOT be
    # routed to any shard→server job edge.  Non-zero slack means the BG
    # budget exceeds what the current job topology can absorb — i.e. the
    # cluster does not have enough reindex work to fill the allocated budget,
    # or all valuable jobs are already saturated.  Zero slack means every
    # unit of background budget was routed to an actual job.
    total_bg    = sum(bg_budgets.values())
    slack_flow  = flow_dict.get("SRC", {}).get("SNK", 0)
    job_flow    = total_bg - slack_flow          # budget that reached servers
    slack_pct   = slack_flow / total_bg * 100 if total_bg > 0 else 0.0
    job_pct     = job_flow   / total_bg * 100 if total_bg > 0 else 0.0

    print(f"      BG budget (total)    : {total_bg  / FLOW_SCALE:.3f} QPS")
    print(f"      Routed to jobs       : {job_flow  / FLOW_SCALE:.3f} QPS  ({job_pct:.1f}%)")
    print(f"      Slack (unused)       : {slack_flow / FLOW_SCALE:.3f} QPS  ({slack_pct:.1f}%)")
    if slack_flow == 0:
        print(f"      Slack edge           : idle — full budget absorbed by jobs ✅")
    elif slack_pct < 25.0:
        print(f"      Slack edge           : minor spillover — most budget utilised")
    elif slack_pct < 75.0:
        print(f"      Slack edge           : significant unused budget — "
              f"consider lowering alpha or there are few high-value jobs this epoch")
    else:
        print(f"      Slack edge           : majority of budget wasted — "
              f"job topology cannot absorb the allocated BG budget ⚠")
    # ─────────────────────────────────────────────────────────────────────

    jobs: list[dict] = []
    total_job_flow = 0    # all flow on shard→server edges (before filter)
    full_job_flow  = 0    # flow only on fully-saturated edges (after filter)

    for sid_str, sdata in shards.items():
        sid        = int(sid_str)
        maint_cost = sdata["maintenance_cost"]
        is_hot     = sdata["is_hot"]
        shard_node = f"shard_{sid}"
        for server_id in sdata["replicas"]:
            server_node = f"server_{server_id}"
            cap         = job_caps.get((sid, server_id), 0)
            if cap == 0:
                continue
            actual_flow = flow_dict.get(shard_node, {}).get(server_node, 0)
            total_job_flow += actual_flow          # count everything
            if actual_flow >= cap:
                full_job_flow += actual_flow       # count only saturated
                val = reindex_value(sid, server_id, maint_cost, is_hot)
                jobs.append({
                    "shard_id"        : sid,
                    "server_id"       : server_id,
                    "maintenance_cost": maint_cost,
                    "scaled_cap"      : cap,
                    "value"           : val,
                    "is_hot"          : is_hot,
                })

    # ── Fractional-flow diagnostic ────────────────────────────────────────
    # total_job_flow includes ALL shard→server flow (partial + full).
    # full_job_flow  includes ONLY fully-saturated edges (committed jobs).
    # The gap is budget that the solver spent on partial jobs that get
    # discarded — pure waste caused by the integer-rounding step.
    frac_lost   = total_job_flow - full_job_flow
    before_pct  = (total_job_flow / total_bg * 100) if total_bg > 0 else 0.0
    after_pct   = (full_job_flow  / total_bg * 100) if total_bg > 0 else 0.0
    lost_pct    = (frac_lost      / total_bg * 100) if total_bg > 0 else 0.0

    print(f"      BG util before job filter : "
          f"{total_job_flow / FLOW_SCALE:.3f} QPS  ({before_pct:.1f}%)")
    print(f"      BG util after  job filter : "
          f"{full_job_flow  / FLOW_SCALE:.3f} QPS  ({after_pct:.1f}%)")
    print(f"      Lost to fractional flows  : "
          f"{frac_lost      / FLOW_SCALE:.3f} QPS  ({lost_pct:.1f}%)")
    # ─────────────────────────────────────────────────────────────────────
    
    
    return jobs, G, flow_dict, job_caps


# ─────────────────────────────────────────────────────────────────────────────
# Post-process: one replica per logical shard
# ─────────────────────────────────────────────────────────────────────────────
def one_replica_per_shard(jobs: list[dict]) -> list[dict]:
    """Keep only the highest-value job per logical shard (fault tolerance)."""
    best: dict[int, dict] = {}
    for job in jobs:
        sid = job["shard_id"]
        if sid not in best or job["value"] > best[sid]["value"]:
            best[sid] = job
    return list(best.values())


# ─────────────────────────────────────────────────────────────────────────────
# Foreground graph — build
# ─────────────────────────────────────────────────────────────────────────────
def build_foreground_graph(shards: dict,
                            fg_caps: dict[int, int],
                            maintenance_deductions: dict[int, float]
                            ) -> nx.DiGraph:
    """
    Four-layer DAG:  [shard supply nodes] → [servers] → SNK
    shard_i demand  = -scaled_qps(i)
    server_j → SNK  capacity = fg_cap[j] − scaled_deduction[j]  (clamped ≥ 0)
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Foreground graph — feasibility check
# ─────────────────────────────────────────────────────────────────────────────
def check_foreground_feasible(shards: dict,
                               fg_caps: dict[int, int],
                               jobs: list[dict]
                               ) -> tuple[bool, nx.DiGraph, dict]:
    """
    Returns (feasible, G, flow_dict).
    G and flow_dict are returned even on infeasibility for visualization reuse.
    """
    deductions: dict[int, float] = {}
    for job in jobs:
        sid = job["server_id"]
        deductions[sid] = deductions.get(sid, 0.0) + job["maintenance_cost"]

    G = build_foreground_graph(shards, fg_caps, deductions)
    try:
        flow_dict = nx.min_cost_flow(G)
        return True, G, flow_dict
    except (nx.NetworkXUnfeasible, nx.NetworkXError):
        return False, G, {}


# ─────────────────────────────────────────────────────────────────────────────
# Visualization — color palette
# ─────────────────────────────────────────────────────────────────────────────
_C = dict(
    src    = "#4488ff",
    shard  = "#aa66ff",
    server = "#44cc88",
    slack  = "#555566",
    src_e  = "#5599ff",
    sat    = "#ff9933",
    unsat  = "#1a1210",
    srv_e  = "#44cc88",
    evict  = "#ff3333",
    commit = "#33ff88",
    ded    = "#ffcc44",
    bg     = "#0f1117",
    text   = "#dddddd",
    dim    = "#2a2a3a",
)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization — shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _check_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return plt, mpatches
    except ImportError:
        print("[VIZ] matplotlib required:  pip install matplotlib",
              file=sys.stderr)
        sys.exit(1)


def _layered_layout(G: nx.DiGraph) -> dict:
    """Four-column left-to-right layout: SRC | shards | servers | SNK."""
    layers = {
        0: ["SRC"],
        1: sorted([n for n in G.nodes if str(n).startswith("shard_")],
                  key=lambda n: int(n.split("_")[1])),
        2: sorted([n for n in G.nodes if str(n).startswith("server_")],
                  key=lambda n: int(n.split("_")[1])),
        3: ["SNK"],
    }
    x_coords = [0.0, 1.2, 2.8, 4.0]
    pos = {}
    for layer_idx, layer_nodes in layers.items():
        n = len(layer_nodes)
        for i, node in enumerate(layer_nodes):
            y = (i - (n - 1) / 2.0) * 1.2
            pos[node] = (x_coords[layer_idx], y)
    return pos


def _node_color(node: str) -> str:
    if node in ("SRC", "SNK"):
        return _C["src"]
    if str(node).startswith("shard_"):
        return _C["shard"]
    if str(node).startswith("server_"):
        return _C["server"]
    return "#888888"


def _draw_edges(ax, G, pos, edge_list, color,
                style="solid", width=1.6, alpha=0.85, rad=0.0):
    if not edge_list:
        return
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list,
        edge_color=color, style=style, width=width, alpha=alpha,
        arrows=True, arrowsize=15, arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        ax=ax,
    )


def _draw_nodes(ax, G, pos: dict):
    import matplotlib.patches as mp
    for node in G.nodes:
        x, y  = pos[node]
        color = _node_color(node)
        circle = mp.Circle(
            (x, y), radius=0.22,
            facecolor=color + "33", edgecolor=color,
            linewidth=2.2, zorder=3,
        )
        ax.add_patch(circle)
        label = (str(node)
                 .replace("shard_",  "S")
                 .replace("server_", "V"))
        ax.text(x, y, label, color=color, fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=4)


def _mid(pos, u, v, dy=0.14):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    return (x1 + x2) / 2, (y1 + y2) / 2 + dy


def _edge_lbl(ax, lx, ly, text, color, fontsize=6.5):
    ax.text(lx, ly, text, color=color, fontsize=fontsize,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15",
                      fc=_C["bg"], ec="none", alpha=0.75))


def _layer_labels(ax, pos: dict):
    x_coords = [0.0, 1.2, 2.8, 4.0]
    labels   = ["Source", "Logical Shards", "Servers", "Sink"]
    all_ys   = [y for (_, y) in pos.values()]
    bottom   = min(all_ys) - 1.0
    for x, lbl in zip(x_coords, labels):
        ax.text(x, bottom, lbl, color="#555566", fontsize=9,
                ha="center", va="top", fontstyle="italic")


def _graph_ax(fig):
    """
    Add a dark-styled axes for graph drawing into the content area
    (below the tab strip, which occupies the top 4.8% of the figure).
    """
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.88])
    ax.set_facecolor(_C["bg"])
    ax.axis("off")
    return ax


def _set_graph_bounds(ax, pos):
    all_xs = [x for (x, _) in pos.values()]
    all_ys = [y for (_, y) in pos.values()]
    ax.set_xlim(min(all_xs) - 0.6, max(all_xs) + 0.6)
    ax.set_ylim(min(all_ys) - 1.5, max(all_ys) + 0.7)


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 1 — Background graph BEFORE solve
# ─────────────────────────────────────────────────────────────────────────────
def viz_bg_before(fig, G: nx.DiGraph, job_caps: dict):
    import matplotlib.patches as mpatches

    pos = _layered_layout(G)
    ax  = _graph_ax(fig)
    ax.set_title(
        "Background Graph — Before Solve\n"
        "edge labels: capacity (QPS)  |  shard→server weight = reindex value",
        color=_C["text"], fontsize=11, fontweight="bold", pad=10,
    )

    slack_e   = [("SRC", "SNK")]
    src_sh_e  = [(u, v) for u, v in G.edges if u == "SRC" and v != "SNK"]
    job_e     = [(u, v) for u, v in G.edges
                 if str(u).startswith("shard_")
                 and str(v).startswith("server_")]
    srv_snk_e = [(u, v) for u, v in G.edges
                 if str(u).startswith("server_") and v == "SNK"]

    _draw_edges(ax, G, pos, slack_e,   _C["slack"], style="dashed",
                width=1.1, alpha=0.5, rad=0.35)
    _draw_edges(ax, G, pos, src_sh_e,  _C["src_e"], width=1.5)
    _draw_edges(ax, G, pos, job_e,     _C["sat"],   width=1.8, alpha=0.7)
    _draw_edges(ax, G, pos, srv_snk_e, _C["srv_e"], width=1.5)

    for u, v, data in G.edges(data=True):
        if (u, v) == ("SRC", "SNK"):
            continue
        cap = data.get("capacity", 0) / FLOW_SCALE
        wt  = data.get("weight",   0)
        lx, ly = _mid(pos, u, v)
        lbl = (f"cap={cap:.1f}\nval={-wt/WEIGHT_SCALE:.2f}"
               if wt != 0 else f"cap={cap:.1f}")
        _edge_lbl(ax, lx, ly, lbl, "#999999")

    _draw_nodes(ax, G, pos)
    _layer_labels(ax, pos)
    _set_graph_bounds(ax, pos)

    ax.legend(handles=[
        mpatches.Patch(color=_C["sat"],   label="Shard→Server job edge"),
        mpatches.Patch(color=_C["src_e"], label="SRC→Shard"),
        mpatches.Patch(color=_C["srv_e"], label="Server→SNK (BG budget)"),
        mpatches.Patch(color=_C["slack"], label="SRC→SNK slack"),
    ], loc="upper right", framealpha=0.2,
       labelcolor=_C["text"], facecolor=_C["bg"], fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 2 — Background graph AFTER solve
# ─────────────────────────────────────────────────────────────────────────────
def viz_bg_after(fig, G: nx.DiGraph, flow_dict: dict,
                 candidates: list[dict], job_caps: dict):
    import matplotlib.patches as mpatches

    pos = _layered_layout(G)
    ax  = _graph_ax(fig)
    ax.set_title(
        f"Background Graph — After Solve\n"
        f"{len(candidates)} saturated job edge(s) selected  |  "
        f"edge labels: flow / capacity",
        color=_C["text"], fontsize=11, fontweight="bold", pad=10,
    )

    selected  = {(j["shard_id"], j["server_id"]) for j in candidates}
    slack_e   = [("SRC", "SNK")]
    src_sh_e  = [(u, v) for u, v in G.edges if u == "SRC" and v != "SNK"]
    sat_e, unsat_e = [], []
    for u, v in G.edges:
        if str(u).startswith("shard_") and str(v).startswith("server_"):
            sid  = int(u.split("_")[1])
            svid = int(v.split("_")[1])
            (sat_e if (sid, svid) in selected else unsat_e).append((u, v))
    srv_snk_e = [(u, v) for u, v in G.edges
                 if str(u).startswith("server_") and v == "SNK"]

    _draw_edges(ax, G, pos, slack_e,   _C["slack"], style="dashed",
                width=1.1, alpha=0.4, rad=0.35)
    _draw_edges(ax, G, pos, src_sh_e,  _C["src_e"], width=1.4, alpha=0.7)
    _draw_edges(ax, G, pos, unsat_e,   _C["unsat"], width=1.2, alpha=0.4)
    _draw_edges(ax, G, pos, sat_e,     _C["sat"],   width=3.2, alpha=1.0)
    _draw_edges(ax, G, pos, srv_snk_e, _C["srv_e"], width=1.5)

    for u, v, data in G.edges(data=True):
        if (u, v) == ("SRC", "SNK"):
            continue
        cap  = data.get("capacity", 0) / FLOW_SCALE
        flow = flow_dict.get(u, {}).get(v, 0) / FLOW_SCALE
        lx, ly = _mid(pos, u, v)
        _edge_lbl(ax, lx, ly, f"{flow:.1f}/{cap:.1f}",
                  "#ffcc88" if (u, v) in sat_e else "#444455")

    _draw_nodes(ax, G, pos)
    _layer_labels(ax, pos)
    _set_graph_bounds(ax, pos)

    ax.legend(handles=[
        mpatches.Patch(color=_C["sat"],   label="Saturated — job selected"),
        mpatches.Patch(color=_C["unsat"], label="Unsaturated"),
        mpatches.Patch(color=_C["src_e"], label="SRC→Shard"),
        mpatches.Patch(color=_C["srv_e"], label="Server→SNK (BG budget)"),
    ], loc="upper right", framealpha=0.2,
       labelcolor=_C["text"], facecolor=_C["bg"], fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 3 — Post-processing (one replica per shard)
# ─────────────────────────────────────────────────────────────────────────────
def viz_post_processing(fig, candidates: list[dict], kept_jobs: list[dict]):
    import matplotlib.patches as mpatches

    kept_set = {(j["shard_id"], j["server_id"]) for j in kept_jobs}

    by_shard: dict[int, list] = {}
    for c in candidates:
        by_shard.setdefault(c["shard_id"], []).append(c)
    for sid in by_shard:
        by_shard[sid].sort(key=lambda j: j["value"], reverse=True)

    rows = []
    for sid in sorted(by_shard.keys()):
        for job in by_shard[sid]:
            rows.append(job)
        rows.append(None)   # shard group separator

    n_data = sum(1 for r in rows if r is not None)

    ax = fig.add_axes([0.04, 0.03, 0.92, 0.90])
    ax.set_facecolor(_C["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Post-Processing — One Replica Per Shard\n"
        "green = KEEP (highest-value replica)  |  red = DROP (duplicate)",
        color=_C["text"], fontsize=11, fontweight="bold", pad=10,
    )

    cols  = ["Shard", "Server", "Cost (QPS)", "Value", "Hot", "Decision"]
    col_x = [0.04, 0.17, 0.31, 0.49, 0.64, 0.75]

    for cx, lbl in zip(col_x, cols):
        ax.text(cx, 0.96, lbl, color="#aaaaaa", fontsize=9,
                fontweight="bold", va="top")

    # FIX: removed transform=ax.transAxes from axhline and axhspan.
    # Since ax.set_xlim(0,1) and ax.set_ylim(0,1), data coordinates are
    # identical to axes coordinates — the transform was redundant and is
    # no longer accepted by axhline in current matplotlib versions.
    ax.axhline(y=0.935, color="#333344", linewidth=1)

    y        = 0.905
    n_rows   = max(n_data + 2, 5)
    dy       = 0.85 / n_rows
    prev_sid = None

    for row in rows:
        if row is None:
            y -= dy * 0.35
            continue

        sid  = row["shard_id"]
        svid = row["server_id"]
        kept = (sid, svid) in kept_set

        if prev_sid is not None and sid != prev_sid:
            ax.axhline(y=y + dy * 0.65, color="#1e1e2e", linewidth=0.8)

        bg_col = "#152515" if kept else "#251515"
        # FIX: removed transform=ax.transAxes from axhspan (same reason)
        ax.axhspan(y - dy * 0.15, y + dy * 0.55,
                   facecolor=bg_col, alpha=0.7)

        fg_col   = "#55ff88" if kept else "#ff5555"
        decision = "✔  KEEP" if kept else "✘  DROP"
        hot_str  = "🔥" if row.get("is_hot") else "—"

        values = [str(sid), str(svid),
                  f"{row['maintenance_cost']:.3f}",
                  f"{row['value']:.3f}",
                  hot_str, decision]
        for cx, txt in zip(col_x, values):
            color = fg_col if cx == col_x[-1] else _C["text"]
            ax.text(cx, y, txt, color=color, fontsize=8.5, va="center")

        y       -= dy
        prev_sid = sid

    ax.legend(handles=[
        mpatches.Patch(color="#55ff88", label="KEEP — highest value for shard"),
        mpatches.Patch(color="#ff5555", label="DROP — duplicate replica"),
    ], loc="lower right", framealpha=0.25,
       labelcolor=_C["text"], facecolor=_C["bg"], fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 4 — Foreground graph (one tab per feasibility-loop iteration)
# ─────────────────────────────────────────────────────────────────────────────
def viz_fg_iteration(fig, shards: dict, fg_caps: dict[int, int],
                     jobs: list[dict], iteration: int,
                     feasible,             # None | True | False
                     evicted=None,         # job dict being removed, or None
                     final: bool = False):
    """
    iteration=0, Picking up exactly where the cut happened — completing `viz_fg_iteration`, then the `VizNavigator`, `schedule()`, print helpers, and CLI.

```python
    feasible=None → initial full-batch view before any solve
    feasible=False → infeasible; evicted highlights the job being removed
    feasible=True, final=True → committed state; active job edges go green
    """
    import matplotlib.patches as mpatches

    deductions: dict[int, float] = {}
    for job in jobs:
        sid = job["server_id"]
        deductions[sid] = deductions.get(sid, 0.0) + job["maintenance_cost"]

    G   = build_foreground_graph(shards, fg_caps, deductions)
    pos = _layered_layout(G)
    ax  = _graph_ax(fig)

    # ── Title ────────────────────────────────────────────────────────────
    if iteration == 0 and feasible is None:
        status_str  = "Initial full batch — pre-feasibility-check"
        title_color = "#aaaaff"
    elif feasible is True and final:
        status_str  = "✅  FEASIBLE — batch committed"
        title_color = "#44ff88"
    elif feasible is False:
        evict_str = ""
        if evicted:
            evict_str = (f"  →  evicting shard {evicted['shard_id']} "
                         f"on server {evicted['server_id']} "
                         f"(value={evicted['value']:.3f}, "
                         f"cost={evicted['maintenance_cost']:.3f})")
        status_str  = f"❌  INFEASIBLE{evict_str}"
        title_color = "#ff5555"
    else:
        status_str  = "✅  FEASIBLE"
        title_color = "#44ff88"

    ax.set_title(
        f"Foreground Graph — Iteration {iteration}   "
        f"[{len(jobs)} job(s) in batch]\n{status_str}",
        color=title_color, fontsize=11, fontweight="bold", pad=10,
    )

    # ── Classify edges ───────────────────────────────────────────────────
    active_pairs = {(j["shard_id"], j["server_id"]) for j in jobs}
    evicted_pair = ((evicted["shard_id"], evicted["server_id"])
                    if evicted else None)

    inactive_e: list = []
    job_e:      list = []
    evict_e:    list = []
    srv_snk_e:  list = []

    for u, v in G.edges:
        if str(u).startswith("shard_") and str(v).startswith("server_"):
            sid  = int(u.split("_")[1])
            svid = int(v.split("_")[1])
            if evicted_pair and (sid, svid) == evicted_pair:
                evict_e.append((u, v))
            elif (sid, svid) in active_pairs:
                job_e.append((u, v))
            else:
                inactive_e.append((u, v))
        elif str(u).startswith("server_") and v == "SNK":
            srv_snk_e.append((u, v))
        else:
            inactive_e.append((u, v))

    job_color = _C["commit"] if (feasible is True) else _C["sat"]

    _draw_edges(ax, G, pos, inactive_e, _C["dim"],   width=1.1, alpha=0.4)
    _draw_edges(ax, G, pos, job_e,      job_color,   width=2.8, alpha=0.9)
    _draw_edges(ax, G, pos, evict_e,    _C["evict"], width=3.2, alpha=1.0)
    _draw_edges(ax, G, pos, srv_snk_e,  _C["ded"],   width=2.0, alpha=0.9)

    # ── Edge labels ──────────────────────────────────────────────────────
    job_lookup = {(j["shard_id"], j["server_id"]): j for j in jobs}

    for u, v, data in G.edges(data=True):
        cap = data.get("capacity", 0)
        lx, ly = _mid(pos, u, v)

        if str(u).startswith("server_") and v == "SNK":
            nid      = int(u.split("_")[1])
            base_cap = fg_caps.get(nid, 0)
            ded_sc   = to_flow(deductions.get(nid, 0.0))
            lbl = (f"eff={cap/FLOW_SCALE:.1f}\n"
                   f"base={base_cap/FLOW_SCALE:.1f}\n"
                   f"ded={ded_sc/FLOW_SCALE:.1f}")
            _edge_lbl(ax, lx, ly, lbl, _C["ded"], fontsize=6.0)

        elif str(u).startswith("shard_") and str(v).startswith("server_"):
            sid  = int(u.split("_")[1])
            svid = int(v.split("_")[1])
            pair = (sid, svid)

            if pair == evicted_pair:
                lbl   = (f"EVICT\n"
                         f"cost={evicted['maintenance_cost']:.2f}\n"
                         f"val={evicted['value']:.2f}")
                color = _C["evict"]
            elif pair in active_pairs:
                j     = job_lookup.get(pair)
                lbl   = (f"job\ncost={j['maintenance_cost']:.2f}\n"
                         f"val={j['value']:.2f}") if j else "job"
                color = job_color
            else:
                lbl   = f"cap={cap/FLOW_SCALE:.1f}"
                color = "#444455"
            _edge_lbl(ax, lx, ly, lbl, color, fontsize=6.0)

    _draw_nodes(ax, G, pos)
    _layer_labels(ax, pos)
    _set_graph_bounds(ax, pos)

    ax.legend(handles=[
        mpatches.Patch(color=job_color,    label="Active job edge (in batch)"),
        mpatches.Patch(color=_C["evict"],  label="Evicted job edge"),
        mpatches.Patch(color=_C["ded"],    label="Server→SNK (eff. cap after deduction)"),
        mpatches.Patch(color=_C["dim"],    label="Inactive replica edge"),
        mpatches.Patch(color=_C["shard"],  label="Shard node"),
        mpatches.Patch(color=_C["server"], label="Server node"),
    ], loc="upper right", framealpha=0.2,
       labelcolor=_C["text"], facecolor=_C["bg"], fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# VizNavigator — single tabbed window
# ─────────────────────────────────────────────────────────────────────────────

class VizNavigator:
    """
    Displays all visualization stages in one matplotlib figure with a row of
    clickable tab buttons across the top.  Clicking a tab clears the content
    area and redraws that stage.

    Parameters
    ----------
    stages : list of (label: str, draw_fn: Callable[[Figure], None])
        draw_fn receives the shared Figure and should call fig.add_axes(...)
        to place its content below the tab strip.
    """

    TAB_H   = 0.048   # fraction of figure height reserved for the tab strip
    TAB_Y   = 1.0 - TAB_H   # bottom edge of tab strip in figure coordinates

    def __init__(self, stages: list):
        plt, _ = _check_matplotlib()
        import matplotlib.patches as mpatches
        from matplotlib.widgets import Button

        self._plt    = plt
        self._stages = stages
        self._current = 0

        n = len(stages)
        self._fig = plt.figure(
            figsize=(16, 9),
            facecolor=_C["bg"],
        )

        # ── Build tab buttons ─────────────────────────────────────────────
        self._buttons: list = []
        self._btn_axes: list = []
        w = 1.0 / n
        for i, (label, _) in enumerate(stages):
            btn_ax = self._fig.add_axes(
                [i * w, self.TAB_Y, w, self.TAB_H]
            )
            btn_ax.set_facecolor(_C["bg"])
            for spine in btn_ax.spines.values():
                spine.set_edgecolor("#2a2a3a")

            btn = Button(
                btn_ax, label,
                color="#1a1a2a",
                hovercolor="#2a2a4a",
            )
            btn.label.set_fontsize(8.5)
            btn.label.set_color(_C["text"])

            # capture i by default argument to avoid late-binding
            btn.on_clicked(lambda _, idx=i: self._show(idx))

            self._buttons.append(btn)
            self._btn_axes.append(btn_ax)

        # ── Draw first stage ──────────────────────────────────────────────
        self._show(0)
        plt.show()

    def _clear_content(self):
        """Remove all axes that are not part of the tab strip."""
        for ax in self._fig.get_axes():
            if ax not in self._btn_axes:
                self._fig.delaxes(ax)

    def _show(self, idx: int):
        self._clear_content()
        self._current = idx

        # Highlight active tab
        for i, (btn, btn_ax) in enumerate(
                zip(self._buttons, self._btn_axes)):
            active = (i == idx)
            btn_ax.set_facecolor("#2a2a5a" if active else "#1a1a2a")
            btn.label.set_color("#aaccff" if active else _C["text"])
            btn.label.set_fontweight("bold" if active else "normal")

        # Call the draw function for this stage
        _, draw_fn = self._stages[idx]
        draw_fn(self._fig)
        self._fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Main scheduler
# ─────────────────────────────────────────────────────────────────────────────

def schedule(cluster_path: str,
             alpha: float    = DEFAULT_ALPHA,
             visualize: bool = False) -> dict:

    with open(cluster_path) as f:
        cluster = json.load(f)

    meta          = cluster["meta"]
    shards        = cluster["shards"]
    nodes         = cluster["nodes"]
    node_capacity = meta["node_capacity"]

    _banner("ANN Index Maintenance Scheduler")
    print(f"  Config file       : {cluster_path}")
    print(f"  Nodes             : {meta['num_nodes']}")
    print(f"  Shards            : {meta['num_shards']}")
    print(f"  Replication k     : {meta['replication_factor']}")
    print(f"  Total QPS         : {meta['total_qps']:.2f}")
    print(f"  Node capacity     : {node_capacity:.2f} QPS")
    print(f"  Target saturation : {meta['target_saturation']:.1%}")
    print(f"  Actual saturation : {meta['actual_saturation']:.1%}")
    print(f"  Solver status     : {meta['solver_status']}")
    print(f"  Load imbalance    : {meta['load_imbalance_ratio']:.4f}")
    print(f"  BG alpha          : {alpha:.1%}  of node_capacity per server")
    print(f"  Visualizations    : {'ON' if visualize else 'OFF'}")
    print()

    print(f"  {'Node':>5}  {'Load QPS':>10}  {'Utilisation':>12}  "
          f"{'Free cap':>10}  {'BG budget':>10}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*10}")
    for nid_str, ndata in sorted(nodes.items(), key=lambda x: int(x[0])):
        util   = ndata["utilisation"]
        free   = node_capacity * (1.0 - util)
        bg_bgt = node_capacity * alpha
        warn   = "  ⚠ α too high" if free < bg_bgt else ""
        print(f"  {int(nid_str):>5}  {ndata['total_load_qps']:>10.2f}  "
              f"{util:>11.1%}  {free:>10.2f}  {bg_bgt:>10.2f}{warn}")
    print("=" * 65)

    # ── 1. Budgets ─────────────────────────────────────────────────────────
    fg_caps, bg_budgets = compute_budgets(nodes, node_capacity, alpha)

    # ── 2. Baseline FG feasibility ─────────────────────────────────────────
    print("\n[0/4] Baseline foreground feasibility check …")
    ok, _, _ = check_foreground_feasible(shards, fg_caps, [])
    if not ok:
        print(
            "\n[ERROR] Foreground is infeasible even with an empty batch.\n"
            "        Check node_capacity or cluster topology.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("        OK — all query load is routable at baseline.")

    # ── 3. Background optimisation ─────────────────────────────────────────
    print("\n[1/4] Solving background graph (maximise reindex reward) …")
    candidates, bg_G, bg_flow, job_caps = solve_background(shards, bg_budgets)
    print(f"      Saturated shard→server edges (raw candidates): "
          f"{len(candidates)}")

    # Stages list for the navigator — populated as we proceed
    stages: list = []

    if visualize:
        # Capture by default arg to avoid closure issues
        stages.append((
            "BG: Before",
            lambda fig, _G=bg_G, _jc=job_caps:
                viz_bg_before(fig, _G, _jc)
        ))
        stages.append((
            "BG: After",
            lambda fig, _G=bg_G, _fd=bg_flow, _c=candidates, _jc=job_caps:
                viz_bg_after(fig, _G, _fd, _c, _jc)
        ))

    if not candidates:
        print("      No jobs selected — skipping maintenance this epoch.")
        if visualize and stages:
            VizNavigator(stages)
        return {"committed": [], "dropped": [], "feasible": True,
                "iterations": 0, "total_value": 0.0, "total_cost": 0.0}

    # ── 4. One-replica-per-shard filter ────────────────────────────────────
    print("\n[2/4] Applying one-replica-per-shard fault-tolerance filter …")
    jobs = one_replica_per_shard(candidates)
    jobs.sort(key=lambda j: j["value"], reverse=True)
    jobs_after_filter = list(jobs)
    print(f"      Candidates in           : {len(candidates)}")
    print(f"      Dropped (dupe replicas) : {len(candidates) - len(jobs)}")
    print(f"      Jobs remaining          : {len(jobs)}")
    _print_job_table(jobs, "Candidate batch after one-replica filter")

    if visualize:
        stages.append((
            "Post-Process",
            lambda fig, _c=candidates, _j=list(jobs):
                viz_post_processing(fig, _c, _j)
        ))

    # ── 5. Foreground feasibility loop ─────────────────────────────────────
    print("[3/4] Foreground feasibility loop …")
    dropped:    list[dict] = []
    iterations: int        = 0
    feasible:   bool       = False
    final_fg_flow: dict    = {}

    if visualize:
        # Initial full-batch view before the first solve
        stages.append((
            "FG: Initial",
            lambda fig, _s=shards, _fc=fg_caps, _j=list(jobs):
                viz_fg_iteration(fig, _s, _fc, _j,
                                 iteration=0, feasible=None,
                                 evicted=None, final=False)
        ))

    while True:
        feasible, _, fg_flow_iter = check_foreground_feasible(shards, fg_caps, jobs)
        iterations    += 1

        if feasible:
            final_fg_flow = fg_flow_iter
            print(f"      Feasible after {iterations} iteration(s) — "
                  f"committing {len(jobs)} job(s).")
            if visualize:
                stages.append((
                    "FG: ✅ Done",
                    lambda fig, _s=shards, _fc=fg_caps,
                           _j=list(jobs), _it=iterations:
                        viz_fg_iteration(fig, _s, _fc, _j,
                                         iteration=_it, feasible=True,
                                         evicted=None, final=True)
                ))
            break

        if not jobs:
            print("      Batch emptied — foreground infeasible at baseline.",
                  file=sys.stderr)
            if visualize:
                stages.append((
                    "FG: ❌ Empty",
                    lambda fig, _s=shards, _fc=fg_caps, _it=iterations:
                        viz_fg_iteration(fig, _s, _fc, [],
                                         iteration=_it, feasible=False,
                                         evicted=None, final=False)
                ))
            break

        # Evict the lowest-value job
        worst_idx = min(range(len(jobs)), key=lambda i: jobs[i]["value"])
        evicted   = jobs[worst_idx]

        print(f"      Iter {iterations:>3}: infeasible — evicting "
              f"shard {evicted['shard_id']:>4} on server "
              f"{evicted['server_id']:>4}  "
              f"(value={evicted['value']:.3f}, "
              f"cost={evicted['maintenance_cost']:.3f})")

        if visualize:
            stages.append((
                f"FG: Iter {iterations}",
                lambda fig, _s=shards, _fc=fg_caps,
                       _j=list(jobs), _it=iterations, _ev=evicted:
                    viz_fg_iteration(fig, _s, _fc, _j,
                                     iteration=_it, feasible=False,
                                     evicted=_ev, final=False)
            ))

        jobs.pop(worst_idx)
        dropped.append(evicted)

    # ── 6. Results ──────────────────────────────────────────────────────────
    total_val  = sum(j["value"]            for j in jobs)
    total_cost = sum(j["maintenance_cost"] for j in jobs)

    print()
    _banner("Results")
    if feasible and jobs:
        print(f"  ✅  COMMITTED   {len(jobs)} job(s)")
        print(f"      Total reindex value  : {total_val:.3f}")
        print(f"      Total resource cost  : {total_cost:.3f} QPS-equivalent")
        print(f"      Jobs dropped (fallback loop): {len(dropped)}")
        _print_job_table(jobs, "Committed jobs")
    elif feasible and not jobs:
        print("  ⚠️   No maintenance jobs committed this epoch.")
    else:
        print("  ❌  Could not achieve foreground feasibility — "
              "no maintenance this epoch.")

    if dropped:
        _print_job_table(dropped, "Dropped jobs (fallback)")

    print("=" * 65)

    # Launch the navigator after all text output is printed
    if visualize and stages:
        VizNavigator(stages)

    # ── 7. Cluster statistics ─────────────────────────────────────────────
    _print_stats(
        meta           = meta,
        nodes          = nodes,
        shards         = shards,
        fg_caps        = fg_caps,
        bg_budgets     = bg_budgets,
        node_capacity  = node_capacity,
        alpha          = alpha,
        candidates     = candidates,
        after_filter   = jobs_after_filter,
        committed      = jobs,
        dropped_fg     = dropped,
        fg_iterations  = iterations,
        final_fg_flow  = final_fg_flow,
    )

    if visualize and stages:
        VizNavigator(stages)

    return {
        "committed"  : jobs,
        "dropped"    : dropped,
        "feasible"   : feasible,
        "iterations" : iterations,
        "total_value": total_val,
        "total_cost" : total_cost,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Cluster & algorithm statistics
# ─────────────────────────────────────────────────────────────────────────────

def _print_stats(*,
                 meta:          dict,
                 nodes:         dict,
                 shards:        dict,
                 fg_caps:       dict[int, int],
                 bg_budgets:    dict[int, int],
                 node_capacity: float,
                 alpha:         float,
                 candidates:    list[dict],
                 after_filter:  list[dict],
                 committed:     list[dict],
                 dropped_fg:    list[dict],
                 fg_iterations: int,
                 final_fg_flow: dict) -> None:
    """
    Print a structured statistics block covering:
      · Cluster resource utilization
      · Background budget usage at each pipeline stage
      · Job counts and value retention through the pipeline
      · Per-server foreground impact after committed deductions
      · Hot-shard coverage and algorithm efficiency
    """

    SEP  = "─" * 65
    SEP2 = "═" * 65

    # ── Derived scalars ───────────────────────────────────────────────────────
    num_nodes      = meta["num_nodes"]
    total_capacity = node_capacity * num_nodes          # whole-cluster ceiling
    total_load     = meta["total_qps"]
    cluster_util   = total_load / total_capacity        # 0-1

    total_bg_budget = sum(bg_budgets.values()) / FLOW_SCALE   # QPS units

    cost_candidates   = sum(j["maintenance_cost"] for j in candidates)
    cost_after_filter = sum(j["maintenance_cost"] for j in after_filter)
    cost_committed    = sum(j["maintenance_cost"] for j in committed)

    val_candidates   = sum(j["value"] for j in candidates)
    val_after_filter = sum(j["value"] for j in after_filter)
    val_committed    = sum(j["value"] for j in committed)

    n_candidates    = len(candidates)
    n_after_filter  = len(after_filter)
    n_committed     = len(committed)
    n_drop_filter   = n_candidates - n_after_filter
    n_drop_fg       = len(dropped_fg)
    n_drop_total    = n_drop_filter + n_drop_fg

    # Hot-shard coverage
    total_hot = sum(1 for s in shards.values() if s["is_hot"])
    hot_committed = sum(1 for j in committed if j.get("is_hot"))

    # Per-server FG utilization after committed deductions
    server_deductions: dict[int, float] = {}
    for job in committed:
        nid = job["server_id"]
        server_deductions[nid] = server_deductions.get(nid, 0.0) + job["maintenance_cost"]

    server_rows = []
    for nid_str, ndata in sorted(nodes.items(), key=lambda x: int(x[0])):
        nid      = int(nid_str)
        base_cap = fg_caps[nid] / FLOW_SCALE
        ded      = server_deductions.get(nid, 0.0)
        eff_cap  = max(0.0, base_cap - ded)

        # Use actual post-rerouting load from the FG solver if available.
        # The solver is free to shift traffic across replicas, so the static
        # total_load_qps from the cluster config reflects the *old* routing
        # and can make servers look over-capacity even when the flow is valid.
        # Summing flow_dict[shard_i][server_j] gives the true routed load
        # the solver actually assigned to this server after deductions.
        if final_fg_flow:
            routed_load = sum(
                final_fg_flow.get(f"shard_{int(sid_str)}", {})
                             .get(f"server_{nid}", 0)
                for sid_str in shards
            ) / FLOW_SCALE
        else:
            # Fallback to config load if no flow solution exists
            # (shouldn't happen for a committed batch, but be safe)
            routed_load = ndata["total_load_qps"]

        util_before = ndata["total_load_qps"] / node_capacity
        util_after  = routed_load / eff_cap if eff_cap > 0 else float("inf")
        server_rows.append((nid, base_cap, ded, eff_cap,
                            routed_load, util_before, util_after))

    util_afters  = [r[6] for r in server_rows if r[6] != float("inf")]
    peak_util_after = max(util_afters)  if util_afters else float("nan")
    avg_util_after  = (sum(util_afters) / len(util_afters)) if util_afters else float("nan")

    # Value density of committed batch
    val_density = (val_committed / cost_committed) if cost_committed > 0 else 0.0

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  Cluster & Algorithm Statistics")
    print(SEP2)

    # 1. Cluster utilization
    print(f"\n  {'━'*3} CLUSTER RESOURCE UTILIZATION {'━'*3}")
    print(f"  {'Total cluster capacity':<40}: {total_capacity:>10.2f} QPS "
          f"  ({num_nodes} nodes × {node_capacity:.2f})")
    print(f"  {'Total query load':<40}: {total_load:>10.2f} QPS")
    print(f"  {'Overall utilization':<40}: {cluster_util:>10.1%}")
    print(f"  {'Peak node utilization (actual_saturation)':<40}: "
          f"{meta['actual_saturation']:>10.1%}")
    print(f"  {'Target saturation':<40}: "
          f"{meta['target_saturation']:>10.1%}")
    print(f"  {'Load imbalance ratio':<40}: "
          f"{meta['load_imbalance_ratio']:>10.4f}  "
          f"(1.000 = perfect balance)")

    # 2. Background budget
    print(f"\n  {'━'*3} BACKGROUND BUDGET USAGE {'━'*3}")
    print(f"  {'Total BG budget (cluster-wide)':<40}: {total_bg_budget:>10.2f} QPS")
    print(f"  {SEP}")
    print(f"  {'Stage':<30}  {'Jobs':>5}  {'Cost (QPS)':>10}  "
          f"{'% of BG budget':>15}  {'Value':>8}")
    print(f"  {'─'*30}  {'─'*5}  {'─'*10}  {'─'*15}  {'─'*8}")

    def _bg_row(label, n, cost, val):
        pct = (cost / total_bg_budget * 100) if total_bg_budget > 0 else 0.0
        print(f"  {label:<30}  {n:>5}  {cost:>10.3f}  {pct:>14.1f}%  {val:>8.3f}")

    _bg_row("Raw BG solve (candidates)",
            n_candidates, cost_candidates, val_candidates)
    _bg_row("After one-replica filter",
            n_after_filter, cost_after_filter, val_after_filter)
    _bg_row("After FG feasibility loop",
            n_committed, cost_committed, val_committed)

    bg_wasted = total_bg_budget - cost_committed
    print(f"\n  {'BG budget committed':<40}: {cost_committed:>10.3f} QPS  "
          f"({cost_committed/total_bg_budget*100:.1f}%)")
    print(f"  {'BG budget unused / wasted':<40}: {bg_wasted:>10.3f} QPS  "
          f"({bg_wasted/total_bg_budget*100:.1f}%)")

    # 3. Job pipeline & value retention
    print(f"\n  {'━'*3} JOB PIPELINE & VALUE RETENTION {'━'*3}")
    print(f"  {'Stage':<36}  {'Jobs':>5}  {'Dropped':>7}  {'Value':>8}  "
          f"{'Value kept':>10}")
    print(f"  {'─'*36}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*10}")

    def _val_kept(v):
        return f"{v/val_candidates*100:.1f}%" if val_candidates > 0 else "N/A"

    print(f"  {'BG solve output (raw candidates)':<36}  "
          f"{n_candidates:>5}  {'—':>7}  {val_candidates:>8.3f}  {'100.0%':>10}")
    print(f"  {'After one-replica filter':<36}  "
          f"{n_after_filter:>5}  {n_drop_filter:>7}  {val_after_filter:>8.3f}  "
          f"{_val_kept(val_after_filter):>10}")
    print(f"  {'After FG feasibility loop':<36}  "
          f"{n_committed:>5}  {n_drop_fg:>7}  {val_committed:>8.3f}  "
          f"{_val_kept(val_committed):>10}")
    print(f"\n  {'Total jobs dropped (all stages)':<40}: {n_drop_total:>5}")
    print(f"  {'  · Dropped by one-replica filter':<40}: {n_drop_filter:>5}")
    print(f"  {'  · Dropped by FG feasibility loop':<40}: {n_drop_fg:>5}")
    print(f"  {'FG feasibility iterations required':<40}: {fg_iterations:>5}")
    print(f"  {'Committed value density (val/cost)':<40}: {val_density:>10.4f}")

    # 4. Per-server FG impact
    print(f"\n  {'━'*3} PER-SERVER FOREGROUND IMPACT AFTER COMMITTED DEDUCTIONS {'━'*3}")
    print(f"  {'Node':>5}  {'Base cap':>9}  {'Deducted':>9}  "
          f"{'Eff. cap':>9}  {'Load QPS':>9}  "
          f"{'Util (before)':>14}  {'Util (after)':>13}  {'Δ util':>7}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  "
          f"{'─'*14}  {'─'*13}  {'─'*7}")
    for (nid, base_cap, ded, eff_cap, load,
         util_before, util_after) in server_rows:
        delta = util_after - util_before
        util_after_str = (f"{util_after:.1%}"
                          if util_after != float("inf") else "  OVER")
        delta_str = (f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}")
        warn = "  ⚠" if util_after > 1.0 else ""
        print(f"  {nid:>5}  {base_cap:>9.2f}  {ded:>9.3f}  "
              f"{eff_cap:>9.2f}  {load:>9.2f}  "
              f"{util_before:>13.1%}  {util_after_str:>13}  "
              f"{delta_str:>7}{warn}")
    print(f"\n  {'Peak server utilization after deductions':<40}: "
          f"{peak_util_after:.1%}")
    print(f"  {'Avg  server utilization after deductions':<40}: "
          f"{avg_util_after:.1%}")

    # 5. Hot-shard coverage
    print(f"\n  {'━'*3} HOT SHARD COVERAGE {'━'*3}")
    pct_hot = (hot_committed / total_hot * 100) if total_hot > 0 else 0.0
    print(f"  {'Hot shards in cluster':<40}: {total_hot:>5}")
    print(f"  {'Hot shards with committed reindex job':<40}: "
          f"{hot_committed:>5}  ({pct_hot:.1f}%)")
    cold_committed = n_committed - hot_committed
    print(f"  {'Cold shards with committed reindex job':<40}: "
          f"{cold_committed:>5}")

    print(f"\n{SEP2}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _print_job_table(jobs: list[dict], title: str) -> None:
    if not jobs:
        return
    print(f"\n  ── {title} {'─' * max(0, 45 - len(title))}")
    print(f"  {'Shard':>6}  {'Server':>6}  {'Cost (QPS)':>10}  "
          f"{'Value':>8}  Hot")
    print(f"  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*3}")
    for j in sorted(jobs, key=lambda x: x["value"], reverse=True):
        hot_flag = "🔥" if j.get("is_hot") else "  "
        print(f"  {j['shard_id']:>6}  {j['server_id']:>6}  "
              f"{j['maintenance_cost']:>10.3f}  "
              f"{j['value']:>8.3f}  {hot_flag}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ann_maintenance_scheduler",
        description=(
            "ANN index maintenance scheduler using NetworkX network simplex.\n\n"
            "Reads cluster_config.json, selects a high-value batch of reindexing "
            "jobs via a background min-cost flow optimisation, then verifies the "
            "batch fits within foreground query-serving capacity constraints.\n\n"
            "Server capacity is read directly from meta.node_capacity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("cluster_config", help="Path to cluster_config.json")
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, metavar="FLOAT",
        help=(
            f"Fraction of node_capacity the BG graph may spend per server "
            f"(default: {DEFAULT_ALPHA}). "
            f"Must satisfy alpha < 1 − utilisation for every node."
        ),
    )
    p.add_argument(
        "--visualize", type=int, default=0, choices=[0, 1], metavar="0|1",
        help=(
            "Set to 1 to open a tabbed visualization window. Tabs: "
            "BG: Before (background graph topology), "
            "BG: After (flow + saturated edges highlighted), "
            "Post-Process (KEEP/DROP table), "
            "FG: Initial (full-batch deductions before first solve), "
            "FG: Iter N (one tab per infeasible iteration, evicted edge in red), "
            "FG: ✅ Done (committed state, job edges in green). "
            "Default: 0."
        ),
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if not Path(args.cluster_config).exists():
        parser.error(f"File not found: {args.cluster_config}")
    if args.alpha <= 0.0 or args.alpha >= 1.0:
        parser.error("--alpha must be in (0, 1)")

    schedule(args.cluster_config,
             alpha=args.alpha,
             visualize=bool(args.visualize))


if __name__ == "__main__":
    main()
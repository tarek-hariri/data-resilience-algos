//! ANN Index Maintenance Scheduler
//! ================================
//! Three-layer min-cost flow network:
//!   [Shard supply nodes]  ──→  [Server transshipment nodes]  ──→  SINK
//! Shard nodes carry supply = qi(shard_qps).
//! Shard→Server arcs are bounded [0, qi(shard_qps)] so any replica can
//! absorb the full load, letting the solver freely reroute when a server
//! is constrained.
//! Server→Sink arcs are bounded [0, qi(server_remaining_cap)].
//! Feasibility = no penalty arcs carry flow = total_cost() ≈ 0.
//! The network is built ONCE and updated incrementally each iteration via
//! set_bounds() on the affected server→sink arc, enabling the solver's
//! warm-restart (dual repair on a primal-infeasible change).
//!
//! Usage:
//!   ann-scheduler --max-replicas-per-shard <N>
//!
//! --max-replicas-per-shard: maximum number of replicas of a single logical
//!   shard that may be scheduled for reindex in one batch. Defaults to 1.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::time::Instant;

use serde::Deserialize;
use netsimplex::{ArcId, Network, NodeId, SolveStats, SolveStatus};

// ── Scaling ───────────────────────────────────────────────────────────────────
const SCALE: f64 = 10.0;
#[inline] fn qi(f: f64) -> i64 { (f * SCALE).round() as i64 }
#[inline] fn qf(i: i64) -> f64 { i as f64 / SCALE }

const PENALTY: f64 = 1e12;
const MAX_CONSECUTIVE_INFEASIBLE: usize = 1;

// ── Logger ────────────────────────────────────────────────────────────────────
// Writes only to the log file — no terminal output at all.

struct Logger {
    file: std::io::BufWriter<File>,
}

impl Logger {
    fn new(path: &str) -> Self {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .expect("Cannot open log file");
        Logger { file: std::io::BufWriter::new(file) }
    }

    fn log(&mut self, line: &str) {
        writeln!(self.file, "{}", line).expect("Failed to write to log file");
    }

    fn flush(&mut self) {
        self.file.flush().expect("Failed to flush log file");
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Meta {
    num_nodes:     usize,
    num_shards:    usize,
    node_capacity: f64,
}

#[derive(Deserialize)]
struct ShardInfo {
    qps:              f64,
    maintenance_cost: HashMap<String, f64>,  // keyed by replica server ID as string
    replicas:         Vec<usize>,
}

/// Per-node info from the config — used to read baseline foreground load.
#[derive(Deserialize)]
struct NodeInfo {
    total_load_qps: f64,
    utilisation:    f64,
}

#[derive(Deserialize)]
struct ClusterConfig {
    meta:   Meta,
    shards: HashMap<String, ShardInfo>,
    nodes:  HashMap<String, NodeInfo>,
}

// ── Domain types ──────────────────────────────────────────────────────────────

/// A candidate reindex job.
/// ⚠️  resource_cost must be in the same unit as node_capacity (QPS).
///     maintenance_cost from the config is used as a stand-in here — verify
///     units before production use.
#[derive(Debug, Clone)]
pub struct ReindexJob {
    pub shard_id:      usize,
    pub server_id:     usize,
    pub value:         f64,
    pub resource_cost: f64,
}

/// QPS routing instruction derived from the final committed-state flow.
#[derive(Debug)]
pub struct SteeringRule {
    pub shard_id:  usize,
    pub server_id: usize,
    pub qps:       f64,
}

#[derive(Debug)]
pub enum JobResult {
    Committed,
    SkippedReplicaCap,
    SkippedInfeasible,
}

/// Per-iteration record including the timed solve and crate diagnostics.
#[derive(Debug)]
pub struct IterationRecord {
    pub shard_id:  usize,
    pub server_id: usize,
    pub solve_ms:  Option<f64>,
    pub stats:     Option<SolveStats>,
    pub result:    JobResult,
}

// ── Live network wrapper ──────────────────────────────────────────────────────

struct FlowNetwork {
    net:               Network,
    server_sink_arcs:  Vec<ArcId>,
    shard_server_arcs: Vec<Vec<ArcId>>,
    server_cap:        Vec<f64>,
}

fn build_flow_network(cfg: &ClusterConfig) -> FlowNetwork {
    let ns = cfg.meta.num_shards;
    let nv = cfg.meta.num_nodes;

    let mut net = Network::new();

    let mut shard_nodes: Vec<NodeId> = Vec::with_capacity(ns);
    for sid in 0..ns {
        let supply = qi(cfg.shards[&sid.to_string()].qps);
        shard_nodes.push(net.add_node(supply, PENALTY));
    }

    let mut server_nodes: Vec<NodeId> = Vec::with_capacity(nv);
    for _ in 0..nv {
        server_nodes.push(net.add_node(0, PENALTY));
    }

    // Quantization fix: derive sink demand from sum of already-quantized
    // shard supplies rather than cfg.meta.total_qps to avoid rounding gaps.
    let total_supply: i64 = (0..ns)
        .map(|sid| qi(cfg.shards[&sid.to_string()].qps))
        .sum();
    let sink = net.add_node(-total_supply, PENALTY);

    let mut shard_server_arcs: Vec<Vec<ArcId>> = vec![vec![]; ns];
    for sid in 0..ns {
        let shard = &cfg.shards[&sid.to_string()];
        let cap   = qi(shard.qps);
        for &srv in &shard.replicas {
            let arc = net
                .add_arc_bounded(shard_nodes[sid], server_nodes[srv], 0.0, 0, cap)
                .expect("add_arc_bounded (shard→server) returned None");
            shard_server_arcs[sid].push(arc);
        }
    }

    let server_cap: Vec<f64> = vec![cfg.meta.node_capacity; nv];
    let mut server_sink_arcs: Vec<ArcId> = Vec::with_capacity(nv);
    for srv in 0..nv {
        let cap = qi(server_cap[srv]);
        let arc = net
            .add_arc_bounded(server_nodes[srv], sink, 0.0, 0, cap)
            .expect("add_arc_bounded (server→sink) returned None");
        server_sink_arcs.push(arc);
    }

    FlowNetwork { net, server_sink_arcs, shard_server_arcs, server_cap }
}

#[inline]
fn flow_is_feasible(net: &Network) -> bool {
    net.total_cost() < PENALTY / 2.0
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

/// Returns:
///   - committed jobs
///   - steering rules
///   - iteration records
///   - final per-server remaining capacities
///   - actual post-schedule per-server foreground flow (summed from arc flows)
pub fn run_scheduler(
    cfg:                    &ClusterConfig,
    mut jobs:               Vec<ReindexJob>,
    max_replicas_per_shard: usize,
    log:                    &mut Logger,
) -> (Vec<ReindexJob>, Vec<SteeringRule>, Vec<IterationRecord>, Vec<f64>, Vec<f64>) {

    jobs.sort_by(|a, b| b.value.partial_cmp(&a.value)
                              .unwrap_or(std::cmp::Ordering::Equal));

    let mut fn_                   = build_flow_network(cfg);
    let mut committed:              Vec<ReindexJob>       = Vec::new();
    let mut records:                Vec<IterationRecord>  = Vec::new();
    let mut consecutive_infeasible: usize                 = 0;
    let mut replica_commits:        HashMap<usize, usize> =
        (0..cfg.meta.num_shards).map(|sid| (sid, 0usize)).collect();

    for job in jobs {

        // ── Replica-cap guard ────────────────────────────────────────────────
        let committed_for_shard = *replica_commits.get(&job.shard_id).unwrap_or(&0);
        if committed_for_shard >= max_replicas_per_shard {
            records.push(IterationRecord {
                shard_id:  job.shard_id,
                server_id: job.server_id,
                solve_ms:  None,
                stats:     None,
                result:    JobResult::SkippedReplicaCap,
            });
            log.log(&format!(
                "  [REPLICA CAP] shard {} already has {}/{} replicas committed — skipping server {}.",
                job.shard_id, committed_for_shard, max_replicas_per_shard, job.server_id
            ));
            continue;
        }
        // ────────────────────────────────────────────────────────────────────

        let snap = fn_.net.snapshot();

        fn_.server_cap[job.server_id] -= job.resource_cost;
        let new_cap = qi(fn_.server_cap[job.server_id].max(0.0));
        fn_.net.set_bounds(fn_.server_sink_arcs[job.server_id], 0, new_cap);

        // ── Timed solve ──────────────────────────────────────────────────────
        let t0       = Instant::now();
        let _status  = fn_.net.solve();
        let solve_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let stats    = fn_.net.stats();
        // ────────────────────────────────────────────────────────────────────

        let ok = flow_is_feasible(&fn_.net);

        if ok {
            *replica_commits.entry(job.shard_id).or_insert(0) += 1;
            committed.push(job.clone());
            consecutive_infeasible = 0;

            records.push(IterationRecord {
                shard_id:  job.shard_id,
                server_id: job.server_id,
                solve_ms:  Some(solve_ms),
                stats:     Some(stats),
                result:    JobResult::Committed,
            });
        } else {
            fn_.server_cap[job.server_id] += job.resource_cost;
            fn_.net = Network::restore(snap);

            consecutive_infeasible += 1;

            records.push(IterationRecord {
                shard_id:  job.shard_id,
                server_id: job.server_id,
                solve_ms:  Some(solve_ms),
                stats:     Some(stats),
                result:    JobResult::SkippedInfeasible,
            });

            log.log(&format!(
                "  [SKIP] Infeasible: shard {} on server {} (cost {:.4}). \
                 Consecutive streak: {}/{}",
                job.shard_id, job.server_id, job.resource_cost,
                consecutive_infeasible, MAX_CONSECUTIVE_INFEASIBLE
            ));

            if consecutive_infeasible >= MAX_CONSECUTIVE_INFEASIBLE {
                log.log(&format!(
                    "  [ABORT] {} consecutive infeasible attempts — stopping.",
                    MAX_CONSECUTIVE_INFEASIBLE
                ));
                break;
            }
        }
    }

    // ── Extract steering rules and actual per-server foreground flow ──────────
    // These come from the same arc flows so we compute them together in one pass.
    // actual_foreground[srv] = sum of all shard→server arc flows into server srv
    // after the final committed schedule. This is the real post-reroute foreground
    // load, which may differ from the static config baseline because the solver
    // shifted traffic away from servers with maintenance reservations.
    let mut rules:             Vec<SteeringRule> = Vec::new();
    let mut actual_foreground: Vec<f64>          = vec![0.0; cfg.meta.num_nodes];

    for sid in 0..cfg.meta.num_shards {
        let shard = &cfg.shards[&sid.to_string()];
        for (i, &srv) in shard.replicas.iter().enumerate() {
            let flow = fn_.net.flow(fn_.shard_server_arcs[sid][i]);
            if flow > 0 {
                let qps = qf(flow);
                rules.push(SteeringRule { shard_id: sid, server_id: srv, qps });
                actual_foreground[srv] += qps;
            }
        }
    }

    let final_server_cap = fn_.server_cap.clone();
    (committed, rules, records, final_server_cap, actual_foreground)
}

// ── Stats helpers ─────────────────────────────────────────────────────────────

// Returns (mean, median, p99, max, n_committed, n_inf, n_cap)
fn compute_solve_stats(
    records: &[IterationRecord],
) -> (f64, f64, f64, f64, usize, usize, usize) {
    let mut solve_times: Vec<f64> = records
        .iter()
        .filter_map(|r| r.solve_ms)
        .collect();

    let n_committed = records.iter().filter(|r| matches!(r.result, JobResult::Committed)).count();
    let n_inf       = records.iter().filter(|r| matches!(r.result, JobResult::SkippedInfeasible)).count();
    let n_cap       = records.iter().filter(|r| matches!(r.result, JobResult::SkippedReplicaCap)).count();

    if solve_times.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, n_committed, n_inf, n_cap);
    }

    let mean = solve_times.iter().sum::<f64>() / solve_times.len() as f64;
    let max  = solve_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    solve_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if solve_times.len() % 2 == 0 {
        (solve_times[solve_times.len() / 2 - 1] + solve_times[solve_times.len() / 2]) / 2.0
    } else {
        solve_times[solve_times.len() / 2]
    };

    let p99_idx = ((solve_times.len() as f64) * 0.99).ceil() as usize;
    let p99     = solve_times[(p99_idx).min(solve_times.len()) - 1];

    (mean, median, p99, max, n_committed, n_inf, n_cap)
}

/// Computes per-server and cluster-wide utilization after scheduling.
///
/// True utilization = (actual_foreground_flow + maintenance_reserved) / node_capacity
///
/// - actual_foreground_flow: summed from real arc flows in the final network state,
///   NOT the static config baseline. This correctly reflects traffic that was
///   rerouted away from servers with maintenance reservations, preventing >100%
///   figures that would appear if we used the stale pre-schedule baseline.
/// - maintenance_reserved:   node_capacity - final_server_cap[srv]
///
/// Returns (per_server_utils, avg_util, total_cluster_util, baseline_avg_util)
fn compute_utilization_stats(
    cfg:               &ClusterConfig,
    final_server_cap:  &[f64],
    actual_foreground: &[f64],
) -> (Vec<f64>, f64, f64, f64) {
    let nv       = cfg.meta.num_nodes;
    let node_cap = cfg.meta.node_capacity;

    let per_server: Vec<f64> = (0..nv)
        .map(|srv| {
            // Use actual post-reroute flow, NOT static total_load_qps from config.
            // The flow network guarantees foreground + maintenance ≤ node_capacity
            // for every feasible committed state, so this will never exceed 100%.
            let foreground_flow      = actual_foreground[srv];
            let maintenance_reserved = node_cap - final_server_cap[srv];
            (foreground_flow + maintenance_reserved) / node_cap
        })
        .collect();

    let avg_util = per_server.iter().sum::<f64>() / nv as f64;

    let total_consumed: f64 = (0..nv)
        .map(|srv| {
            let foreground_flow      = actual_foreground[srv];
            let maintenance_reserved = node_cap - final_server_cap[srv];
            foreground_flow + maintenance_reserved
        })
        .sum();
    let total_cluster_util = total_consumed / (node_cap * nv as f64);

    // Baseline from config — unchanged pre-schedule reference point.
    let baseline_avg = (0..nv)
        .map(|srv| cfg.nodes[&srv.to_string()].utilisation)
        .sum::<f64>() / nv as f64;

    (per_server, avg_util, total_cluster_util, baseline_avg)
}

// ── CLI arg parsing ───────────────────────────────────────────────────────────

fn parse_args() -> usize {
    let args: Vec<String> = std::env::args().collect();
    let mut max_replicas: usize = 1;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--max-replicas-per-shard" {
            i += 1;
            if i >= args.len() {
                eprintln!("Error: --max-replicas-per-shard requires a value.");
                std::process::exit(1);
            }
            max_replicas = args[i].parse().unwrap_or_else(|_| {
                eprintln!("Error: --max-replicas-per-shard must be a positive integer.");
                std::process::exit(1);
            });
            if max_replicas == 0 {
                eprintln!("Error: --max-replicas-per-shard must be >= 1.");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    max_replicas
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let max_replicas_per_shard = parse_args();
    let mut log = Logger::new("scheduler_output.log");

    let raw = std::fs::read_to_string("cluster_gen.json")
        .expect("cluster_gen.json not found in working directory");
    let cfg: ClusterConfig = serde_json::from_str(&raw)
        .expect("Failed to parse cluster_gen.json");

    log.log(&format!(
        "Cluster loaded: {} shards  {} nodes  node_cap={:.0}",
        cfg.meta.num_shards, cfg.meta.num_nodes, cfg.meta.node_capacity
    ));
    log.log(&format!(
        "Fault tolerance: max {} replica(s) per shard in one batch.",
        max_replicas_per_shard
    ));

    // ⚠️  value         = per-replica maintenance_cost  (stand-in — replace with real function)
    // ⚠️  resource_cost = per-replica maintenance_cost  (verify units match node_capacity)
    let jobs: Vec<ReindexJob> = (0..cfg.meta.num_shards)
        .flat_map(|sid| {
            let s = &cfg.shards[&sid.to_string()];
            s.replicas.iter().map(move |&srv| {
                let cost = s.maintenance_cost
                    .get(&srv.to_string())
                    .copied()
                    .unwrap_or(0.0);
                ReindexJob {
                    shard_id:      sid,
                    server_id:     srv,
                    value:         cost,
                    resource_cost: cost,
                }
            }).collect::<Vec<_>>()
        })
        .collect();

    log.log(&format!("Scheduling {} candidate reindex jobs ({} shards × replicas)...\n",
                     jobs.len(), cfg.meta.num_shards));

    // ── Total runtime clock ───────────────────────────────────────────────────
    let total_t0 = Instant::now();
    let (committed, mut rules, records, final_server_cap, actual_foreground) =
        run_scheduler(&cfg, jobs, max_replicas_per_shard, &mut log);
    let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;
    // ─────────────────────────────────────────────────────────────────────────

    // ── Committed batch ───────────────────────────────────────────────────────
    log.log(&format!(
        "=== Committed Reindex Batch ({} jobs) ===",
        committed.len()
    ));
    for j in &committed {
        log.log(&format!(
            "  shard={:>2}  server={}  value={:>10.4}  cost={:>10.4}",
            j.shard_id, j.server_id, j.value, j.resource_cost
        ));
    }

    // ── QPS steering rules ────────────────────────────────────────────────────
    log.log("\n=== QPS Steering Rules ===");
    rules.sort_by_key(|r| (r.shard_id, r.server_id));
    for r in &rules {
        log.log(&format!(
            "  shard={:>2} → server={}  {:>8.2} QPS",
            r.shard_id, r.server_id, r.qps
        ));
    }

    // ── Solve timings + diagnostics ───────────────────────────────────────────
    log.log("\n=== Network Simplex Solve Timings ===");
    log.log(&format!(
        "  {:<8} {:<8} {:>12}  {:>12}  {:>14}  {:>14}  {}",
        "shard", "server", "solve_ms", "dual_pivots", "primal_pivots", "subtree_nodes", "result"
    ));
    log.log(&format!("  {}", "-".repeat(100)));
    for rec in &records {
        let solve_str = match rec.solve_ms {
            Some(ms) => format!("{:>12.4}", ms),
            None     => format!("{:>12}", "—"),
        };
        let (dual, primal, subtree) = match &rec.stats {
            Some(s) => (
                format!("{:>12}", s.dual_pivots),
                format!("{:>14}", s.primal_pivots),
                format!("{:>14}", s.subtree_nodes),
            ),
            None => (
                format!("{:>12}", "—"),
                format!("{:>14}", "—"),
                format!("{:>14}", "—"),
            ),
        };
        let result_str = match rec.result {
            JobResult::Committed         => "COMMITTED",
            JobResult::SkippedReplicaCap => "SKIPPED (replica cap)",
            JobResult::SkippedInfeasible => "SKIPPED (infeasible)",
        };
        log.log(&format!(
            "  {:<8} {:<8} {}  {}  {}  {}  {}",
            rec.shard_id, rec.server_id, solve_str, dual, primal, subtree, result_str
        ));
    }

    // ── Utilization stats ─────────────────────────────────────────────────────
    let (per_server_utils, avg_util, total_cluster_util, baseline_avg) =
        compute_utilization_stats(&cfg, &final_server_cap, &actual_foreground);

    log.log("\n=== Post-Schedule Server Utilization ===");
    log.log(&format!(
        "  {:<10} {:>18}  {:>20}  {:>22}  {:>20}  {:>16}",
        "server", "baseline_util", "post_schedule_util", "actual_foreground_qps",
        "maintenance_reserved", "remaining_cap_qps"
    ));
    log.log(&format!("  {}", "-".repeat(115)));
    for srv in 0..cfg.meta.num_nodes {
        let baseline             = cfg.nodes[&srv.to_string()].utilisation;
        let maintenance_reserved = cfg.meta.node_capacity - final_server_cap[srv];
        log.log(&format!(
            "  {:<10} {:>17.2}%  {:>19.2}%  {:>22.2}  {:>19.2}   {:>16.2}",
            srv,
            baseline * 100.0,
            per_server_utils[srv] * 100.0,
            actual_foreground[srv],
            maintenance_reserved,
            final_server_cap[srv]
        ));
    }

    // ── End-of-log summary stats ──────────────────────────────────────────────
    let (mean, median, p99, max_solve, n_committed, n_inf, n_cap) =
        compute_solve_stats(&records);

    log.log(&format!("\n{}", "=".repeat(60)));
    log.log("=== Run Statistics ===");
    log.log(&format!("{}", "=".repeat(60)));
    log.log(&format!("  Total scheduler runtime      : {:.4} ms", total_ms));
    log.log(&format!("  Jobs evaluated by solver     : {}", n_committed + n_inf));
    log.log(&format!("  Jobs committed               : {}", n_committed));
    log.log(&format!("  Jobs skipped (infeasible)    : {}", n_inf));
    log.log(&format!("  Jobs skipped (replica cap)   : {}", n_cap));
    log.log(&format!("  Mean   solve time per iter   : {:.4} ms", mean));
    log.log(&format!("  Median solve time per iter   : {:.4} ms", median));
    log.log(&format!("  p99    solve time per iter   : {:.4} ms", p99));
    log.log(&format!("  Max    solve time per iter   : {:.4} ms", max_solve));
    log.log("  ---");
    log.log(&format!("  Baseline avg server util     : {:.10}%", baseline_avg * 100.0));
    log.log(&format!("  Post-schedule avg server util: {:.10}%", avg_util * 100.0));
    log.log(&format!("  Post-schedule total cluster  : {:.10}%", total_cluster_util * 100.0));
    log.log(&format!("  Utilization delta (avg)      : +{:.10}%",
             (avg_util - baseline_avg) * 100.0));
    log.log(&format!("{}", "=".repeat(60)));

    log.flush();
}
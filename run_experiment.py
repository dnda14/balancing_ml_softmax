"""
Experimento comparativo: RR vs WRR vs ML-argmin (baseline, réplica de
Rahimov & Aghayev 2026) vs ML-softmax (propuesta de esta tesis).

Todas las estrategias reciben EXACTAMENTE la misma secuencia de tareas
(misma semilla), para que la comparación sea justa.
"""
import sys
import os
import json
import time
import statistics
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from catboost import CatBoostRegressor

from common.config import LOCAL_NODES, NODE_SPEED, WRR_WEIGHTS
from common.local_workers import start_local_workers, stop_local_workers
from common.docker_workers import ensure_docker_nodes, noop_teardown, reset_all_nodes, restart_node_containers
from common.stats_poller import StatsPoller
from loadgen.generator import generate_task_sizes
from loadgen.client import run_load
from balancers.strategies import RoundRobin, WeightedRoundRobin, MLArgmin, MLSoftmax, LeastConnection, PowerOfTwoChoices


def get_runtime(mode, stats_latency_ms):
    """Devuelve (start_fn, stop_fn) según el modo elegido."""
    if mode == "docker":
        return (lambda nodes: ensure_docker_nodes(nodes)), noop_teardown
    else:
        return (lambda nodes: start_local_workers(nodes, stats_latency_ms=stats_latency_ms)), stop_local_workers


def summarize(name, results, extra=None):
    ok = [r for r in results if r and r["ok"]]
    n_fail = len(results) - len(ok)

    if not ok:
        summary = {
            "strategy": name, "n_requests": len(results), "n_ok": 0, "n_fail": n_fail,
            "latency_mean_ms": None, "latency_median_ms": None, "latency_p90_ms": None,
            "latency_stdev_ms": None, "throughput_req_s": None, "per_node_counts": {},
            "load_distribution_std": None, "FAILED_RUN": True,
        }
        if extra:
            summary.update(extra)
        return summary

    lat = [r["latency_ms"] for r in ok]

    per_node_counts = {}
    for r in ok:
        per_node_counts[r["node"]] = per_node_counts.get(r["node"], 0) + 1
    counts = list(per_node_counts.values())
    std_load = statistics.pstdev(counts) if len(counts) > 1 else 0.0

    duration_s = (max(r["t_start"] for r in ok) - min(r["t_start"] for r in ok)) + (lat[-1] / 1000.0 if lat else 0)
    throughput = len(ok) / duration_s if duration_s > 0 else 0.0

    summary = {
        "strategy": name,
        "n_requests": len(results),
        "n_ok": len(ok),
        "n_fail": n_fail,
        "latency_mean_ms": statistics.mean(lat) if lat else None,
        "latency_median_ms": statistics.median(lat) if lat else None,
        "latency_p90_ms": sorted(lat)[int(len(lat) * 0.9)] if lat else None,
        "latency_stdev_ms": statistics.pstdev(lat) if len(lat) > 1 else None,
        "throughput_req_s": throughput,
        "per_node_counts": per_node_counts,
        "load_distribution_std": std_load,
    }
    if extra:
        summary.update(extra)
    return summary


def _diag(name, res):
    ok = [r for r in res if r and r["ok"]]
    if not ok:
        errs = [r.get("error") for r in res if r][:5]
        print(f"  [DIAG] {name}: 0 exitosas de {len(res)}. Errores de muestra: {errs}")


def run_all(n_requests=150, arrival_rate=3.2, seed=7, stats_latency_ms=35, mode="local", rep=0):
    model = CatBoostRegressor()
    model.load_model("data/model.cbm")

    task_sizes = generate_task_sizes(n_requests, seed=seed)
    start_fn, stop_fn = get_runtime(mode, stats_latency_ms)

    all_summaries = []
    raw_records = []

    def _collect_raw(strategy_name, res):
        for r in res:
            if r is None:
                continue
            raw_records.append({
                "strategy": strategy_name,
                "rep": rep,
                "seed": seed,
                "node": r["node"],
                "task_size": r["task_size"],
                "latency_ms": r["latency_ms"],
                "ok": r["ok"],
            })

    # --- 1. Round Robin ---
    if mode == "docker":
        restart_node_containers(LOCAL_NODES)
    handle = start_fn(LOCAL_NODES)
    try:
        strat = RoundRobin(LOCAL_NODES)
        res = run_load(strat, task_sizes, arrival_rate_per_s=arrival_rate, seed=seed)
        _diag(strat.name if hasattr(strat, "name") else "?", res)
        all_summaries.append(summarize("round_robin", res))
        _collect_raw("round_robin", res)
    finally:
        stop_fn(handle)

    # --- 2. Weighted Round Robin ---
    if mode == "docker":
        restart_node_containers(LOCAL_NODES)
    handle = start_fn(LOCAL_NODES)
    try:
        strat = WeightedRoundRobin(LOCAL_NODES, WRR_WEIGHTS)
        res = run_load(strat, task_sizes, arrival_rate_per_s=arrival_rate, seed=seed)
        _diag(strat.name if hasattr(strat, "name") else "?", res)
        all_summaries.append(summarize("weighted_round_robin", res))
        _collect_raw("weighted_round_robin", res)
    finally:
        stop_fn(handle)

    # --- 3. Least Connection ---
    if mode == "docker":
        restart_node_containers(LOCAL_NODES)
    handle = start_fn(LOCAL_NODES)
    poller = StatsPoller(LOCAL_NODES, interval_ms=100, concurrent=True)
    poller.start()
    try:
        strat = LeastConnection(LOCAL_NODES, poller)
        res = run_load(strat, task_sizes, arrival_rate_per_s=arrival_rate, seed=seed)
        _diag(strat.name if hasattr(strat, "name") else "?", res)
        all_summaries.append(summarize("least_connection", res))
        _collect_raw("least_connection", res)
    finally:
        poller.stop()
        stop_fn(handle)

    # --- 4. Power of Two Choices ---
    if mode == "docker":
        restart_node_containers(LOCAL_NODES)
    handle = start_fn(LOCAL_NODES)
    poller = StatsPoller(LOCAL_NODES, interval_ms=100, concurrent=True)
    poller.start()
    try:
        strat = PowerOfTwoChoices(LOCAL_NODES, poller)
        res = run_load(strat, task_sizes, arrival_rate_per_s=arrival_rate, seed=seed)
        _diag(strat.name if hasattr(strat, "name") else "?", res)
        all_summaries.append(summarize("power_of_two_choices", res))
        _collect_raw("power_of_two_choices", res)
    finally:
        poller.stop()
        stop_fn(handle)

    # --- 5. ML-argmin (baseline, staleness por sondeo SECUENCIAL) ---
    if mode == "docker":
        restart_node_containers(LOCAL_NODES)
    handle = start_fn(LOCAL_NODES)
    poller = StatsPoller(LOCAL_NODES, interval_ms=100, concurrent=False)
    poller.start()
    try:
        strat = MLArgmin(LOCAL_NODES, model, poller, NODE_SPEED)
        res = run_load(strat, task_sizes, arrival_rate_per_s=arrival_rate, seed=seed)
        _diag(strat.name if hasattr(strat, "name") else "?", res)
        all_summaries.append(summarize("ml_argmin_baseline", res, {
            "avg_poll_cycle_ms": poller.avg_cycle_ms(),
        }))
        _collect_raw("ml_argmin_baseline", res)
    finally:
        poller.stop()
        stop_fn(handle)

    # --- 6. ML-softmax (propuesta, sondeo CONCURRENTE + selección probabilística) ---
    if mode == "docker":
        restart_node_containers(LOCAL_NODES)
    handle = start_fn(LOCAL_NODES)
    poller = StatsPoller(LOCAL_NODES, interval_ms=100, concurrent=True)
    poller.start()
    try:
        strat = MLSoftmax(LOCAL_NODES, model, poller, NODE_SPEED, temperature_fraction=0.20)
        res = run_load(strat, task_sizes, arrival_rate_per_s=arrival_rate, seed=seed)
        _diag(strat.name if hasattr(strat, "name") else "?", res)
        all_summaries.append(summarize("ml_softmax_propuesto", res, {
            "avg_poll_cycle_ms": poller.avg_cycle_ms(),
        }))
        _collect_raw("ml_softmax_propuesto", res)
    finally:
        poller.stop()
        stop_fn(handle)

    return all_summaries, raw_records


def run_repeated(n_reps=3, n_requests=150, arrival_rate=3.2, stats_latency_ms=35, base_seed=100, mode="local"):
    all_runs = []
    all_raw = []
    for rep in range(n_reps):
        seed = base_seed + rep
        print(f"--- Repetición {rep+1}/{n_reps} (seed={seed}, modo={mode}) ---")
        summaries, raw = run_all(n_requests=n_requests, arrival_rate=arrival_rate, seed=seed,
                                  stats_latency_ms=stats_latency_ms, mode=mode, rep=rep)
        for s in summaries:
            s["rep"] = rep
            s["seed"] = seed
        all_runs.extend(summaries)
        all_raw.extend(raw)
        for s in summaries:
            if s.get("FAILED_RUN"):
                print(f"  {s['strategy']:<24} *** CORRIDA FALLIDA (n_fail={s['n_fail']}) ***")
            else:
                print(f"  {s['strategy']:<24} lat_media={s['latency_mean_ms']:.1f}ms "
                      f"lat_mediana={s['latency_median_ms']:.1f}ms std_carga={s['load_distribution_std']:.1f}")
    return all_runs, all_raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimento comparativo de balanceo de carga")
    parser.add_argument("--mode", choices=["local", "docker"], default="local",
                         help="'local' = procesos Python (rápido, para pruebas). "
                              "'docker' = contenedores reales ya corriendo con 'docker compose up -d' "
                              "(recomendado para los resultados finales de la tesis).")
    parser.add_argument("--reps", type=int, default=2, help="Número de repeticiones")
    parser.add_argument("--n", type=int, default=120, help="Peticiones por repetición")
    parser.add_argument("--rate", type=float, default=3.2, help="Tasa de llegada (peticiones/seg)")
    parser.add_argument("--stats-latency-ms", type=float, default=35,
                         help="Latencia de red simulada para /stats (solo aplica en modo local; "
                              "en modo docker, configúrala en docker-compose.yml)")
    args = parser.parse_args()

    print(f"Modo de ejecución: {args.mode}")
    if args.mode == "docker":
        print("Asegúrate de haber corrido: docker compose up -d --build")

    runs, raw = run_repeated(n_reps=args.reps, n_requests=args.n, arrival_rate=args.rate,
                              stats_latency_ms=args.stats_latency_ms, base_seed=100, mode=args.mode)
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_repeated.json", "w") as f:
        json.dump(runs, f, indent=2, default=str)

    import csv
    with open("results/raw_requests.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "rep", "seed", "node", "task_size", "latency_ms", "ok"])
        writer.writeheader()
        for r in raw:
            writer.writerow(r)
    print(f"\nDatos crudos guardados: results/raw_requests.csv ({len(raw)} peticiones)")

    # agregación por estrategia (media +- desviación estándar entre repeticiones)
    by_strategy = {}
    for r in runs:
        by_strategy.setdefault(r["strategy"], []).append(r)

    print("\n" + "=" * 110)
    print(f"{'Estrategia':<24}{'Lat.media(ms)':>18}{'Lat.mediana':>16}{'Throughput':>13}{'StdDev carga':>16}")
    print("=" * 110)
    agg_summary = []
    for name, items in by_strategy.items():
        means = [i["latency_mean_ms"] for i in items]
        medians = [i["latency_median_ms"] for i in items]
        thr = [i["throughput_req_s"] for i in items]
        loadstd = [i["load_distribution_std"] for i in items]
        agg = {
            "strategy": name,
            "latency_mean_avg": statistics.mean(means),
            "latency_mean_std": statistics.pstdev(means) if len(means) > 1 else 0.0,
            "latency_median_avg": statistics.mean(medians),
            "throughput_avg": statistics.mean(thr),
            "load_std_avg": statistics.mean(loadstd),
        }
        agg_summary.append(agg)
        print(f"{name:<24}{agg['latency_mean_avg']:>10.1f} +-{agg['latency_mean_std']:>5.1f}"
              f"{agg['latency_median_avg']:>16.1f}{agg['throughput_avg']:>13.2f}{agg['load_std_avg']:>16.2f}")
    print("=" * 110)

    with open("results/aggregate_summary.json", "w") as f:
        json.dump(agg_summary, f, indent=2, default=str)

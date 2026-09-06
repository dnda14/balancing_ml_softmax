import sys
import os
import time
import subprocess
import threading
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from catboost import CatBoostRegressor

from common.config import LOCAL_NODES, NODE_SPEED, WRR_WEIGHTS
from common.docker_workers import ensure_docker_nodes, restart_node_containers
from common.stats_poller import StatsPoller
from loadgen.generator import generate_task_sizes
from loadgen.client import run_load
from balancers.strategies import RoundRobin, WeightedRoundRobin, MLArgmin, MLSoftmax, LeastConnection, PowerOfTwoChoices


def change_cpu_limit(container, cpus):
    print(f"\n[!] Cambiando limite de CPU de {container} a {cpus}...")
    subprocess.run(["docker", "update", "--cpus", str(cpus), container], check=True, capture_output=True)
    print(f"[!] Limite actualizado a {cpus}.")


def run_with_dynamic_change(strategy, task_sizes, arrival_rate, change_delay_s):
    # Restaurar a 1.0 antes de empezar
    change_cpu_limit("lb_thesis-node-a-1", 1.0)
    
    # Hilo para aplicar la degradacion
    def degradator():
        time.sleep(change_delay_s)
        change_cpu_limit("lb_thesis-node-a-1", 0.4)

    t = threading.Thread(target=degradator)
    t.start()
    
    t0_exp = time.time()
    res = run_load(strategy, task_sizes, arrival_rate_per_s=arrival_rate, seed=42)
    
    # Asegurar que el hilo termine
    t.join()
    
    # Calcular timestamp relativo a 0 y asignar fase
    for r in res:
        if r and r["ok"]:
            r["rel_time"] = r["t_start"] - t0_exp
            r["phase"] = "post" if r["rel_time"] >= change_delay_s else "pre"
            
    # Restaurar otra vez al terminar
    change_cpu_limit("lb_thesis-node-a-1", 1.0)
    
    return [r for r in res if r and r["ok"]]


def main():
    parser = argparse.ArgumentParser(description="Escenario 5: Fluctuaciones Dinámicas (Concept Drift)")
    parser.add_argument("--reps", type=int, default=5, help="Número de repeticiones")
    parser.add_argument("--n", type=int, default=400, help="Peticiones por repetición")
    parser.add_argument("--rate", type=float, default=14.5, help="Tasa de llegada (peticiones/seg)")
    args = parser.parse_args()

    print(f"Iniciando escenario dinamico con {args.reps} repeticiones de {args.n} peticiones a {args.rate} req/s...")
    
    # Degradar aprox a la mitad del experimento
    change_delay_s = (args.n / args.rate) * 0.45 
    print(f"La degradación ocurrirá a los {change_delay_s:.1f} segundos.")
    
    ensure_docker_nodes(LOCAL_NODES)
    
    model = CatBoostRegressor()
    model.load_model("data/model.cbm")
    
    all_raw_data = []

    for rep in range(args.reps):
        seed = 42 + rep
        print(f"\n========== REPETICIÓN {rep + 1}/{args.reps} (seed={seed}) ==========")
        task_sizes = generate_task_sizes(args.n, seed=seed)
        
        # 1. Round Robin
        print("\n--- Ejecutando Round Robin ---")
        restart_node_containers(LOCAL_NODES)
        rr_strat = RoundRobin(LOCAL_NODES)
        res_rr = run_with_dynamic_change(rr_strat, task_sizes, args.rate, change_delay_s)
        for r in res_rr:
            all_raw_data.append({"strategy": "Round Robin", "rep": rep, **r})
            
        # 2. Weighted Round Robin
        print("\n--- Ejecutando Weighted Round Robin ---")
        restart_node_containers(LOCAL_NODES)
        wrr_strat = WeightedRoundRobin(LOCAL_NODES, WRR_WEIGHTS)
        res_wrr = run_with_dynamic_change(wrr_strat, task_sizes, args.rate, change_delay_s)
        for r in res_wrr:
            all_raw_data.append({"strategy": "Weighted Round Robin", "rep": rep, **r})
            
        # 3. Least Connection
        print("\n--- Ejecutando Least Connection ---")
        restart_node_containers(LOCAL_NODES)
        poller_lc = StatsPoller(LOCAL_NODES, interval_ms=100, concurrent=True)
        poller_lc.start()
        try:
            lc_strat = LeastConnection(LOCAL_NODES, poller_lc)
            res_lc = run_with_dynamic_change(lc_strat, task_sizes, args.rate, change_delay_s)
            for r in res_lc:
                all_raw_data.append({"strategy": "Least Connection", "rep": rep, **r})
        finally:
            poller_lc.stop()
            
        # 4. Power of Two Choices
        print("\n--- Ejecutando Power of Two Choices ---")
        restart_node_containers(LOCAL_NODES)
        poller_p2c = StatsPoller(LOCAL_NODES, interval_ms=100, concurrent=True)
        poller_p2c.start()
        try:
            p2c_strat = PowerOfTwoChoices(LOCAL_NODES, poller_p2c)
            res_p2c = run_with_dynamic_change(p2c_strat, task_sizes, args.rate, change_delay_s)
            for r in res_p2c:
                all_raw_data.append({"strategy": "Power of Two Choices", "rep": rep, **r})
        finally:
            poller_p2c.stop()
            
        # 5. ML-argmin (baseline)
        print("\n--- Ejecutando ML-Argmin ---")
        restart_node_containers(LOCAL_NODES)
        poller_argmin = StatsPoller(LOCAL_NODES, interval_ms=100, concurrent=False)
        poller_argmin.start()
        try:
            argmin_strat = MLArgmin(LOCAL_NODES, model, poller_argmin, NODE_SPEED)
            res_argmin = run_with_dynamic_change(argmin_strat, task_sizes, args.rate, change_delay_s)
            for r in res_argmin:
                all_raw_data.append({"strategy": "ML-Argmin", "rep": rep, **r})
        finally:
            poller_argmin.stop()
            
        # 6. ML-softmax (propuesto)
        print("\n--- Ejecutando ML-Softmax ---")
        restart_node_containers(LOCAL_NODES)
        poller_softmax = StatsPoller(LOCAL_NODES, interval_ms=100, concurrent=True)
        poller_softmax.start()
        try:
            softmax_strat = MLSoftmax(LOCAL_NODES, model, poller_softmax, NODE_SPEED, temperature_fraction=0.20)
            res_softmax = run_with_dynamic_change(softmax_strat, task_sizes, args.rate, change_delay_s)
            for r in res_softmax:
                all_raw_data.append({"strategy": "ML-Softmax", "rep": rep, **r})
        finally:
            poller_softmax.stop()

    # Guardar CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/dynamic_experiment.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "rep", "phase", "rel_time", "node", "task_size", "latency_ms"])
        writer.writeheader()
        for r in all_raw_data:
            writer.writerow({
                "strategy": r["strategy"], "rep": r["rep"], "phase": r["phase"], 
                "rel_time": r["rel_time"], "node": r["node"], 
                "task_size": r["task_size"], "latency_ms": r["latency_ms"]
            })

    print(f"\n[V] Datos crudos de {args.reps} repeticiones guardados en {csv_path}")

    # Generar grafico promediado
    plt.figure(figsize=(12, 6))
    
    # Agrupar por estrategia y bucket de tiempo (segundo exacto)
    bucketed_data = defaultdict(lambda: defaultdict(list))
    for r in all_raw_data:
        bucket = int(r["rel_time"])
        bucketed_data[r["strategy"]][bucket].append(r["latency_ms"])
        
    colors = {
        "Round Robin": "gray",
        "Weighted Round Robin": "red",
        "Least Connection": "purple",
        "Power of Two Choices": "green",
        "ML-Argmin": "orange",
        "ML-Softmax": "blue"
    }

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    for strat, buckets in bucketed_data.items():
        sorted_buckets = sorted(buckets.keys())
        avg_latencies = [np.mean(buckets[b]) for b in sorted_buckets]
        
        w = 3 # Ventana de media móvil más pequeña porque ya agrupamos por segundo
        if len(sorted_buckets) > w:
            plt.plot(sorted_buckets[w-1:], moving_average(avg_latencies, w), 
                     label=strat, color=colors[strat], linewidth=2)
        
    plt.axvline(x=change_delay_s, color='black', linestyle='--', label=f'Degradacion node-a (t={change_delay_s:.1f}s)')
    
    plt.title(f"Adaptabilidad ante fallas de capacidad (Promedio sobre {args.reps} repeticiones)")
    plt.xlabel("Tiempo del experimento (segundos)")
    plt.ylabel("Latencia Promedio (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/dynamic_experiment_plot.png")
    print("[V] Grafico promediado guardado en results/dynamic_experiment_plot.png")

if __name__ == "__main__":
    main()

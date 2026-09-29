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
import xgboost as xgb
import lightgbm as lgb

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
    parser.add_argument("--strategies", nargs="+", default=["rr", "wrr", "lc", "p2c", "argmin", "softmax", "qubo_batch", "pure_qubo_batch", "ml_qubo_light", "pure_qubo_light"],
                        help="Estrategias a ejecutar (ej. --strategies argmin softmax)")
    args = parser.parse_args()

    print(f"Iniciando escenario dinamico con {args.reps} repeticiones de {args.n} peticiones a {args.rate} req/s...")
    
    # Degradar aprox a la mitad del experimento
    change_delay_s = (args.n / args.rate) * 0.45 
    print(f"La degradación ocurrirá a los {change_delay_s:.1f} segundos.")
    
    ensure_docker_nodes(LOCAL_NODES)
    
    model_cat = CatBoostRegressor()
    model_cat.load_model("data/model_catboost.cbm")
    
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model("data/model_xgboost.json")
    
    model_lgb = lgb.Booster(model_file="data/model_lightgbm.txt")
    
    ml_models = [
        ("catboost", model_cat),
        ("xgboost", model_xgb),
        ("lightgbm", model_lgb)
    ]
    
    all_raw_data = []

    for rep in range(args.reps):
        seed = 42 + rep
        print(f"\n========== REPETICIÓN {rep + 1}/{args.reps} (seed={seed}) ==========")
        task_sizes = generate_task_sizes(args.n, seed=seed)
        
        # 1. Round Robin
        if "rr" in args.strategies:
            print("\n--- Ejecutando Round Robin ---")
            restart_node_containers(LOCAL_NODES)
            rr_strat = RoundRobin(LOCAL_NODES)
            res_rr = run_with_dynamic_change(rr_strat, task_sizes, args.rate, change_delay_s)
            for r in res_rr:
                all_raw_data.append({"strategy": "Round Robin", "rep": rep, **r})
            
        # 2. Weighted Round Robin
        if "wrr" in args.strategies:
            print("\n--- Ejecutando Weighted Round Robin ---")
            restart_node_containers(LOCAL_NODES)
            wrr_strat = WeightedRoundRobin(LOCAL_NODES, WRR_WEIGHTS)
            res_wrr = run_with_dynamic_change(wrr_strat, task_sizes, args.rate, change_delay_s)
            for r in res_wrr:
                all_raw_data.append({"strategy": "Weighted Round Robin", "rep": rep, **r})
            
        # 3. Least Connection
        if "lc" in args.strategies:
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
        if "p2c" in args.strategies:
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
        if "argmin" in args.strategies:
            print("\n--- Ejecutando ML-Argmin ---")
            restart_node_containers(LOCAL_NODES)
            poller_argmin = StatsPoller(LOCAL_NODES, interval_ms=100, concurrent=False)
            poller_argmin.start()
            try:
                argmin_strat = MLArgmin(LOCAL_NODES, model_cat, poller_argmin, NODE_SPEED, model_name="catboost")
                res_argmin = run_with_dynamic_change(argmin_strat, task_sizes, args.rate, change_delay_s)
                for r in res_argmin:
                    all_raw_data.append({"strategy": "ML-Argmin", "rep": rep, **r})
            finally:
                poller_argmin.stop()
            
        # 6. ML-softmax (propuesto)
        if "softmax" in args.strategies:
            for model_name, model_obj in ml_models:
                print(f"\n--- Ejecutando ML-Softmax ({model_name}) ---")
                restart_node_containers(LOCAL_NODES)
                poller_softmax = StatsPoller(LOCAL_NODES, interval_ms=50, concurrent=True)
                poller_softmax.start()
                try:
                    softmax_strat = MLSoftmax(LOCAL_NODES, model_obj, poller_softmax, NODE_SPEED, temperature_fraction=0.20, model_name=model_name)
                    res_softmax = run_with_dynamic_change(softmax_strat, task_sizes, args.rate, change_delay_s)
                    for r in res_softmax:
                        all_raw_data.append({"strategy": softmax_strat.name, "rep": rep, **r})
                finally:
                    poller_softmax.stop()

        # 7. QUBO Batch
        if "qubo_batch" in args.strategies:
            print(f"\n--- Ejecutando QUBO Batch (catboost) ---")
            restart_node_containers(LOCAL_NODES)
            poller_qubo = StatsPoller(LOCAL_NODES, interval_ms=50, concurrent=True)
            poller_qubo.start()
            try:
                from balancers.strategies import MLQuboBatch
                qubo_strat = MLQuboBatch(LOCAL_NODES, model_cat, poller_qubo, NODE_SPEED, model_name="catboost")
                
                # Para micro-batching dinámico, necesitamos envolver select()
                # run_with_dynamic_change espera un objeto con select(task_size)
                # Crearemos un adaptador para que funcione en el framework asincrono actual
                # O como el batching se maneja dentro, quizás deberiamos llamar select_batch
                # Wait, the current runner (run_with_dynamic_change) assumes one-by-one.
                # Actually run_experiment uses process_requests which supports select_batch.
                
                # Let's check how run_experiment handles it... ah, I should just use the wrapper
                class QuboBatchWrapper:
                    def __init__(self, strat): self.strat = strat; self.batch = []; self.name = strat.name
                    def select(self, task_size):
                        return self.strat.select_batch([task_size])[0]
                    def on_complete(self, node_id):
                        self.strat.on_complete(node_id)
                        
                res_qubo = run_with_dynamic_change(QuboBatchWrapper(qubo_strat), task_sizes, args.rate, change_delay_s)
                for r in res_qubo:
                    all_raw_data.append({"strategy": qubo_strat.name, "rep": rep, **r})
            finally:
                poller_qubo.stop()

        # 8. PURE QUBO Batch
        if "pure_qubo_batch" in args.strategies:
            print(f"\n--- Ejecutando Pure QUBO Batch ---")
            restart_node_containers(LOCAL_NODES)
            poller_pure_qubo = StatsPoller(LOCAL_NODES, interval_ms=50, concurrent=True)
            poller_pure_qubo.start()
            try:
                from balancers.strategies import PureQuboBatch
                pure_qubo_strat = PureQuboBatch(LOCAL_NODES, poller_pure_qubo, NODE_SPEED)
                
                class QuboBatchWrapper:
                    def __init__(self, strat): self.strat = strat; self.batch = []; self.name = strat.name
                    def select(self, task_size):
                        return self.strat.select_batch([task_size])[0]
                    def on_complete(self, node_id):
                        self.strat.on_complete(node_id)
                        
                res_pure_qubo = run_with_dynamic_change(QuboBatchWrapper(pure_qubo_strat), task_sizes, args.rate, change_delay_s)
                for r in res_pure_qubo:
                    all_raw_data.append({"strategy": pure_qubo_strat.name, "rep": rep, **r})
            finally:
                poller_pure_qubo.stop()

        # 9. ML QUBO Light (num_reads=1, num_sweeps=10)
        if "ml_qubo_light" in args.strategies:
            print(f"\n--- Ejecutando ML QUBO Light (catboost, sweeps=10) ---")
            restart_node_containers(LOCAL_NODES)
            poller_ml_light = StatsPoller(LOCAL_NODES, interval_ms=50, concurrent=True)
            poller_ml_light.start()
            try:
                from balancers.strategies import MLQuboBatchLight
                ml_light_strat = MLQuboBatchLight(LOCAL_NODES, model_cat, poller_ml_light, NODE_SPEED, model_name="catboost")
                
                class QuboBatchWrapper:
                    def __init__(self, strat): self.strat = strat; self.batch = []; self.name = strat.name
                    def select(self, task_size):
                        return self.strat.select_batch([task_size])[0]
                    def on_complete(self, node_id):
                        self.strat.on_complete(node_id)
                        
                res_ml_light = run_with_dynamic_change(QuboBatchWrapper(ml_light_strat), task_sizes, args.rate, change_delay_s)
                for r in res_ml_light:
                    all_raw_data.append({"strategy": ml_light_strat.name, "rep": rep, **r})
            finally:
                poller_ml_light.stop()

        # 10. Pure QUBO Light (num_reads=1, num_sweeps=10)
        if "pure_qubo_light" in args.strategies:
            print(f"\n--- Ejecutando Pure QUBO Light (sweeps=10) ---")
            restart_node_containers(LOCAL_NODES)
            poller_pure_light = StatsPoller(LOCAL_NODES, interval_ms=50, concurrent=True)
            poller_pure_light.start()
            try:
                from balancers.strategies import PureQuboBatchLight
                pure_light_strat = PureQuboBatchLight(LOCAL_NODES, poller_pure_light, NODE_SPEED)
                
                class QuboBatchWrapper:
                    def __init__(self, strat): self.strat = strat; self.batch = []; self.name = strat.name
                    def select(self, task_size):
                        return self.strat.select_batch([task_size])[0]
                    def on_complete(self, node_id):
                        self.strat.on_complete(node_id)
                        
                res_pure_light = run_with_dynamic_change(QuboBatchWrapper(pure_light_strat), task_sizes, args.rate, change_delay_s)
                for r in res_pure_light:
                    all_raw_data.append({"strategy": pure_light_strat.name, "rep": rep, **r})
            finally:
                poller_pure_light.stop()

    # Guardar CSV
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/dynamic_experiment_{timestamp}.csv"
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
        "ml_softmax_catboost": "blue",
        "ml_softmax_xgboost": "dodgerblue",
        "ml_softmax_lightgbm": "cyan"
    }

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    for strat, buckets in bucketed_data.items():
        sorted_buckets = sorted(buckets.keys())
        avg_latencies = [np.mean(buckets[b]) for b in sorted_buckets]
        
        w = 3 # Ventana de media móvil más pequeña porque ya agrupamos por segundo
        if len(sorted_buckets) > w:
            plt.plot(sorted_buckets[w-1:], moving_average(avg_latencies, w), 
                     label=strat, color=colors.get(strat, "black"), linewidth=2)
        
    plt.axvline(x=change_delay_s, color='black', linestyle='--', label=f'Degradacion node-a (t={change_delay_s:.1f}s)')
    
    plt.title(f"Adaptabilidad ante fallas de capacidad (Promedio sobre {args.reps} repeticiones)")
    plt.xlabel("Tiempo del experimento (segundos)")
    plt.ylabel("Latencia Promedio (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"results/dynamic_experiment_plot_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"[V] Grafico promediado guardado en {plot_path}")

if __name__ == "__main__":
    main()

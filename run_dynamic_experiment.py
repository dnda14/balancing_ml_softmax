import sys
import os
import time
import subprocess
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, os.path.dirname(__file__))
from catboost import CatBoostRegressor

from common.config import LOCAL_NODES, NODE_SPEED, WRR_WEIGHTS
from common.docker_workers import ensure_docker_nodes
from common.stats_poller import StatsPoller
from loadgen.generator import generate_task_sizes
from loadgen.client import run_load
from balancers.strategies import WeightedRoundRobin, MLSoftmax


def change_cpu_limit(container, cpus):
    print(f"\n[!] Cambiando limite de CPU de {container} a {cpus}...")
    subprocess.run(["docker", "update", "--cpus", str(cpus), container], check=True, capture_output=True)
    print(f"[!] Limite actualizado.")


def run_with_dynamic_change(strategy, task_sizes, arrival_rate, change_delay_s):
    # Restaurar a 1.0 antes de empezar
    change_cpu_limit("lb_thesis-node-a-1", 1.0)
    
    # Hilo para aplicar la degradacion a mitad del experimento
    def degradator():
        time.sleep(change_delay_s)
        change_cpu_limit("lb_thesis-node-a-1", 0.4)

    t = threading.Thread(target=degradator)
    t.start()
    
    t0_exp = time.time()
    res = run_load(strategy, task_sizes, arrival_rate_per_s=arrival_rate, seed=42)
    
    # Asegurar que el hilo termine
    t.join()
    
    # Calcular timestamp relativo a 0
    for r in res:
        if r and r["ok"]:
            r["rel_time"] = r["t_start"] - t0_exp
            
    # Restaurar otra vez al terminar
    change_cpu_limit("lb_thesis-node-a-1", 1.0)
    
    return [r for r in res if r and r["ok"]]


def main():
    print("Iniciando escenario dinamico...")
    n_requests = 400
    arrival_rate = 14.5
    seed = 42
    # El experimento dura aprox 400 / 14.5 = 27.5 segundos
    # Degradar a los 13 segundos
    change_delay_s = 13.0
    
    ensure_docker_nodes(LOCAL_NODES)
    
    task_sizes = generate_task_sizes(n_requests, seed=seed)
    
    # 1. Ejecutar WRR
    print("\n--- Ejecutando Weighted Round Robin ---")
    wrr_strat = WeightedRoundRobin(LOCAL_NODES, WRR_WEIGHTS)
    res_wrr = run_with_dynamic_change(wrr_strat, task_sizes, arrival_rate, change_delay_s)
    
    # 2. Ejecutar ML-Softmax
    print("\n--- Ejecutando ML-Softmax ---")
    model = CatBoostRegressor()
    model.load_model("data/model.cbm")
    poller = StatsPoller(LOCAL_NODES, interval_ms=100, concurrent=True)
    poller.start()
    try:
        ml_strat = MLSoftmax(LOCAL_NODES, model, poller, NODE_SPEED, temperature_fraction=0.20)
        res_ml = run_with_dynamic_change(ml_strat, task_sizes, arrival_rate, change_delay_s)
    finally:
        poller.stop()

    # Guardar CSV
    os.makedirs("results", exist_ok=True)
    with open("results/dynamic_experiment.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "rel_time", "node", "task_size", "latency_ms"])
        writer.writeheader()
        for r in res_wrr:
            writer.writerow({"strategy": "WRR", "rel_time": r["rel_time"], "node": r["node"], "task_size": r["task_size"], "latency_ms": r["latency_ms"]})
        for r in res_ml:
            writer.writerow({"strategy": "ML-Softmax", "rel_time": r["rel_time"], "node": r["node"], "task_size": r["task_size"], "latency_ms": r["latency_ms"]})

    print("\n[V] Datos crudos guardados en results/dynamic_experiment.csv")

    # Generar grafico
    plt.figure(figsize=(12, 6))
    
    # Extraer data
    t_wrr = [r["rel_time"] for r in res_wrr]
    l_wrr = [r["latency_ms"] for r in res_wrr]
    
    t_ml = [r["rel_time"] for r in res_ml]
    l_ml = [r["latency_ms"] for r in res_ml]

    def moving_average(x, w):
        import numpy as np
        return np.convolve(x, np.ones(w), 'valid') / w
    
    w = 15
    if len(t_wrr) > w:
        plt.plot(t_wrr[w-1:], moving_average(l_wrr, w), label="WRR (Pesos estaticos)", color="red", linewidth=2)
    if len(t_ml) > w:
        plt.plot(t_ml[w-1:], moving_average(l_ml, w), label="ML-Softmax (Propuesto)", color="blue", linewidth=2)
        
    plt.axvline(x=change_delay_s, color='black', linestyle='--', label=f'Degradacion node-a (t={change_delay_s}s)')
    
    plt.title("Adaptabilidad ante cambios dinamicos en la capacidad (Media movil latencia)")
    plt.xlabel("Tiempo del experimento (segundos)")
    plt.ylabel("Latencia (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/dynamic_experiment_plot.png")
    print("[V] Grafico guardado en results/dynamic_experiment_plot.png")

if __name__ == "__main__":
    main()

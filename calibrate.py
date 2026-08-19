"""
Calibracion previa -- corre esto ANTES de train_model.py / run_experiment.py.

Tu maquina tiene una velocidad de CPU distinta a la mia. Los valores por
defecto (CYCLES_PER_UNIT=20, arrival_rate=3.2) fueron calibrados en mi
entorno de pruebas y probablemente NO sean los correctos en la tuya. Este
script te ayuda a encontrar valores razonables en minutos, en vez de
ajustar a ciegas.

Uso:
    python3 calibrate.py --mode local     # procesos locales (rapido)
    python3 calibrate.py --mode docker    # contra contenedores ya levantados
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
import requests

from common.config import LOCAL_NODES
from common.local_workers import start_local_workers, stop_local_workers
from common.docker_workers import ensure_docker_nodes, noop_teardown


def single_request_latency(node, task_size):
    t0 = time.time()
    r = requests.post(node["url"] + "/process", json={"task_size": task_size}, timeout=15)
    r.raise_for_status()
    return (time.time() - t0) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "docker"], default="local")
    parser.add_argument("--task-size", type=float, default=120_000)
    args = parser.parse_args()

    if args.mode == "docker":
        print("Verificando contenedores Docker (asume 'docker compose up -d --build' ya corrido)...")
        ensure_docker_nodes(LOCAL_NODES)
        handle, stop_fn = None, noop_teardown
    else:
        print("Levantando workers locales...")
        handle = start_local_workers(LOCAL_NODES, stats_latency_ms=0)
        stop_fn = stop_local_workers

    try:
        print(f"\nEnviando 1 peticion a cada nodo (task_size={args.task_size:.0f} unidades)...\n")
        latencies = {}
        for n in LOCAL_NODES:
            lat = single_request_latency(n, args.task_size)
            latencies[n["id"]] = lat
            print(f"  {n['id']:<10} (speed nominal={n['speed']}) -> {lat:.1f} ms")

        avg_lat = sum(latencies.values()) / len(latencies)
        print(f"\nLatencia promedio de una sola peticion: {avg_lat:.1f} ms")

        # --- Recomendacion de CYCLES_PER_UNIT ---
        TARGET_MS = 120.0  # objetivo razonable para una tarea "promedio"
        current_cycles_per_unit = float(os.environ.get("CYCLES_PER_UNIT", "20"))
        suggested = current_cycles_per_unit * (TARGET_MS / avg_lat)
        print(f"\n--- Recomendacion 1: CYCLES_PER_UNIT ---")
        if 60 <= avg_lat <= 250:
            print(f"  OK, esta en un rango razonable ({avg_lat:.0f} ms). No hace falta cambiar nada.")
        else:
            print(f"  Tu latencia actual ({avg_lat:.0f} ms) esta fuera del rango ideal (60-250 ms).")
            print(f"  Prueba con CYCLES_PER_UNIT={suggested:.0f} en vez de {current_cycles_per_unit:.0f}")
            print(f"  (variable de entorno CYCLES_PER_UNIT en worker/app.py o docker-compose.yml)")

        # --- Recomendacion de arrival_rate ---
        approx_capacity = sum(1000.0 / lat for lat in latencies.values())
        suggested_rate_low = approx_capacity * 0.35
        suggested_rate_high = approx_capacity * 0.55
        print(f"\n--- Recomendacion 2: --rate (tasa de llegada) ---")
        print(f"  Capacidad agregada aproximada: {approx_capacity:.1f} peticiones/seg")
        print(f"  Rango sugerido para --rate: entre {suggested_rate_low:.1f} y {suggested_rate_high:.1f}")
        print(f"  (por debajo de la capacidad total para evitar colapso de colas,")
        print(f"   pero alto para que la estrategia de balanceo si importe)")

        mid_rate = (suggested_rate_low + suggested_rate_high) / 2
        print(f"\nEjemplo de comando sugerido:")
        print(f"  python3 run_experiment.py --mode {args.mode} --reps 10 --n 150 --rate {mid_rate:.1f}")

    finally:
        stop_fn(handle)


if __name__ == "__main__":
    main()

"""
Fase de recolección de datos + entrenamiento del modelo predictivo.

Envía tareas a nodos elegidos aleatoriamente (para cubrir una variedad de
combinaciones carga/nodo/tamaño de tarea), registra las características
observadas en el momento de la decisión junto con la latencia REAL medida,
y entrena un CatBoostRegressor para predecir la latencia esperada.

IMPORTANTE: Para que el modelo aprenda a predecir latencia en escenarios de
degradación (concept drift), se recolectan datos en DOS fases:
  Fase 1: Carga normal (todos los nodos en capacidad plena)
  Fase 2: Degradación (se reduce la CPU de node-a a 0.4 durante la recolección)
Esto entrena al modelo con el rango completo de active_requests, avg_latency
y los deltas de tendencia, evitando la saturación que ocurre si solo se
entrena con datos de condiciones estables.
"""
import sys
import os
import time
import random
import json
import argparse
import subprocess
import threading
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(__file__))
from common.config import LOCAL_NODES, NODE_SPEED
from common.local_workers import start_local_workers, stop_local_workers
from common.docker_workers import ensure_docker_nodes, noop_teardown, restart_node_containers
from common.stats_poller import StatsPoller
from loadgen.generator import generate_task_sizes
from balancers.strategies import build_features


def get_runtime(mode, stats_latency_ms):
    if mode == "docker":
        return (lambda nodes: ensure_docker_nodes(nodes)), noop_teardown
    return (lambda nodes: start_local_workers(nodes, stats_latency_ms=stats_latency_ms)), stop_local_workers


def collect_one(node, task_size, poller, rows, idx):
    stats = poller.get_stats(node["id"])
    feats = build_features(stats, task_size, NODE_SPEED[node["id"]])
    try:
        r = requests.post(node["url"] + "/process", json={"task_size": task_size}, timeout=15)
        latency = r.json()["latency_ms"]
    except Exception:
        latency = None
    rows[idx] = (feats, latency)


def _run_collection_batch(n_samples, arrival_rate, seed, poller, assign_probs):
    """Recolecta un lote de muestras con el poller activo."""
    rng = random.Random(seed)
    task_sizes = generate_task_sizes(n_samples, seed=seed)
    rows = [None] * n_samples

    inter_arrivals = np.random.default_rng(seed).exponential(1.0 / arrival_rate, size=n_samples)
    arrival_times = np.cumsum(inter_arrivals)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=60) as ex:
        futures = []
        for i, task_size in enumerate(task_sizes):
            target = t_start + arrival_times[i]
            now = time.time()
            if target > now:
                time.sleep(target - now)
            node = rng.choices(LOCAL_NODES, weights=assign_probs, k=1)[0]
            futures.append(ex.submit(collect_one, node, task_size, poller, rows, i))
        for f in futures:
            f.result()

    return [(r[0], r[1]) for r in rows if r is not None and r[1] is not None]


def collect_training_data(n_samples=600, arrival_rate=11.0, seed=42, mode="local", stats_latency_ms=0):
    start_fn, stop_fn = get_runtime(mode, stats_latency_ms)
    handle = start_fn(LOCAL_NODES)
    poller = StatsPoller(LOCAL_NODES, interval_ms=50, concurrent=True)
    poller.start()

    speeds = [NODE_SPEED[n["id"]] for n in LOCAL_NODES]
    total_speed = sum(speeds)
    assign_probs = [s / total_speed for s in speeds]

    all_rows = []

    try:
        # --- Fase 1: Condiciones normales ---
        print(f"  Fase 1: Recolectando {n_samples} muestras en condiciones normales...")
        rows_normal = _run_collection_batch(n_samples, arrival_rate, seed, poller, assign_probs)
        all_rows.extend(rows_normal)
        print(f"  Fase 1 completada: {len(rows_normal)} muestras.")

        # --- Fase 2: Con degradación (solo modo Docker) ---
        if mode == "docker":
            n_degraded = n_samples // 2  # mitad de muestras en condiciones degradadas
            print(f"  Fase 2: Recolectando {n_degraded} muestras con node-a degradado (0.4 CPUs)...")

            # Degradar node-a
            subprocess.run(
                ["docker", "update", "--cpus", "0.4", "lb_thesis-node-a-1"],
                check=True, capture_output=True,
            )

            # Esperar un momento para que el cambio surta efecto en las métricas
            time.sleep(2)

            rows_degraded = _run_collection_batch(
                n_degraded, arrival_rate, seed + 1000, poller, assign_probs
            )
            all_rows.extend(rows_degraded)
            print(f"  Fase 2 completada: {len(rows_degraded)} muestras.")

            # Restaurar node-a
            subprocess.run(
                ["docker", "update", "--cpus", "1.0", "lb_thesis-node-a-1"],
                check=True, capture_output=True,
            )
            print("  node-a restaurado a 1.0 CPUs.")
        else:
            print("  (Fase 2 omitida: solo disponible en modo Docker)")
    finally:
        poller.stop()
        stop_fn(handle)

    X = [r[0] for r in all_rows]
    y = [r[1] for r in all_rows]
    return X, y


def train_and_save(X, y, out_path="data/model.cbm"):
    model = CatBoostRegressor(
        iterations=400,
        depth=6,
        learning_rate=0.08,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )
    n = len(X)
    split = int(n * 0.85)
    idx = list(range(n))
    random.Random(1).shuffle(idx)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    preds = model.predict(X_val)
    ss_res = sum((yv - p) ** 2 for yv, p in zip(y_val, preds))
    y_mean = sum(y_val) / len(y_val)
    ss_tot = sum((yv - y_mean) ** 2 for yv in y_val)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = (ss_res / len(y_val)) ** 0.5

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save_model(out_path)

    metrics = {"r2_val": r2, "rmse_val_ms": rmse, "n_train": len(X_train), "n_val": len(X_val)}
    with open("data/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recolección de datos + entrenamiento del modelo")
    parser.add_argument("--mode", choices=["local", "docker"], default="local")
    parser.add_argument("--n", type=int, default=600, help="Muestras a recolectar")
    parser.add_argument("--rate", type=float, default=11.0, help="Tasa de llegada (peticiones/seg)")
    parser.add_argument("--stats-latency-ms", type=float, default=0)
    args = parser.parse_args()

    print(f"Modo de ejecución: {args.mode}")
    if args.mode == "docker":
        print("Asegúrate de haber corrido: docker compose up -d --build")

    print("Recolectando datos de entrenamiento (esto puede tardar ~1-2 minutos)...")
    X, y = collect_training_data(n_samples=args.n, arrival_rate=args.rate, seed=42,
                                  mode=args.mode, stats_latency_ms=args.stats_latency_ms)
    print(f"Muestras recolectadas: {len(X)}")

    with open("data/training_data.json", "w") as f:
        json.dump({"X": X, "y": y}, f)

    print("Entrenando modelo CatBoost...")
    model, metrics = train_and_save(X, y)
    print("Métricas de validación:", metrics)

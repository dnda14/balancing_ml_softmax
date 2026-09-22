import time
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor


def _send(node, task_size, out, idx, on_complete=None, target_time=None, overhead_ms=0.0):
    t0 = time.time()
    try:
        r = requests.post(node["url"] + "/process", json={"task_size": task_size}, timeout=15)
        r.raise_for_status()
        server_latency = r.json().get("latency_ms")
        ok = True
        err = None
    except Exception as e:
        server_latency = None
        ok = False
        err = repr(e)
    
    t1 = time.time()
    # Si se provee target_time, medimos end-to-end desde que la petición "llegó" teóricamente
    start_measurement = target_time if target_time else t0
    total_ms = (t1 - start_measurement) * 1000.0
    
    out[idx] = {
        "task_size": task_size,
        "node": node["id"],
        "latency_ms": total_ms,
        "server_latency_ms": server_latency,
        "overhead_ms": overhead_ms,
        "ok": ok,
        "error": err,
        "t_start": start_measurement,
    }
    if on_complete:
        on_complete(node["id"])


def run_load(strategy, task_sizes, arrival_rate_per_s=15.0, max_workers=60, seed=None):
    """Despacha task_sizes contra la estrategia dada, con llegadas Poisson
    (inter-arrival exponencial) para simular tráfico concurrente realista."""
    rng = np.random.default_rng(seed)
    n = len(task_sizes)
    inter_arrivals = rng.exponential(1.0 / arrival_rate_per_s, size=n)
    arrival_times = np.cumsum(inter_arrivals)

    results = [None] * n
    t_start = time.time()

    on_complete = getattr(strategy, 'on_complete', None)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, task_size in enumerate(task_sizes):
            target_time = t_start + arrival_times[i]
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)
            t_algo_start = time.time()
            node, _preds = strategy.select(task_size)
            overhead_ms = (time.time() - t_algo_start) * 1000.0
            futures.append(ex.submit(_send, node, task_size, results, i, on_complete, target_time, overhead_ms))
        for f in futures:
            f.result()

    return results

def run_load_batched(strategy, task_sizes, arrival_rate_per_s=15.0, max_workers=60, seed=None, batch_window_ms=50):
    """
    Despacha task_sizes pero los agrupa en micro-lotes en ventanas de batch_window_ms.
    El balanceador (strategy) recibe un lote completo para decidir de forma conjunta,
    optimizando la carga global de los nodos de destino.
    """
    rng = np.random.default_rng(seed)
    n = len(task_sizes)
    inter_arrivals = rng.exponential(1.0 / arrival_rate_per_s, size=n)
    arrival_times = np.cumsum(inter_arrivals)

    results = [None] * n
    t_start = time.time()

    on_complete = getattr(strategy, 'on_complete', None)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        i = 0
        while i < n:
            target_time = t_start + arrival_times[i]
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)
            
            # Agrupar peticiones que caen dentro de la misma ventana de micro-batching
            batch_start_time = arrival_times[i]
            batch_indices = []
            while i < n and arrival_times[i] <= batch_start_time + (batch_window_ms / 1000.0):
                batch_indices.append(i)
                i += 1
            
            # Asegurar que todas las peticiones del batch hayan llegado cronológicamente
            last_target = t_start + arrival_times[batch_indices[-1]]
            now = time.time()
            if last_target > now:
                time.sleep(last_target - now)

            batch_task_sizes = [task_sizes[j] for j in batch_indices]
            
            t_algo_start = time.time()
            batch_decisions = strategy.select_batch(batch_task_sizes)
            overhead_ms_per_task = ((time.time() - t_algo_start) * 1000.0) / len(batch_task_sizes)
            
            for j_idx, idx in enumerate(batch_indices):
                node, _preds = batch_decisions[j_idx]
                req_target_time = t_start + arrival_times[idx]
                futures.append(ex.submit(_send, node, batch_task_sizes[j_idx], results, idx, on_complete, req_target_time, overhead_ms_per_task))
                
        for f in futures:
            f.result()

    return results

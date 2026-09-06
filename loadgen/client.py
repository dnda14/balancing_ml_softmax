import time
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor


def _send(node, task_size, out, idx, on_complete=None):
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
    total_ms = (time.time() - t0) * 1000.0
    out[idx] = {
        "task_size": task_size,
        "node": node["id"],
        "latency_ms": total_ms,
        "server_latency_ms": server_latency,
        "ok": ok,
        "error": err,
        "t_start": t0,
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
            target = t_start + arrival_times[i]
            now = time.time()
            if target > now:
                time.sleep(target - now)
            node, _preds = strategy.select(task_size)
            futures.append(ex.submit(_send, node, task_size, results, i, on_complete))
        for f in futures:
            f.result()

    return results

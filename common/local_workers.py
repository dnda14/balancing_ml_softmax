import os
import subprocess
import time
import signal
import requests

WORKER_APP = os.path.join(os.path.dirname(__file__), "..", "worker", "app.py")


def start_local_workers(nodes, stats_latency_ms=0):
    procs = []
    for n in nodes:
        port = n["url"].rsplit(":", 1)[-1]
        env = os.environ.copy()
        env["NODE_ID"] = n["id"]
        env["NODE_SPEED"] = str(n["speed"])
        env["SOFTWARE_THROTTLE"] = str(n["speed"])  # solo en modo local: simula heterogeneidad
        env["PORT"] = port
        env["STATS_LATENCY_MS"] = str(stats_latency_ms)
        p = subprocess.Popen(
            ["python3", WORKER_APP],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)

    # esperar a que todos respondan /health
    for n in nodes:
        ok = False
        for _ in range(100):
            try:
                r = requests.get(n["url"] + "/health", timeout=0.5)
                if r.status_code == 200:
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(0.2)
        if not ok:
            raise RuntimeError(f"El worker {n['id']} no respondió a tiempo en {n['url']}")
    time.sleep(0.3)  # margen extra de estabilización
    return procs


def stop_local_workers(procs):
    for p in procs:
        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            pass
    time.sleep(0.5)
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass

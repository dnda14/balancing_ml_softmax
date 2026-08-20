"""
Worker de balanceo de carga — simula un nodo de cómputo heterogéneo.

Expone:
  POST /process   -> procesa una tarea (carga de CPU real, no time.sleep) y devuelve la latencia
  GET  /stats      -> estado actual del nodo (peticiones activas, latencia reciente)
  GET  /health     -> chequeo de salud

Variables de entorno:
  NODE_ID            identificador del nodo (ej. "node-a")
  NODE_SPEED         velocidad relativa de cómputo (1.0 = referencia; 0.6 = 40% más lento)
  CYCLES_PER_UNIT     ciclos de trabajo por unidad de tarea a velocidad=1.0 (calibración)
  PORT                puerto HTTP
"""
import os
import time
import threading
from flask import Flask, request, jsonify

app = Flask(__name__)

NODE_ID = os.environ.get("NODE_ID", "node0")
SPEED = float(os.environ.get("NODE_SPEED", "1.0"))  # metadato: capacidad nominal conocida del nodo
CYCLES_PER_UNIT = float(os.environ.get("CYCLES_PER_UNIT", "20"))
STATS_LATENCY_MS = float(os.environ.get("STATS_LATENCY_MS", "0"))

# SOLO para modo local (sin Docker): simula heterogeneidad de capacidad
# dividiendo el trabajo por este factor. En Docker se deja en 1.0 por
# defecto (no seteado en docker-compose.yml) porque la heterogeneidad real
# la impone el límite de CPU del contenedor (cgroups) -- ver docker-compose.yml.
SOFTWARE_THROTTLE = float(os.environ.get("SOFTWARE_THROTTLE", "1.0"))

_lock = threading.Lock()
_active_requests = 0
_completed_requests = 0
_recent_latencies = []
_MAX_RECENT = 50


def busy_work(units: float) -> None:
    """Genera carga de CPU real (no time.sleep) proporcional a 'units'.

    En Docker, la heterogeneidad de capacidad entre nodos la genera el
    límite real de CPU del contenedor (cgroups, vía 'cpus:' en
    docker-compose.yml). SOFTWARE_THROTTLE queda en 1.0 en ese caso.
    En modo local (sin Docker) no existe ese límite real, así que
    SOFTWARE_THROTTLE permite seguir simulando heterogeneidad para
    pruebas rápidas de la lógica."""
    cycles = max(1, int(units * CYCLES_PER_UNIT / SOFTWARE_THROTTLE))
    x = 0.0001
    for _ in range(cycles):
        x = (x * 1.0000001 + 1.0) % 999999937.0


@app.route("/process", methods=["POST"])
def process():
    global _active_requests, _completed_requests
    payload = request.get_json(force=True)
    units = float(payload.get("task_size", 100_000))

    with _lock:
        _active_requests += 1

    t0 = time.time()
    busy_work(units)
    elapsed_ms = (time.time() - t0) * 1000.0

    with _lock:
        _active_requests -= 1
        _completed_requests += 1
        _recent_latencies.append(elapsed_ms)
        if len(_recent_latencies) > _MAX_RECENT:
            _recent_latencies.pop(0)

    return jsonify({"node": NODE_ID, "latency_ms": elapsed_ms})


@app.route("/stats", methods=["GET"])
def stats():
    if STATS_LATENCY_MS > 0:
        time.sleep(STATS_LATENCY_MS / 1000.0)  # simula latencia de red real entre nodos
    with _lock:
        avg_recent = (sum(_recent_latencies) / len(_recent_latencies)) if _recent_latencies else 0.0
        return jsonify({
            "node": NODE_ID,
            "active_requests": _active_requests,
            "completed_requests": _completed_requests,
            "avg_recent_latency_ms": avg_recent,
            "speed": SPEED,
        })


@app.route("/reset", methods=["POST"])
def reset():
    global _active_requests, _completed_requests, _recent_latencies
    with _lock:
        _active_requests = 0
        _completed_requests = 0
        _recent_latencies.clear()
    return jsonify({"status": "reset", "node": NODE_ID})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "node": NODE_ID})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)

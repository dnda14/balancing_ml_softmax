"""
StatsPoller: mantiene una caché de las métricas de cada nodo, actualizada en
segundo plano de forma SECUENCIAL (reproduce el diseño de Rahimov & Aghayev,
2026) o CONCURRENTE (mejora propuesta en esta tesis).

La diferencia clave: con N nodos y una latencia de red/estado de ~L ms por
consulta, un ciclo de actualización secuencial tarda ~N*L ms en completarse,
mientras que uno concurrente tarda ~L ms independientemente de N. Esto
determina cuán "desactualizada" (stale) puede estar la información que usa
el balanceador al tomar una decisión.
"""
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor


class StatsPoller:
    def __init__(self, nodes, interval_ms=100, concurrent=False, max_workers=8, timeout_s=1.0):
        self.nodes = nodes
        self.interval = interval_ms / 1000.0
        self.concurrent = concurrent
        self.timeout_s = timeout_s

        self.cache = {
            n["id"]: {"active_requests": 0, "avg_recent_latency_ms": 0.0}
            for n in nodes
        }
        self.last_update_ts = {n["id"]: 0.0 for n in nodes}

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=max_workers) if concurrent else None
        self._thread = threading.Thread(target=self._run, daemon=True)

        # telemetría para el reporte de la tesis
        self.cycle_durations_ms = []

    def _fetch_one(self, node):
        try:
            r = requests.get(node["url"] + "/stats", timeout=self.timeout_s)
            data = r.json()
            with self._lock:
                self.cache[node["id"]] = data
                self.last_update_ts[node["id"]] = time.time()
        except Exception:
            pass  # nodo no disponible momentáneamente; se conserva el valor previo en caché

    def _run(self):
        while not self._stop.is_set():
            t0 = time.time()
            if self.concurrent:
                list(self._executor.map(self._fetch_one, self.nodes))
            else:
                for n in self.nodes:
                    self._fetch_one(n)
            cycle_ms = (time.time() - t0) * 1000.0
            with self._lock:
                self.cycle_durations_ms.append(cycle_ms)
            sleep_time = max(0.0, self.interval - (time.time() - t0))
            time.sleep(sleep_time)

    def start(self):
        self._thread.start()
        time.sleep(0.3)  # esperar al menos un ciclo antes de servir tráfico

    def stop(self):
        self._stop.set()
        if self._executor:
            self._executor.shutdown(wait=False)

    def get_stats(self, node_id):
        with self._lock:
            return dict(self.cache[node_id])

    def get_staleness_ms(self, node_id):
        with self._lock:
            ts = self.last_update_ts[node_id]
        if ts == 0.0:
            return None
        return (time.time() - ts) * 1000.0

    def avg_cycle_ms(self):
        with self._lock:
            if not self.cycle_durations_ms:
                return 0.0
            return sum(self.cycle_durations_ms) / len(self.cycle_durations_ms)

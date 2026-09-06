"""
Las 4 estrategias de balanceo de carga comparadas en la tesis:

  1. round_robin              -> Grupo A (heurística clásica)
  2. weighted_round_robin     -> Grupo A (heurística clásica, pesos por capacidad)
  3. ml_argmin (baseline)     -> Grupo B, réplica del diseño de Rahimov & Aghayev (2026):
                                  selección determinística (argmin) sobre estadísticas
                                  actualizadas de forma secuencial/periódica.
  4. ml_softmax (propuesta)   -> Grupo B, aporte de esta tesis: selección probabilística
                                  (softmax) sobre estadísticas actualizadas de forma
                                  concurrente.
"""
import math
import random
import threading
import numpy as np


def build_features(stats: dict, task_size: float, node_speed: float, staleness_ms: float) -> list:
    """Vector de características usado por el modelo predictivo.

    Incluye features de tendencia (delta) que capturan la *dirección* del
    cambio en el nodo entre ciclos consecutivos del StatsPoller:
      - delta_active_requests > 0  →  la cola está creciendo (empeorando)
      - delta_avg_latency_ms > 0   →  la latencia está subiendo (empeorando)
    
    Nuevos features ingenierizados:
      - estimated_service_time: ratio puro de trabajo/velocidad (ahorra al árbol hacer la división)
      - staleness_ms: penaliza o contextualiza datos antiguos
      - node_utilization: normaliza la carga activa en función de la capacidad de hilos
    """
    estimated_service_time = task_size / node_speed if node_speed > 0 else 0
    node_utilization = stats.get("active_requests", 0) / 8.0

    return [
        task_size,
        stats.get("active_requests", 0),
        stats.get("avg_recent_latency_ms", 0.0),
        stats.get("delta_active_requests", 0),
        stats.get("delta_avg_latency_ms", 0.0),
        estimated_service_time,
        staleness_ms,
        node_utilization,
    ]


class RoundRobin:
    name = "round_robin"

    def __init__(self, nodes):
        self.nodes = nodes
        self._idx = 0

    def select(self, task_size=None):
        node = self.nodes[self._idx % len(self.nodes)]
        self._idx += 1
        return node, None


class WeightedRoundRobin:
    """Smooth Weighted Round Robin (mismo algoritmo que usa nginx internamente)."""
    name = "weighted_round_robin"

    def __init__(self, nodes, weights: dict):
        self.nodes = nodes
        self.weights = weights
        self.current = {n["id"]: 0 for n in nodes}
        self.total_weight = sum(weights.values())

    def select(self, task_size=None):
        for n in self.nodes:
            self.current[n["id"]] += self.weights[n["id"]]
        best = max(self.nodes, key=lambda n: self.current[n["id"]])
        self.current[best["id"]] -= self.total_weight
        return best, None


class LeastConnection:
    name = "least_connection"

    def __init__(self, nodes, poller):
        self.nodes = nodes
        self.poller = poller

    def select(self, task_size=None):
        best_node = None
        min_active = float('inf')
        for n in self.nodes:
            stats = self.poller.get_stats(n["id"])
            active = stats.get("active_requests", 0)
            if active < min_active:
                min_active = active
                best_node = n
        return best_node, None


class PowerOfTwoChoices:
    name = "power_of_two_choices"

    def __init__(self, nodes, poller):
        self.nodes = nodes
        self.poller = poller

    def select(self, task_size=None):
        n1, n2 = random.sample(self.nodes, 2)
        stats1 = self.poller.get_stats(n1["id"])
        stats2 = self.poller.get_stats(n2["id"])
        if stats1.get("active_requests", 0) <= stats2.get("active_requests", 0):
            return n1, None
        return n2, None


class MLArgmin:
    """Baseline: réplica del enfoque de Rahimov y Aghayev (2026).
    Selección determinística (argmin) sobre la predicción del modelo."""
    
    def __init__(self, nodes, model, poller, node_speed: dict, model_name: str = "catboost"):
        self.nodes = nodes
        self.model = model
        self.poller = poller
        self.node_speed = node_speed
        self.name = f"ml_argmin_{model_name}"

    def select(self, task_size):
        preds = []
        for n in self.nodes:
            stats = self.poller.get_stats(n["id"])
            staleness = self.poller.get_staleness_ms(n["id"])
            staleness = staleness if staleness is not None else 0.0
            feats = build_features(stats, task_size, self.node_speed[n["id"]], staleness)
            pred = self.model.predict(np.array([feats]))[0]
            preds.append(pred)
        best_idx = min(range(len(preds)), key=lambda i: preds[i])
        return self.nodes[best_idx], preds


class MLSoftmax:
    """Propuesta de la tesis: selección probabilística (softmax) sobre la
    predicción del modelo, usada junto con recolección concurrente de
    métricas (ver StatsPoller con concurrent=True).

    Mejora clave: contador LOCAL de peticiones en vuelo (pending).
    Entre ciclos del poller (~100 ms), pueden llegar decenas de peticiones
    que ven los mismos datos remotos "desactualizados". Sin el contador
    local, todas eligen el mismo nodo (efecto manada). Con él, cada
    llamada a select() ve instantáneamente las peticiones que ya se
    despacharon en este ciclo, porque el contador se incrementa en
    select() y se decrementa vía on_complete() cuando la petición termina.

    La temperatura se escala de forma adaptativa como una fracción de la
    magnitud de las predicciones actuales (en vez de una constante fija),
    porque el rango absoluto de latencias predichas varía mucho según el
    nivel de congestión del sistema en cada momento."""

    def __init__(self, nodes, model, poller, node_speed: dict, temperature_fraction: float = 0.20, model_name: str = "catboost"):
        self.nodes = nodes
        self.model = model
        self.poller = poller
        self.node_speed = node_speed
        self.temperature_fraction = temperature_fraction
        self.name = f"ml_softmax_{model_name}"
        # Contador local de peticiones en vuelo (despachadas pero no terminadas)
        self._pending = {n["id"]: 0 for n in nodes}
        self._pending_lock = threading.Lock()

    def select(self, task_size):
        preds = []
        for n in self.nodes:
            stats = self.poller.get_stats(n["id"])
            # Combinar datos remotos del poller con el contador local en tiempo real
            with self._pending_lock:
                stats["active_requests"] = stats.get("active_requests", 0) + self._pending[n["id"]]
            staleness = self.poller.get_staleness_ms(n["id"])
            staleness = staleness if staleness is not None else 0.0
            feats = build_features(stats, task_size, self.node_speed[n["id"]], staleness)
            pred = self.model.predict(np.array([feats]))[0]
            preds.append(pred)

        mean_pred = sum(preds) / len(preds)
        temperature = max(1.0, self.temperature_fraction * mean_pred)

        # menor latencia predicha -> mayor probabilidad
        neg = [-p / temperature for p in preds]
        m = max(neg)
        exps = [math.exp(v - m) for v in neg]
        total = sum(exps)
        probs = [e / total for e in exps]

        r = random.random()
        cum = 0.0
        chosen_idx = len(self.nodes) - 1
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                chosen_idx = i
                break

        # Registrar la petición como "en vuelo" instantáneamente
        with self._pending_lock:
            self._pending[self.nodes[chosen_idx]["id"]] += 1

        return self.nodes[chosen_idx], preds

    def on_complete(self, node_id):
        """Callback invocado por el cliente cuando la petición HTTP termina."""
        with self._pending_lock:
            self._pending[node_id] = max(0, self._pending[node_id] - 1)


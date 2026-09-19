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

class MLQubo:
    """Implementación de prueba usando QUBO para una sola petición.
    Matemáticamente equivalente a Argmin, pero usa Simulated Annealing
    para encontrar la solución."""
    
    def __init__(self, nodes, model, poller, node_speed: dict, model_name: str = "catboost"):
        self.nodes = nodes
        self.model = model
        self.poller = poller
        self.node_speed = node_speed
        self.name = f"ml_qubo_{model_name}"
        try:
            import dimod
            self.sampler = dimod.SimulatedAnnealingSampler()
        except ImportError:
            print("Error: librería 'dimod' no instalada. Ejecuta: pip install dimod")
            self.sampler = None

    def select(self, task_size):
        if not self.sampler:
            # Fallback a argmin si no hay sampler
            return MLArgmin(self.nodes, self.model, self.poller, self.node_speed).select(task_size)
            
        import dimod
        preds = []
        for n in self.nodes:
            stats = self.poller.get_stats(n["id"])
            staleness = self.poller.get_staleness_ms(n["id"])
            staleness = staleness if staleness is not None else 0.0
            feats = build_features(stats, task_size, self.node_speed[n["id"]], staleness)
            pred = self.model.predict(np.array([feats]))[0]
            preds.append(pred)

        # Formular el modelo QUBO:
        # Variables x_i = 1 si se elige el nodo i
        # Costo = sum(pred_i * x_i)
        # Restricción: sum(x_i) == 1
        # Penalización: P * (sum(x_i) - 1)^2
        
        P = max(preds) * 10
        
        linear = {i: preds[i] - P for i in range(len(self.nodes))}
        quadratic = {}
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                quadratic[(i, j)] = 2 * P

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, P, dimod.BINARY)

        # Resolver
        sampleset = self.sampler.sample(bqm, num_reads=1)
        best_sample = sampleset.first.sample
        
        chosen_idx = 0
        for i in range(len(self.nodes)):
            if best_sample[i] == 1:
                chosen_idx = i
                break
                
        return self.nodes[chosen_idx], preds

class MLQuboBatch:
    """Implementación de QUBO para Micro-Batching.
    Recibe un lote de tareas y las distribuye para minimizar la varianza de carga esperada
    en todos los nodos de destino."""
    
    MAX_BATCH_SIZE = 8  # Lotes más grandes se parten para evitar explosión combinatoria
    
    def __init__(self, nodes, model, poller, node_speed: dict, model_name: str = "catboost"):
        self.nodes = nodes
        self.model = model
        self.poller = poller
        self.node_speed = node_speed
        self.name = f"ml_qubo_batch_{model_name}"
        
        self._pending = {n["id"]: 0 for n in nodes}
        self._pending_lock = threading.Lock()
        
        try:
            import dimod
            self.sampler = dimod.SimulatedAnnealingSampler()
        except ImportError:
            print("Error: librería 'dimod' no instalada. Ejecuta: pip install dimod")
            self.sampler = None

    def select_batch(self, task_sizes):
        if not self.sampler:
            fallback_strat = MLArgmin(self.nodes, self.model, self.poller, self.node_speed)
            return [fallback_strat.select(ts) for ts in task_sizes]
            
        num_tasks = len(task_sizes)
        
        if num_tasks == 0:
            return []
            
        if num_tasks == 1:
            strat_single = MLQubo(self.nodes, self.model, self.poller, self.node_speed)
            node, preds = strat_single.select(task_sizes[0])
            with self._pending_lock:
                self._pending[node["id"]] += 1
            return [(node, preds)]

        # Partir lotes demasiado grandes para evitar explosión combinatoria
        if num_tasks > self.MAX_BATCH_SIZE:
            all_decisions = []
            for start in range(0, num_tasks, self.MAX_BATCH_SIZE):
                sub_batch = task_sizes[start:start + self.MAX_BATCH_SIZE]
                sub_decisions = self._solve_batch(sub_batch)
                all_decisions.extend(sub_decisions)
            return all_decisions
        
        return self._solve_batch(task_sizes)

    def _solve_batch(self, task_sizes):
        import dimod
        num_tasks = len(task_sizes)
        num_nodes = len(self.nodes)

        avg_task = sum(task_sizes) / num_tasks
        base_preds = []
        for n in self.nodes:
            stats = self.poller.get_stats(n["id"])
            with self._pending_lock:
                stats["active_requests"] = stats.get("active_requests", 0) + self._pending[n["id"]]
            staleness = self.poller.get_staleness_ms(n["id"])
            staleness = staleness if staleness is not None else 0.0
            feats = build_features(stats, avg_task, self.node_speed[n["id"]], staleness)
            pred = self.model.predict(np.array([feats]))[0]
            base_preds.append(pred)

        linear = {}
        quadratic = {}
        
        # Penalización para restricción "cada tarea va a exactamente 1 nodo"
        P_constraint = max(base_preds) * 10 if base_preds else 100
        
        for i in range(num_tasks):
            for j in range(num_nodes):
                v_ij = i * num_nodes + j
                linear[v_ij] = linear.get(v_ij, 0) - P_constraint
                for k in range(j + 1, num_nodes):
                    v_ik = i * num_nodes + k
                    var1, var2 = min(v_ij, v_ik), max(v_ij, v_ik)
                    quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + 2 * P_constraint
                    
        # Términos de costo: minimizar varianza de carga entre nodos
        for j in range(num_nodes):
            base_j = base_preds[j]
            speed_j = self.node_speed[self.nodes[j]["id"]]
            
            for i in range(num_tasks):
                v_ij = i * num_nodes + j
                W_ij = task_sizes[i] / (speed_j if speed_j > 0 else 1)
                
                cost_lin = 2 * base_j * W_ij + (W_ij ** 2)
                linear[v_ij] = linear.get(v_ij, 0) + cost_lin
                
                for k in range(i + 1, num_tasks):
                    v_kj = k * num_nodes + j
                    W_kj = task_sizes[k] / (speed_j if speed_j > 0 else 1)
                    cost_quad = 2 * W_ij * W_kj
                    
                    var1, var2 = min(v_ij, v_kj), max(v_ij, v_kj)
                    quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + cost_quad

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, dimod.BINARY)
        
        # Más lecturas para lotes más grandes = mejor convergencia
        reads = max(20, num_tasks * 5)
        sampleset = self.sampler.sample(bqm, num_reads=reads)
        best_sample = sampleset.first.sample
        
        decisions = []
        for i in range(num_tasks):
            chosen_j = None
            for j in range(num_nodes):
                v_ij = i * num_nodes + j
                if best_sample.get(v_ij, 0) == 1:
                    chosen_j = j
                    break
            
            # Fallback: si el solver no asignó esta tarea, usar el nodo con menor carga predicha
            if chosen_j is None:
                chosen_j = int(np.argmin(base_preds))
            
            node = self.nodes[chosen_j]
            decisions.append((node, base_preds))
            
            with self._pending_lock:
                self._pending[node["id"]] += 1
                
        return decisions

    def on_complete(self, node_id):
        with self._pending_lock:
            self._pending[node_id] = max(0, self._pending[node_id] - 1)


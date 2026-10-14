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
import copy


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

        avg_ts = avg_task if avg_task > 0 else 1.0
        norm_tasks = [ts / avg_ts for ts in task_sizes]
        
        # FACTOR CLAVE: Fairness Dinámico Resistente al Herd Effect
        # Reducimos la escala de base_preds para evitar explosión numérica en QUBO
        scaled_base = [p / 10.0 for p in base_preds]
        
        linear = {}
        quadratic = {}
        
        # P_constraint debe ser dominante absoluto
        max_base = max(scaled_base) if scaled_base else 1.0
        P_constraint = (max_base ** 2) * num_tasks * 1000 + 10000

        for j in range(num_nodes):
            L_j = scaled_base[j]
            # Extraemos la capacidad física real del nodo (para aislarla del Concept Drift)
            speed_j = self.node_speed[self.nodes[j]["id"]]
            C_j = speed_j if speed_j > 0.1 else 0.1
            
            for i in range(num_tasks):
                v_ij = i * num_nodes + j
                
                # W_ij = impacto de la tarea i en el nodo j basado puramente en capacidad física
                W_ij = (norm_tasks[i] * 10.0) / C_j
                
                # Lineal: penaliza asignar a nodos que YA tienen carga (L_j)
                cost_lin = 2 * L_j * W_ij + (W_ij ** 2)
                linear[v_ij] = linear.get(v_ij, 0) + cost_lin
                linear[v_ij] -= P_constraint
                
                for k in range(j + 1, num_nodes):
                    v_ik = i * num_nodes + k
                    var1, var2 = min(v_ij, v_ik), max(v_ij, v_ik)
                    quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + 2 * P_constraint
                
                for k2 in range(i + 1, num_tasks):
                    v_kj = k2 * num_nodes + j
                    W_kj = (norm_tasks[k2] * 10.0) / C_j
                    
                    # Cuadrático: penaliza colisiones basado en capacidad FÍSICA, no en carga.
                    # Así, aunque L_j sea 0, meter dos tareas a un Raspberry Pi dolerá mucho.
                    cost_quad = 2 * W_ij * W_kj
                    var1, var2 = min(v_ij, v_kj), max(v_ij, v_kj)
                    quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + cost_quad

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, dimod.BINARY)
        
        # Más lecturas para lotes más grandes = mejor convergencia
        reads = max(20, num_tasks * 5)
        sampleset = self.sampler.sample(bqm, num_reads=reads)
        best_sample = sampleset.first.sample
        
        decisions = []
        # Fix 3: Copia mutable de base_preds para actualizar intra-lote
        dynamic_preds = list(base_preds)
        
        for i in range(num_tasks):
            chosen_j = None
            for j in range(num_nodes):
                v_ij = i * num_nodes + j
                if best_sample.get(v_ij, 0) == 1:
                    chosen_j = j
                    break
            
            # Fix 2: Fallback proporcional a capacidad física (en vez de argmin deterministico)
            if chosen_j is None:
                speeds = [self.node_speed[n["id"]] for n in self.nodes]
                total_speed = sum(speeds)
                probs = [s / total_speed for s in speeds]
                chosen_j = random.choices(range(num_nodes), weights=probs, k=1)[0]
            
            node = self.nodes[chosen_j]
            decisions.append((node, list(dynamic_preds)))
            
            # Fix 3: Actualizar predicciones intra-lote para que las siguientes
            # tareas "vean" que este nodo ya recibió carga adicional
            speed_j = self.node_speed[node["id"]]
            impact = (norm_tasks[i] * avg_ts) / max(speed_j, 0.1)
            dynamic_preds[chosen_j] += impact
            
            with self._pending_lock:
                self._pending[node["id"]] += 1
                
        return decisions

    def on_complete(self, node_id):
        with self._pending_lock:
            self._pending[node_id] = max(0, self._pending[node_id] - 1)



class MLQuboBatchLight(MLQuboBatch):
    """Variante ultra-ligera de ML+QUBO: num_reads=1, num_sweeps=10.
    Objetivo: aislar cuánto del overhead viene del solver SA vs del ML."""

    def __init__(self, nodes, model, poller, node_speed, model_name="catboost"):
        super().__init__(nodes, model, poller, node_speed, model_name=model_name)
        self.name = f"ml_qubo_light_{model_name}"

    def _solve_batch(self, task_sizes):
        import dimod
        import numpy as np
        num_tasks = len(task_sizes)
        num_nodes = len(self.nodes)

        avg_task = sum(task_sizes) / num_tasks if num_tasks > 0 else 1.0
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

        norm_tasks = [ts / avg_task for ts in task_sizes]
        
        linear = {}
        quadratic = {}
        
        P_constraint = max(base_preds) * num_tasks * 1000 if base_preds else 10000

        for j in range(num_nodes):
            base_j = base_preds[j]
            for i in range(num_tasks):
                v_ij = i * num_nodes + j
                W_ij = norm_tasks[i] * 10.0
                
                cost_lin = 2 * base_j * W_ij + (W_ij ** 2)
                linear[v_ij] = linear.get(v_ij, 0) + cost_lin
                linear[v_ij] -= P_constraint
                
                for k in range(j + 1, num_nodes):
                    v_ik = i * num_nodes + k
                    var1, var2 = min(v_ij, v_ik), max(v_ij, v_ik)
                    quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + 2 * P_constraint
                
                for k2 in range(i + 1, num_tasks):
                    v_kj = k2 * num_nodes + j
                    W_kj = norm_tasks[k2] * 10.0
                    cost_quad = 2 * W_ij * W_kj
                    var1, var2 = min(v_ij, v_kj), max(v_ij, v_kj)
                    quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + cost_quad
        
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, dimod.BINARY)

        # SA ultra-ligero
        sampleset = self.sampler.sample(bqm, num_reads=1, num_sweeps=10)
        best_sample = sampleset.first.sample

        decisions = []
        dynamic_preds = list(base_preds)
        n_fallback = 0
        for i in range(num_tasks):
            chosen_j = None
            for j in range(num_nodes):
                v_ij = i * num_nodes + j
                if best_sample.get(v_ij, 0) == 1:
                    chosen_j = j
                    break
            # Fix 2: Fallback proporcional
            if chosen_j is None:
                speeds = [self.node_speed[n["id"]] for n in self.nodes]
                total_speed = sum(speeds)
                probs = [s / total_speed for s in speeds]
                chosen_j = random.choices(range(num_nodes), weights=probs, k=1)[0]
                n_fallback += 1
            node = self.nodes[chosen_j]
            decisions.append((node, list(dynamic_preds)))
            # Fix 3: Actualizar predicciones intra-lote
            speed_j = self.node_speed[node["id"]]
            impact = (norm_tasks[i] * avg_task) / max(speed_j, 0.1)
            dynamic_preds[chosen_j] += impact
            with self._pending_lock:
                self._pending[node["id"]] += 1

        if n_fallback > 0:
            print(f"    [QUBO-Light] WARN: {n_fallback}/{num_tasks} tareas usaron fallback")

        return decisions


class PureQuboBatch:
    """Implementación de QUBO SIN Machine Learning."""
    
    MAX_BATCH_SIZE = 8
    
    def __init__(self, nodes, poller, node_speed: dict):
        import threading
        self.nodes = nodes
        self.poller = poller
        self.node_speed = node_speed
        self.name = "pure_qubo_batch"
        
        self._pending = {n["id"]: 0 for n in nodes}
        self._pending_lock = threading.Lock()
        
        try:
            import dimod
            self.sampler = dimod.SimulatedAnnealingSampler()
        except ImportError:
            self.sampler = None

    def select_batch(self, task_sizes):
        if not self.sampler:
            return []
            
        num_tasks = len(task_sizes)
        if num_tasks == 0:
            return []
            
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
        import numpy as np
        num_tasks = len(task_sizes)
        num_nodes = len(self.nodes)

        avg_task = sum(task_sizes) / num_tasks if num_tasks > 0 else 1.0
        norm_tasks = [ts / avg_task for ts in task_sizes]

        # 1. Costos basados en heurística simple (sin ML)
        base_costs = []
        for n in self.nodes:
            stats = self.poller.get_stats(n["id"])
            with self._pending_lock:
                active = stats.get("active_requests", 0) + self._pending[n["id"]]
            speed = self.node_speed[n["id"]]
            # Estimación naive de latencia: (peticiones activas + 1) / velocidad²
            cost = (active + 1) * 10.0 / max(speed ** 2, 0.01)
            base_costs.append(cost)

        # FACTOR CLAVE: Fairness Dinámico Resistente al Herd Effect
        scaled_base = [p / 10.0 for p in base_costs]

        linear = {}
        quadratic = {}
        
        max_base = max(scaled_base) if scaled_base else 1.0
        P_constraint = (max_base ** 2) * num_tasks * 1000 + 10000

        for j in range(num_nodes):
            L_j = scaled_base[j]
            speed_j = self.node_speed[self.nodes[j]["id"]]
            C_j = speed_j if speed_j > 0.1 else 0.1
            
            for i in range(num_tasks):
                v_ij = i * num_nodes + j
                
                W_ij = (norm_tasks[i] * 10.0) / C_j
                
                cost_lin = 2 * L_j * W_ij + (W_ij ** 2)
                linear[v_ij] = linear.get(v_ij, 0) + cost_lin
                linear[v_ij] -= P_constraint
                
                for k in range(j + 1, num_nodes):
                    v_ik = i * num_nodes + k
                    var1, var2 = min(v_ij, v_ik), max(v_ij, v_ik)
                    quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + 2 * P_constraint
                
                for k2 in range(i + 1, num_tasks):
                    v_kj = k2 * num_nodes + j
                    W_kj = (norm_tasks[k2] * 10.0) / C_j
                    
                    cost_quad = 2 * W_ij * W_kj
                    var1, var2 = min(v_ij, v_kj), max(v_ij, v_kj)
                    quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + cost_quad
        
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, dimod.BINARY)
        reads = max(20, num_tasks * 5)
        sampleset = self.sampler.sample(bqm, num_reads=reads)
        best_sample = sampleset.first.sample
        
        decisions = []
        dynamic_costs = list(base_costs)
        
        for i in range(num_tasks):
            chosen_j = None
            for j in range(num_nodes):
                v_ij = i * num_nodes + j
                if best_sample.get(v_ij, 0) == 1:
                    chosen_j = j
                    break
            
            # Fix 2: Fallback proporcional a capacidad física
            if chosen_j is None:
                speeds = [self.node_speed[n["id"]] for n in self.nodes]
                total_speed = sum(speeds)
                probs = [s / total_speed for s in speeds]
                chosen_j = random.choices(range(num_nodes), weights=probs, k=1)[0]
            
            node = self.nodes[chosen_j]
            decisions.append((node, list(dynamic_costs)))
            
            # Fix 3: Actualizar costos intra-lote
            speed_j = self.node_speed[node["id"]]
            impact = (norm_tasks[i] * avg_task) / max(speed_j, 0.1)
            dynamic_costs[chosen_j] += impact
            
            with self._pending_lock:
                self._pending[node["id"]] += 1
                
        return decisions

    def on_complete(self, node_id):
        with self._pending_lock:
            self._pending[node_id] = max(0, self._pending[node_id] - 1)


class PureQuboBatchLight(PureQuboBatch):
    """Variante ultra-ligera de Pure QUBO: num_reads=1, num_sweeps=10."""

    def __init__(self, nodes, poller, node_speed):
        super().__init__(nodes, poller, node_speed)
        self.name = "pure_qubo_light"

    def _solve_batch(self, task_sizes):
        import dimod
        import numpy as np
        num_tasks = len(task_sizes)
        num_nodes = len(self.nodes)

        avg_task = sum(task_sizes) / num_tasks if num_tasks > 0 else 1.0
        norm_tasks = [ts / avg_task for ts in task_sizes]

        base_costs = []
        for n in self.nodes:
            stats = self.poller.get_stats(n["id"])
            with self._pending_lock:
                active = stats.get("active_requests", 0) + self._pending[n["id"]]
            speed = self.node_speed[n["id"]]
            cost = (active + 1) * 10.0 / max(speed ** 2, 0.01)
            base_costs.append(cost)

        linear = {}
        quadratic = {}
        
        P_constraint = max(base_costs) * num_tasks * 1000 if base_costs else 10000

        for j in range(num_nodes):
            base_j = base_costs[j]
            for i in range(num_tasks):
                v_ij = i * num_nodes + j
                W_ij = norm_tasks[i] * 10.0
                
                cost_lin = 2 * base_j * W_ij + (W_ij ** 2)
                linear[v_ij] = linear.get(v_ij, 0) + cost_lin
                linear[v_ij] -= P_constraint
                
                for k in range(j + 1, num_nodes):
                    v_ik = i * num_nodes + k
                    var1, var2 = min(v_ij, v_ik), max(v_ij, v_ik)
                    quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + 2 * P_constraint
                
                for k2 in range(i + 1, num_tasks):
                    v_kj = k2 * num_nodes + j
                    W_kj = norm_tasks[k2] * 10.0
                    cost_quad = 2 * W_ij * W_kj
                    var1, var2 = min(v_ij, v_kj), max(v_ij, v_kj)
                    quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + cost_quad
        
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, dimod.BINARY)

        sampleset = self.sampler.sample(bqm, num_reads=1, num_sweeps=10)
        best_sample = sampleset.first.sample

        decisions = []
        dynamic_costs = list(base_costs)
        for i in range(num_tasks):
            chosen_j = None
            for j in range(num_nodes):
                v_ij = i * num_nodes + j
                if best_sample.get(v_ij, 0) == 1:
                    chosen_j = j
                    break
            # Fix 2: Fallback proporcional
            if chosen_j is None:
                speeds = [self.node_speed[n["id"]] for n in self.nodes]
                total_speed = sum(speeds)
                probs = [s / total_speed for s in speeds]
                chosen_j = random.choices(range(num_nodes), weights=probs, k=1)[0]
            node = self.nodes[chosen_j]
            decisions.append((node, list(dynamic_costs)))
            # Fix 3: Actualizar costos intra-lote
            speed_j = self.node_speed[node["id"]]
            impact = (norm_tasks[i] * avg_task) / max(speed_j, 0.1)
            dynamic_costs[chosen_j] += impact
            with self._pending_lock:
                self._pending[node["id"]] += 1

        return decisions

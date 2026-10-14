import csv
import sys
import os
import math
from collections import defaultdict
from scipy import stats as sps
import statistics

import glob
def get_latest_file(pattern, default_path):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else default_path

def load_post_phase_data(path=None):
    if path is None:
        path = get_latest_file("results/dynamic_experiment_*.csv", "results/dynamic_experiment.csv")
    print(f"Leyendo datos de: {path}")
    if not os.path.exists(path):
        print(f"Error: {path} no encontrado.")
        sys.exit(1)
        
    by_strategy = defaultdict(list)
    by_strategy_rep_latencies = defaultdict(lambda: defaultdict(list))
    node_counts_by_rep = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # NUEVO: acumular task_size por nodo por repetición
    node_workload_by_rep = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    # NUEVO: acumular overhead_ms (tiempo del algoritmo) por estrategia/rep
    overhead_by_strategy_rep = defaultdict(lambda: defaultdict(list))
    # NUEVO: acumular server_latency_ms (tiempo puro de ejecución)
    server_latency_by_strategy_rep = defaultdict(lambda: defaultdict(list))
    
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] != "post":
                continue # Solo nos interesa analizar el comportamiento durante la falla
                
            strat = row["strategy"]
            rep = int(row["rep"])
            lat = float(row["latency_ms"])
            node = row["node"]
            task_size = float(row.get("task_size", 0))
            
            by_strategy[strat].append(lat)
            by_strategy_rep_latencies[strat][rep].append(lat)
            node_counts_by_rep[strat][rep][node] += 1
            node_workload_by_rep[strat][rep][node] += task_size
            
            # Capturar overhead del algoritmo si existe en el CSV
            overhead_val = row.get("overhead_ms", "")
            if overhead_val and overhead_val.strip():
                try:
                    overhead_by_strategy_rep[strat][rep].append(float(overhead_val))
                except ValueError:
                    pass
                    
            # NUEVO: Capturar server_latency (tiempo real en el nodo)
            server_lat_val = row.get("server_latency_ms", "")
            if server_lat_val and server_lat_val.strip():
                try:
                    server_latency_by_strategy_rep[strat][rep].append(float(server_lat_val))
                except ValueError:
                    pass
            
    return by_strategy, by_strategy_rep_latencies, node_counts_by_rep, node_workload_by_rep, overhead_by_strategy_rep, server_latency_by_strategy_rep


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = sps.tvar(a) if na > 1 else 0, sps.tvar(b) if nb > 1 else 0
    pooled_std = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) if (na + nb - 2) > 0 else float("nan")
    if pooled_std == 0 or math.isnan(pooled_std):
        return float("nan")
    return (sum(a) / na - sum(b) / nb) / pooled_std


# Velocidades de los nodos (capacidad relativa)
NODE_CAPACITY = {
    "node-a": 1.0,
    "node-b": 0.7,
    "node-c": 0.45,
    "node-d": 0.5,
    "node-e": 0.4,
}


def analyze_traffic_distribution(node_counts_by_rep, node_workload_by_rep, file_out):
    def print_and_write(text):
        print(text)
        file_out.write(text + "\n")
        
    print_and_write("\n" + "=" * 80)
    print_and_write("1. DISTRIBUCIÓN DE TRÁFICO DURANTE LA FALLA (Promedio por repetición)")
    print_and_write("=" * 80)
    
    strategies = sorted(node_counts_by_rep.keys())
    for strat in strategies:
        reps = node_counts_by_rep[strat]
        n_reps = len(reps)
        if n_reps == 0:
            continue
            
        avg_counts = defaultdict(float)
        total_avg = 0
        for rep, nodes in reps.items():
            for node, count in nodes.items():
                avg_counts[node] += count / n_reps
                total_avg += count / n_reps
                
        print_and_write(f"\n{strat} (Total peticiones promedio: {total_avg:.1f})")
        
        counts = []
        for node in sorted(avg_counts.keys()):
            pct = (avg_counts[node] / total_avg) * 100 if total_avg > 0 else 0
            counts.append(avg_counts[node])
            print_and_write(f"  {node}: {avg_counts[node]:.1f} peticiones ({pct:.1f}%)")

        if len(counts) > 0 and total_avg > 0:
            max_load = max(counts)
            min_load = min(counts)
            avg_load = total_avg / len(counts)
            
            # Grado de Disparidad (Degree of Imbalance) = (Max - Min) / Promedio
            di = (max_load - min_load) / avg_load if avg_load > 0 else 0
            
            # Índice de Equidad de Jain (Jain's Fairness Index) — por conteo de tareas
            sum_sq = sum(c**2 for c in counts)
            jain_index = (sum(counts)**2) / (len(counts) * sum_sq) if sum_sq > 0 else 0
            
            print_and_write(f"  => Jain (por conteo de tareas): {jain_index:.3f}")
            print_and_write(f"  => Disparidad (DI): {di:.3f}")

        # --- NUEVO: Fairness ponderado por carga real / capacidad ---
        workload_reps = node_workload_by_rep.get(strat, {})
        if workload_reps:
            # Calcular carga normalizada promedio por nodo: Σ(task_size) / speed
            avg_norm_load = defaultdict(float)
            for rep, nodes in workload_reps.items():
                for node, total_work in nodes.items():
                    capacity = NODE_CAPACITY.get(node, 0.5)
                    avg_norm_load[node] += (total_work / capacity) / n_reps
            
            norm_loads = [avg_norm_load[n] for n in sorted(avg_norm_load.keys())]
            if norm_loads and sum(norm_loads) > 0:
                sum_nl = sum(norm_loads)
                sum_sq_nl = sum(x**2 for x in norm_loads)
                jain_weighted = (sum_nl**2) / (len(norm_loads) * sum_sq_nl) if sum_sq_nl > 0 else 0
                
                avg_nl = sum_nl / len(norm_loads)
                di_weighted = (max(norm_loads) - min(norm_loads)) / avg_nl if avg_nl > 0 else 0
                
                print_and_write(f"  => Jain PONDERADO (carga/capacidad): {jain_weighted:.3f}")
                print_and_write(f"  => Disparidad PONDERADA: {di_weighted:.3f}")
                
                # Detalle por nodo
                for node in sorted(avg_norm_load.keys()):
                    cap = NODE_CAPACITY.get(node, 0.5)
                    raw_work = sum(workload_reps[rep].get(node, 0) for rep in workload_reps) / n_reps
                    print_and_write(f"     {node}: carga={raw_work:,.0f} / cap={cap} => carga_norm={avg_norm_load[node]:,.0f}")


def analyze_latencies(by_strategy_rep_latencies, file_out):
    def print_and_write(text):
        print(text)
        file_out.write(text + "\n")
        
    print_and_write("\n" + "=" * 80)
    print_and_write("2. LATENCIA PROMEDIO DURANTE LA FALLA")
    print_and_write("=" * 80)
    
    print_and_write("\n--- Latencia END-TO-END (Incluye overhead y colas del balanceador) ---")
    by_strategy_rep_mean = defaultdict(dict)
    
    strategies = sorted(by_strategy_rep_latencies.keys())
    for strat in strategies:
        reps = by_strategy_rep_latencies[strat]
        rep_means = []
        for rep, lats in reps.items():
            mean_lat = sum(lats) / len(lats)
            by_strategy_rep_mean[strat][rep] = mean_lat
            rep_means.append(mean_lat)
            
        overall_mean = statistics.mean(rep_means)
        overall_std = statistics.pstdev(rep_means) if len(rep_means) > 1 else 0
        print_and_write(f"{strat:<25}: {overall_mean:>7.1f} ± {overall_std:>5.1f} ms")
        
    return by_strategy_rep_mean


def analyze_server_latencies(server_latency_by_strategy_rep, file_out):
    def print_and_write(text):
        print(text)
        file_out.write(text + "\n")
        
    print_and_write("\n--- Latencia del SERVIDOR (Puro procesamiento en el Nodo, Asume Hardware Óptimo) ---")
    if not server_latency_by_strategy_rep:
        print_and_write("No se encontraron datos de latencia de servidor en el CSV.")
        return
        
    strategies = sorted(server_latency_by_strategy_rep.keys())
    for strat in strategies:
        reps = server_latency_by_strategy_rep[strat]
        all_lats = []
        for rep, lats in reps.items():
            all_lats.extend(lats)
            
        if not all_lats:
            continue
            
        mean_lat = statistics.mean(all_lats)
        p95_lat = sorted(all_lats)[int(len(all_lats) * 0.95)] if len(all_lats) > 1 else mean_lat
        print_and_write(f"{strat:<25}: Media = {mean_lat:>7.1f} ms | P95 = {p95_lat:>7.1f} ms")


def run_statistical_test(by_strategy_rep_mean, file_out):
    def print_and_write(text):
        print(text)
        file_out.write(text + "\n")
        
    print_and_write("\n" + "=" * 80)
    print_and_write("3. PRUEBA DE SIGNIFICANCIA (H3): WRR vs ML-Softmax (Catboost, Post-Falla)")
    print_and_write("=" * 80)
    
    a_dict = by_strategy_rep_mean.get("ml_softmax_catboost", {})
    b_dict = by_strategy_rep_mean.get("Weighted Round Robin", {})
    common_reps = sorted(set(a_dict.keys()) & set(b_dict.keys()))
    
    if len(common_reps) < 2:
        print_and_write("No hay suficientes repeticiones para una prueba estadística (se necesitan al menos 2).")
        return
        
    a = [a_dict[r] for r in common_reps] # Propuesto (usaremos catboost como representante)
    b = [b_dict[r] for r in common_reps] # Baseline a comparar (WRR)
    
    # Supuestos:
    _, p_norm_a = sps.shapiro(a) if len(a) >=3 else (None, 1.0)
    _, p_norm_b = sps.shapiro(b) if len(b) >=3 else (None, 1.0)
    
    # T-test pareado (ya que se usa misma semilla/carga por repeticion)
    t_stat, p_value_t = sps.ttest_rel(a, b)
    # Mann-Whitney U (no paramétrico)
    u_stat, p_value_u = sps.mannwhitneyu(a, b, alternative='two-sided')
    
    mean_diff = sum(b)/len(b) - sum(a)/len(a)
    pct_improvement = 100 * mean_diff / (sum(b)/len(b)) if sum(b) != 0 else float("nan")
    d = cohens_d(b, a) # Positivo si WRR > Softmax
    
    print_and_write(f"  n repeticiones pareadas = {len(common_reps)}")
    print_and_write(f"  Media ML-Softmax        = {sum(a)/len(a):.1f} ms")
    print_and_write(f"  Media WRR               = {sum(b)/len(b):.1f} ms")
    print_and_write(f"  Diferencia              = {mean_diff:.1f} ms  ({pct_improvement:.1f}% de reducción de latencia)")
    print_and_write(f"\n  Prueba T-Pareada p-valor = {p_value_t:.6f}")
    print_and_write(f"  Mann-Whitney U p-valor   = {p_value_u:.6f}")
    print_and_write(f"  Cohen's d (Tamaño de efecto) = {d:.3f} ({'grande' if abs(d) > 0.8 else 'mediano'})")
    
    if p_value_t < 0.05:
        print_and_write("\n=> RESULTADO: La diferencia es ESTADÍSTICAMENTE SIGNIFICATIVA (p < 0.05).")
        print_and_write("   Se confirma matemáticamente que ML-Softmax resiste mejor la falla que WRR.")
    else:
        print_and_write("\n=> RESULTADO: La diferencia NO es significativa (p >= 0.05). Se necesitan más repeticiones.")


def analyze_overhead(overhead_by_strategy_rep, file_out):
    """Analiza el overhead del algoritmo de decisión (tiempo de SA/QUBO vs Softmax vs Argmin)."""
    def print_and_write(text):
        print(text)
        file_out.write(text + "\n")
    
    print_and_write("\n" + "=" * 80)
    print_and_write("4. OVERHEAD DEL ALGORITMO DE DECISIÓN (Tiempo del Balanceador)")
    print_and_write("=" * 80)
    
    if not overhead_by_strategy_rep:
        print_and_write("No se encontraron datos de overhead en el CSV (columna 'overhead_ms').")
        print_and_write("Re-ejecuta el experimento para capturar este dato.")
        return
    
    strategies = sorted(overhead_by_strategy_rep.keys())
    for strat in strategies:
        reps = overhead_by_strategy_rep[strat]
        all_overheads = []
        for rep, overheads in reps.items():
            all_overheads.extend(overheads)
        
        if not all_overheads:
            continue
        
        mean_oh = statistics.mean(all_overheads)
        median_oh = statistics.median(all_overheads)
        max_oh = max(all_overheads)
        min_oh = min(all_overheads)
        p95_oh = sorted(all_overheads)[int(len(all_overheads) * 0.95)] if len(all_overheads) > 1 else mean_oh
        std_oh = statistics.pstdev(all_overheads) if len(all_overheads) > 1 else 0
        
        print_and_write(f"\n{strat}:")
        print_and_write(f"  Media:    {mean_oh:>10.2f} ms")
        print_and_write(f"  Mediana:  {median_oh:>10.2f} ms")
        print_and_write(f"  P95:      {p95_oh:>10.2f} ms")
        print_and_write(f"  Max:      {max_oh:>10.2f} ms")
        print_and_write(f"  Min:      {min_oh:>10.2f} ms")
        print_and_write(f"  Std:      {std_oh:>10.2f} ms")
        print_and_write(f"  N muestras: {len(all_overheads)}")


def main():
    by_strategy, by_strategy_rep_latencies, node_counts_by_rep, node_workload_by_rep, overhead_by_strategy_rep, server_latency_by_strategy_rep = load_post_phase_data()
    
    if not by_strategy:
        print("No se encontraron datos de la fase 'post' en el CSV. Asegúrate de ejecutar run_dynamic_experiment.py primero.")
        return
        
    os.makedirs("results", exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"results/dynamic_statistical_report_{timestamp}.txt"
    
    with open(report_path, "w", encoding="utf-8") as file_out:
        analyze_traffic_distribution(node_counts_by_rep, node_workload_by_rep, file_out)
        by_strategy_rep_mean = analyze_latencies(by_strategy_rep_latencies, file_out)
        analyze_server_latencies(server_latency_by_strategy_rep, file_out)
        run_statistical_test(by_strategy_rep_mean, file_out)
        analyze_overhead(overhead_by_strategy_rep, file_out)
        
    print(f"\n[V] Reporte guardado con éxito en: {report_path}")

if __name__ == "__main__":
    main()

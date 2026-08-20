import csv
import sys
import os
import math
from collections import defaultdict
from scipy import stats as sps
import statistics

def load_post_phase_data(path="results/dynamic_experiment.csv"):
    if not os.path.exists(path):
        print(f"Error: {path} no encontrado.")
        sys.exit(1)
        
    by_strategy = defaultdict(list)
    by_strategy_rep_latencies = defaultdict(lambda: defaultdict(list))
    node_counts_by_rep = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] != "post":
                continue # Solo nos interesa analizar el comportamiento durante la falla
                
            strat = row["strategy"]
            rep = int(row["rep"])
            lat = float(row["latency_ms"])
            node = row["node"]
            
            by_strategy[strat].append(lat)
            by_strategy_rep_latencies[strat][rep].append(lat)
            node_counts_by_rep[strat][rep][node] += 1
            
    return by_strategy, by_strategy_rep_latencies, node_counts_by_rep


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = sps.tvar(a) if na > 1 else 0, sps.tvar(b) if nb > 1 else 0
    pooled_std = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) if (na + nb - 2) > 0 else float("nan")
    if pooled_std == 0 or math.isnan(pooled_std):
        return float("nan")
    return (sum(a) / na - sum(b) / nb) / pooled_std


def analyze_traffic_distribution(node_counts_by_rep, file_out):
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
        for node in sorted(avg_counts.keys()):
            pct = (avg_counts[node] / total_avg) * 100 if total_avg > 0 else 0
            print_and_write(f"  {node}: {avg_counts[node]:.1f} peticiones ({pct:.1f}%)")


def analyze_latencies(by_strategy_rep_latencies, file_out):
    def print_and_write(text):
        print(text)
        file_out.write(text + "\n")
        
    print_and_write("\n" + "=" * 80)
    print_and_write("2. LATENCIA PROMEDIO DURANTE LA FALLA")
    print_and_write("=" * 80)
    
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


def run_statistical_test(by_strategy_rep_mean, file_out):
    def print_and_write(text):
        print(text)
        file_out.write(text + "\n")
        
    print_and_write("\n" + "=" * 80)
    print_and_write("3. PRUEBA DE SIGNIFICANCIA (H3): WRR vs ML-Softmax (Post-Falla)")
    print_and_write("=" * 80)
    
    a_dict = by_strategy_rep_mean.get("ML-Softmax", {})
    b_dict = by_strategy_rep_mean.get("Weighted Round Robin", {})
    common_reps = sorted(set(a_dict.keys()) & set(b_dict.keys()))
    
    if len(common_reps) < 2:
        print_and_write("No hay suficientes repeticiones para una prueba estadística (se necesitan al menos 2).")
        return
        
    a = [a_dict[r] for r in common_reps] # Propuesto
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


def main():
    by_strategy, by_strategy_rep_latencies, node_counts_by_rep = load_post_phase_data()
    
    if not by_strategy:
        print("No se encontraron datos de la fase 'post' en el CSV. Asegúrate de ejecutar run_dynamic_experiment.py primero.")
        return
        
    os.makedirs("results", exist_ok=True)
    report_path = "results/dynamic_statistical_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as file_out:
        analyze_traffic_distribution(node_counts_by_rep, file_out)
        by_strategy_rep_mean = analyze_latencies(by_strategy_rep_latencies, file_out)
        run_statistical_test(by_strategy_rep_mean, file_out)
        
    print(f"\n[V] Reporte guardado con éxito en: {report_path}")

if __name__ == "__main__":
    main()

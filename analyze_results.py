"""
Análisis estadístico de los resultados del experimento comparativo.

Corre después de run_experiment.py (que genera results/raw_requests.csv
y results/comparison_repeated.json).

Incluye:
  1. Verificación de supuestos: normalidad (Shapiro-Wilk) y homogeneidad
     de varianza (Levene), sobre las latencias a nivel de petición.
  2. ANOVA de una vía (one-way) sobre las 4 estrategias, a nivel de
     petición individual (mayor poder estadístico, todas las peticiones
     agrupadas por estrategia).
  3. Prueba post-hoc de Tukey HSD, si el ANOVA resulta significativo,
     para saber EXACTAMENTE qué pares de estrategias difieren.
  4. Prueba t PAREADA (H3, la comparación central de la tesis): ML-softmax
     (propuesto) vs. ML-argmin (baseline), usando las medias de latencia
     POR REPETICIÓN. Se usa la versión pareada porque ambas estrategias
     reciben la misma secuencia de tareas dentro de cada repetición
     (misma semilla), lo que controla la variabilidad entre corridas y
     da más poder estadístico que una prueba no pareada.
  5. Tamaño del efecto (Cohen's d) para la comparación central.
  6. Un boxplot comparando las 4 distribuciones de latencia.

Uso:
    python3 analyze_results.py
"""
import os
import json
import csv
import math
import glob
from collections import defaultdict

def get_latest_file(pattern, default_path):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else default_path

from scipy import stats as sps

STRATEGY_ORDER = ["round_robin", "weighted_round_robin", "least_connection", "power_of_two_choices", "ml_argmin_baseline", "ml_softmax_propuesto"]
STRATEGY_LABELS = {
    "round_robin": "Round Robin",
    "weighted_round_robin": "Weighted Round Robin",
    "least_connection": "Least Connection",
    "power_of_two_choices": "Power of Two Choices",
    "ml_argmin_baseline": "ML-argmin (baseline)",
    "ml_softmax_propuesto": "ML-softmax (propuesto)",
}


def load_raw(path=None):
    if path is None:
        path = get_latest_file("results/raw_requests_*.csv", "results/raw_requests.csv")
    print(f"Leyendo datos crudos de: {path}")
    by_strategy = defaultdict(list)
    by_strategy_rep = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ok"] not in ("True", "1", "true"):
                continue
            strat = row["strategy"]
            rep = int(row["rep"])
            lat = float(row["latency_ms"])
            by_strategy[strat].append(lat)
            by_strategy_rep[strat][rep].append(lat)
    return by_strategy, by_strategy_rep


def load_summaries(path=None):
    if path is None:
        path = get_latest_file("results/comparison_repeated_*.json", "results/comparison_repeated.json")
    print(f"Leyendo resúmenes de: {path}")
    with open(path) as f:
        runs = json.load(f)
    by_strategy_rep_mean = defaultdict(dict)
    for r in runs:
        if r.get("FAILED_RUN"):
            continue
        by_strategy_rep_mean[r["strategy"]][r["rep"]] = r["latency_mean_ms"]
    return by_strategy_rep_mean


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = sps.tvar(a) if na > 1 else 0, sps.tvar(b) if nb > 1 else 0
    pooled_std = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) if (na + nb - 2) > 0 else float("nan")
    if pooled_std == 0 or math.isnan(pooled_std):
        return float("nan")
    return (sum(a) / na - sum(b) / nb) / pooled_std


def check_assumptions(by_strategy):
    print("\n" + "=" * 80)
    print("1. VERIFICACIÓN DE SUPUESTOS (sobre latencias a nivel de petición)")
    print("=" * 80)
    report = {}
    for strat in STRATEGY_ORDER:
        if strat not in by_strategy or len(by_strategy[strat]) < 3:
            continue
        sample = by_strategy[strat]
        # Shapiro-Wilk pierde potencia/es lento con muestras muy grandes; se
        # toma una submuestra de hasta 500 para el test de normalidad.
        sub = sample[:500]
        w, p_norm = sps.shapiro(sub)
        report[strat] = {"shapiro_w": w, "shapiro_p": p_norm, "n": len(sample)}
        normal = "SÍ" if p_norm > 0.05 else "NO"
        print(f"  {STRATEGY_LABELS[strat]:<26} Shapiro-Wilk p={p_norm:.4f}  ¿normal? {normal}  (n={len(sample)})")

    groups = [by_strategy[s] for s in STRATEGY_ORDER if s in by_strategy and len(by_strategy[s]) > 1]
    if len(groups) >= 2:
        lev_stat, lev_p = sps.levene(*groups)
        homo = "SÍ" if lev_p > 0.05 else "NO"
        print(f"\n  Levene (homogeneidad de varianza): p={lev_p:.4f}  ¿varianzas iguales? {homo}")
        report["levene"] = {"statistic": lev_stat, "p_value": lev_p}

    print("\n  Nota: si la normalidad o la homogeneidad de varianza no se cumplen,")
    print("  considera reportar también la alternativa no paramétrica (Kruskal-Wallis")
    print("  en vez de ANOVA, y Mann-Whitney U en vez de la prueba t).")
    return report


def run_anova(by_strategy):
    print("\n" + "=" * 80)
    print("2. ANOVA DE UNA VÍA (las 4 estrategias, a nivel de petición individual)")
    print("=" * 80)
    groups = [by_strategy[s] for s in STRATEGY_ORDER if s in by_strategy and len(by_strategy[s]) > 1]
    labels_present = [s for s in STRATEGY_ORDER if s in by_strategy and len(by_strategy[s]) > 1]

    f_stat, p_value = sps.f_oneway(*groups)
    sig = "SIGNIFICATIVO (p<0.05)" if p_value < 0.05 else "no significativo (p>=0.05)"
    print(f"  F = {f_stat:.4f}   p = {p_value:.6f}   -> {sig}")

    # Kruskal-Wallis como alternativa no paramétrica de respaldo
    h_stat, kw_p = sps.kruskal(*groups)
    print(f"  (Kruskal-Wallis no paramétrico: H={h_stat:.4f}, p={kw_p:.6f})")

    result = {"f_statistic": f_stat, "p_value": p_value, "significant": p_value < 0.05,
              "kruskal_h": h_stat, "kruskal_p": kw_p, "groups": labels_present}

    if p_value < 0.05:
        print("\n  --- Post-hoc Tukey HSD (qué pares difieren específicamente) ---")
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            all_vals, all_labels = [], []
            for s in labels_present:
                all_vals.extend(by_strategy[s])
                all_labels.extend([STRATEGY_LABELS[s]] * len(by_strategy[s]))
            tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
            print(tukey.summary())
            result["tukey_summary"] = str(tukey.summary())
        except ImportError:
            print("  (statsmodels no disponible; instala con: pip install statsmodels)")
    return result


def run_key_comparison(by_strategy_rep_mean):
    """La comparación central de la tesis (H3): ML-softmax vs ML-argmin,
    usando la media de latencia POR REPETICIÓN (prueba t pareada, ya que
    ambas estrategias corrieron con la misma secuencia de tareas en cada
    repetición)."""
    print("\n" + "=" * 80)
    print("3. COMPARACIÓN CENTRAL DE LA TESIS (H3): ML-softmax vs. ML-argmin")
    print("   Prueba t PAREADA sobre la media de latencia por repetición")
    print("=" * 80)

    a_dict = by_strategy_rep_mean.get("ml_softmax_propuesto", {})
    b_dict = by_strategy_rep_mean.get("ml_argmin_baseline", {})
    common_reps = sorted(set(a_dict.keys()) & set(b_dict.keys()))

    if len(common_reps) < 2:
        print("  No hay suficientes repeticiones completas en ambas estrategias "
              "para una prueba t pareada (se necesitan al menos 2, se recomiendan 10+).")
        return None

    a = [a_dict[r] for r in common_reps]
    b = [b_dict[r] for r in common_reps]

    t_stat, p_value = sps.ttest_rel(a, b)
    diff = [bi - ai for ai, bi in zip(a, b)]  # positivo = softmax mejor (menor latencia)
    mean_diff = sum(diff) / len(diff)
    pct_improvement = 100 * mean_diff / (sum(b) / len(b)) if sum(b) != 0 else float("nan")

    d = cohens_d(b, a)  # d>0 indica que argmin (b) tiene mayor latencia que softmax (a)

    print(f"  n repeticiones pareadas = {len(common_reps)}")
    print(f"  Media ML-softmax  = {sum(a)/len(a):.1f} ms")
    print(f"  Media ML-argmin   = {sum(b)/len(b):.1f} ms")
    print(f"  Diferencia media  = {mean_diff:.1f} ms  ({pct_improvement:.1f}% de mejora)")
    print(f"  t = {t_stat:.4f}   p = {p_value:.6f}")
    print(f"  Cohen's d = {d:.3f}  ({'grande' if abs(d) > 0.8 else 'mediano' if abs(d) > 0.5 else 'pequeño'})")

    if len(common_reps) < 10:
        print(f"\n  *** ADVERTENCIA: solo {len(common_reps)} repeticiones. Tu metodología")
        print("  especifica un mínimo razonable para esta prueba (recomendado: 10+).")
        print("  Este resultado es orientativo, no concluyente todavía.")

    sig = "SIGNIFICATIVO" if p_value < 0.05 else "NO significativo"
    print(f"\n  => Con estos datos, la diferencia es {sig} al nivel alpha=0.05.")

    return {
        "n_reps": len(common_reps), "mean_softmax_ms": sum(a) / len(a), "mean_argmin_ms": sum(b) / len(b),
        "mean_diff_ms": mean_diff, "pct_improvement": pct_improvement,
        "t_statistic": t_stat, "p_value": p_value, "cohens_d": d, "significant": p_value < 0.05,
    }


def make_boxplot(by_strategy, out_path=None):
    if out_path is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"results/latency_boxplot_{timestamp}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib no disponible; se omite el gráfico. pip install matplotlib)")
        return

    labels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER if s in by_strategy]
    data = [by_strategy[s] for s in STRATEGY_ORDER if s in by_strategy]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Latencia (ms)")
    ax.set_title("Distribución de latencia por estrategia de balanceo de carga")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nGráfico guardado en: {out_path}")


def main():
    by_strategy, by_strategy_rep = load_raw()
    by_strategy_rep_mean = load_summaries()

    check_assumptions(by_strategy)
    anova_result = run_anova(by_strategy)
    key_result = run_key_comparison(by_strategy_rep_mean)
    make_boxplot(by_strategy)

    report = {"anova": anova_result, "key_comparison_h3": key_result}
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"results/statistical_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"Reporte completo guardado en: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

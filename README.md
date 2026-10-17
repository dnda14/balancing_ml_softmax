# Balanceador de Carga Predictivo

Este repositorio contiene la implementación, el banco de pruebas experimental y los análisis estadísticos del balanceador de carga predictivo desarrollado como trabajo de investigación de tesis. El proyecto aborda y soluciona el problema de saturación causado por el **Efecto Manada** (*Herd Effect*) y la degradación por fluctuaciones de hardware (**Concept Drift**) en clústeres heterogéneos de servidores web.

El sistema evoluciona desde las heurísticas clásicas hacia enfoques predictivos basados en Machine Learning (CatBoost, LightGBM, XGBoost), formulaciones probabilísticas (ML-Softmax) y optimización combinatoria en micro-lotes mediante modelos cuadráticos **QUBO** (*Quantum Unconstrained Binary Optimization*) resueltos vía *Simulated Annealing*.

---

## 🎯 Objetivos de la Investigación

1. **Minimizar la latencia media y de cola ($P_{95}, P_{99}$)** en clústeres con capacidad de cómputo heterogénea.
2. **Mitigar el Efecto Manada (*Herd Effect*)**: Evitar que algoritmos determinísticos (*argmin* o *greedy*) colapsen el nodo aparentemente más rápido debido a información de telemetría ligeramente desactualizada (*staleness*).
3. **Resiliencia ante *Concept Drift* (Fallas en Caliente y Throttling)**: Eliminar variables estáticas de hardware en el modelo predictivo, apoyándose exclusivamente en telemetría de congestión dinámica para permitir la auto-recuperación (*self-healing*) ante degradaciones térmicas o "vecinos ruidosos".
4. **Optimización Global por Micro-Lotes (Micro-Batching)**: Agrupar peticiones en ventanas ultracortas (50 ms) para resolver la asignación óptima de carga mediante modelos cuadráticos QUBO.

---

## 📁 Estructura del Repositorio

```text
lb_thesis/
├── balancers/                   # Estrategias de balanceo de carga
│   └── strategies.py            # RR, WRR, Least Connection, P2C, ML-Argmin, ML-Softmax, ML-QUBO
├── common/                      # Módulos transversales y telemetría
│   ├── config.py                # Configuración de red, nodos y capacidades nominales
│   ├── docker_workers.py        # Orquestación y ciclo de vida de contenedores Docker
│   ├── local_workers.py         # Soporte de workers para pruebas locales sin Docker
│   └── stats_poller.py          # Hilo de sondeo asíncrono y concurrente de métricas (/stats)
├── worker/                      # Servicio web que simula procesamiento intensivo de CPU
│   ├── Dockerfile               # Imagen base del worker en Python
│   ├── app.py                   # API Flask + Gunicorn con endpoints /work y /stats
│   └── requirements.txt         # Dependencias del contenedor worker
├── loadgen/                     # Generador de tráfico sintético
│   ├── generator.py             # Generación de tareas con tamaños log-normales
│   └── client.py                # Inyector de carga HTTP concurrente y micro-batching
├── data/                        # Datasets y modelos de inferencia entrenados
│   ├── training_data.json       # Dataset recolectado bajo régimen nominal y falla
│   ├── model.cbm                # Modelo CatBoost principal serializado
│   ├── model_lightgbm.txt       # Modelo LightGBM para comparativas
│   ├── model_xgboost.json       # Modelo XGBoost para comparativas
│   └── train_metrics.json       # Métricas de entrenamiento y validación (R², RMSE)
├── results/                     # Datasets de salida, reportes estadísticos y gráficos
│   ├── raw_requests.csv         # Registro detallado de peticiones individuales
│   ├── dynamic_experiment.csv   # Telemetría de series de tiempo con inyección de falla
│   ├── dynamic_statistical_report.txt # Pruebas ANOVA, T-Tests pareados y efecto Cohen's d
│   └── dynamic_experiment_plot.png    # Gráficos comparativos de respuesta ante fallas
├── analysis/                    # Scripts complementarios de visualización
│   └── generate_dynamic_plots.py# Generación de gráficos detallados de series de tiempo
├── calibrate.py                 # Calibrador de capacidad de CPU del host
├── train_model.py               # Pipeline de recolección de datos y entrenamiento ML
├── run_experiment.py            # Ejecución de experimentos en entorno estable (Régimen Base)
├── run_dynamic_experiment.py    # Ejecución de experimentos con inyección de falla (Concept Drift)
├── analyze_results.py           # Análisis estadístico del experimento base
├── analyze_dynamic_results.py   # Análisis estadístico del experimento dinámico con fallback
├── docker-compose.yml           # Definición del clúster heterogéneo con cgroups
├── requirements.txt             # Dependencias del balanceador y análisis
├── ARCHITECTURE.md              # Especificación técnica detallada de la arquitectura QUBO + ML
└── METODOLOGIA_PROPUESTA_METRICAS.md # Justificación metodológica, variables y métricas
```

---

## 🧠 Estrategias de Balanceo Evaluadas

| Grupo | Estrategia | Descripción |
| :--- | :--- | :--- |
| **Heurísticas Clásicas** | `RoundRobin` | Asignación secuencial cíclica sin estado. |
| | `WeightedRoundRobin` (WRR) | Asignación proporcional a pesos nominales fijos (estándar de la industria). |
| | `LeastConnection` | Asignación al nodo con menor número de peticiones activas. |
| | `PowerOfTwoChoices` (P2C) | Selección aleatoria de 2 nodos y despacho al de menor carga. |
| **Baseline Predictivo** | `MLArgmin` | Réplica de la literatura reciente (Rahimov & Aghayev, 2026). Selección determinística del mínimo predicho por ML. |
| **Propuesta 1 (Probabilística)** | `MLSoftmax` | Selección estocástica con distribución Softmax sobre latencias estimadas, temperatura adaptativa y contador local de peticiones en vuelo (`_pending`). |
| **Propuesta 2 (Optimización Combinatoria)** | `MLQuboBatch` | Agrupamiento en micro-lotes (50 ms) y resolución de asignación óptima cuadrática mediante *Simulated Annealing* (`dimod`). Minimiza la varianza y colisiones en nodos de baja capacidad. |

---

## 🛠 Stack Tecnológico y Requisitos

* **Lenguaje:** Python 3.10+
* **Framework Web de Workers:** Flask `3.0.3` sobre Gunicorn (8 hilos de atención).
* **Modelos Predictivos (Machine Learning):**
  * `CatBoost 1.2.7` (Modelo principal: inferencia sub-milisegundo gracias a árboles simétricos).
  * `LightGBM 4.5.0` y `XGBoost 2.1.1` (Evaluación comparativa).
* **Optimización Cuadrática (QUBO):**
  * D-Wave `dimod 0.12.18` con *Simulated Annealing Sampler*.
* **Virtualización y Aislamiento:**
  * Docker Engine & Docker Compose con restricciones de kernel (`cgroups`).
  * Nodos heterogéneos configurados:
    * `node-a`: 1.00 vCPU (referencia alta).
    * `node-b`: 0.70 vCPU (capacidad media).
    * `node-c`: 0.45 vCPU (capacidad baja).
* **Análisis Estadístico:**
  * `numpy 1.26.4`, `scipy 1.13.1`, `statsmodels 0.14.2`, `matplotlib 3.9.0`.

---

## 🔬 Metodología y Mitigación del *Concept Drift*

### Exclusión de Variables Estáticas
A diferencia de modelos convencionales que incluyen el identificador o velocidad fija del nodo (`node_speed`) como variable independiente fija, este diseño utiliza exclusivamente variables de congestión dinámicas:
$$\vec{X} = [\text{task\_size}, \text{active\_requests}, \text{avg\_recent\_latency\_ms}, \Delta\text{active}, \Delta\text{latency}, \text{estimated\_service\_time}, \text{staleness\_ms}, \text{node\_utilization}]$$

Si el hardware de un nodo se degrada en caliente (ej. de 1.0 CPU a 0.4 CPU por estrangulamiento térmico), la longitud de cola y la latencia reciente aumentan inmediatamente. El modelo detecta esta saturación dinámicamente sin requerir re-entrenamiento ni sufrir de sesgos estáticos.

---

## 💻 Instrucciones de Reproducción

### 1. Preparación del Entorno
```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd lb_thesis

# Crear y activar entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Despliegue del Clúster Heterogéneo (Docker)
```bash
# Construir y levantar los 3 nodos de cómputo aislados
docker compose up -d --build

# Verificar que los contenedores estén activos y saludables
docker compose ps
```

### 3. Calibración y Entrenamiento del Modelo
```bash
# Calibrar la capacidad de procesamiento del host
python calibrate.py --mode docker

# Recolectar datos bajo régimen normal y con inyección de falla, luego entrenar CatBoost
python train_model.py --mode docker
```

### 4. Ejecución de Experimentos

#### A. Escenario Base (Régimen Estable)
Evalúa las estrategias bajo condiciones normales de operación:
```bash
python run_experiment.py --mode docker --reps 10 --n 150 --rate 14.5
```
Generar análisis estadístico del experimento base (ANOVA y T-Tests):
```bash
python analyze_results.py
```

#### B. Escenario Dinámico (Inyección de Falla / Concept Drift)
Simula la degradación en caliente del nodo principal (`node-a`: de 1.0 CPU a 0.4 CPU) a los 13 segundos de ejecución:
```bash
python run_dynamic_experiment.py
```
Generar reporte estadístico pareado y gráficos de series de tiempo:
```bash
python analyze_dynamic_results.py
```

### 5. Limpieza del Entorno
```bash
docker compose down
```

---

## 📊 Hitos de Desarrollo del Proyecto

Para fines de seguimiento y entregables de avance de tesis, la implementación se estructura en los siguientes hitos:

* **Hito 1 (30% - Fundamentos y Testbed):**
  Despliegue del clúster heterogéneo en Docker (`cgroups`), servicio de telemetría concurrente (`stats_poller.py`), inyector de tráfico sintético (`loadgen/`), script de calibración (`calibrate.py`) y estrategias de línea base (`RoundRobin`, `WeightedRoundRobin`).
* **Hito 2 (60% - Modelado ML y Algoritmos Híbridos):**
  Muestreo activo en dos fases, entrenamiento y benchmarking de modelos (`CatBoost`, `LightGBM`, `XGBoost`), implementación de la estrategia probabilística `MLSoftmax` e integración del solver combinatorio `MLQuboBatch` con micro-batching.
* **Hito 3 (100% - Experimentación Masiva y Validación Estadística):**
  Inyección de fallas dinámicas en caliente (*Concept Drift*), pruebas de hipótesis con significancia estadística (ANOVA de medidas repetidas, T-Test de Student pareado, prueba de rangos de Wilcoxon, cálculo de $d$ de Cohen) y análisis comparativo de overhead computacional.

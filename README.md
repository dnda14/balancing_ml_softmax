# Balanceador de Carga Predictivo — Código de Tesis

Este repositorio contiene la implementación y los experimentos del balanceador de carga predictivo desarrollado como trabajo de tesis. El proyecto extiende los enfoques tradicionales y la literatura reciente (ej. Rahimov & Aghayev, 2026) solucionando el problema de saturación causado por el "efecto manada" y el *Concept Drift* (fluctuaciones de hardware).

## 🎯 Objetivos

1. **Minimizar la latencia** en clústeres heterogéneos de servidores web.
2. **Evitar la saturación (Efecto Manada)** común en algoritmos que seleccionan determinísticamente el "mejor" nodo (argmin) basándose en métricas ligeramente desactualizadas.
3. **Lograr adaptabilidad ante fallas y *Concept Drift*** (ej. vecinos ruidosos, *throttling* de CPU) eliminando dependencias de variables estáticas de hardware y apoyándose 100% en telemetría dinámica concurrente.

## 🚀 Cómo se implementó

La arquitectura del sistema consta de los siguientes componentes principales:

*   **Generador de Carga (`loadgen/`)**: Genera peticiones sintéticas con tamaños de tarea log-normales y simula un tráfico de llegada de alta concurrencia.
*   **Nodos de Cómputo (`worker/`)**: Servicios construidos con Flask y Gunicorn que simulan procesamiento real intensivo en CPU. En modo producción (Docker), la heterogeneidad de los nodos se logra limitando el hardware a nivel del kernel utilizando `cgroups` (ej. 1.0, 0.7 y 0.45 CPUs).
*   **Recolector de Métricas (`common/stats_poller.py`)**: Implementa un hilo de sondeo **concurrente** y asíncrono de métricas (`active_requests`, `avg_recent_latency_ms`) hacia los nodos para reducir la latencia de observación (*staleness*) y mantener la información lo más fresca posible.
*   **Estrategias de Balanceo (`balancers/strategies.py`)**: Implementa cuatro algoritmos para su comparación cruzada:
    *   *Round Robin (RR)*: Asignación cíclica clásica.
    *   *Weighted Round Robin (WRR)*: Asignación basada en pesos fijos proporcionales al tamaño del nodo.
    *   *ML-Argmin*: Modelo predictivo con selección determinística del nodo con menor latencia estimada (Baseline de Machine Learning).
    *   **ML-Softmax (Propuesto)**: Modelo predictivo con selección probabilística que distribuye la carga usando una función Softmax sobre las estimaciones, logrando un balance perfecto entre la explotación de nodos rápidos y la exploración para evitar colapsos.

## 🛠 Herramientas Usadas y Versiones

El proyecto fue desarrollado y probado utilizando el siguiente stack tecnológico:

*   **Lenguaje**: Python 3.10+
*   **Servidor Web / Framework**: Flask `3.0.3` sobre Gunicorn.
*   **Machine Learning**: CatBoost `1.2.7`. Se eligió este algoritmo de *Gradient Boosting* de árboles simétricos (*oblivious trees*) por su **excepcional velocidad de inferencia (microsegundos)**, lo cual es crítico para no agregar sobrecarga (*overhead*) en el enrutamiento de peticiones. Además, domina en robustez ante datos tabulares sin necesidad de enormes volúmenes de datos.
*   **Orquestación**: Docker y Docker Compose (para el asilamiento de nodos y configuración de `cgroups`).
*   **Análisis Estadístico**: `numpy==1.26.4`, `scipy==1.13.1`, `statsmodels==0.14.2`, `matplotlib==3.9.0` (utilizados en `analyze_results.py` para pruebas ANOVA, T-Test pareado, Shapiro-Wilk y cálculos de tamaño de efecto).

## 📊 Escenarios y Resultados

### Escenario Base: Comparación de Estrategias en Entorno Estable
Se evaluaron las cuatro estrategias bajo una carga constante de procesamiento para observar su desempeño general.
*   **WRR y RR**: Mostraron latencias altas promedio (~650-700 ms) al ser "ciegos" y no adaptarse dinámicamente al peso real de las tareas individuales que llegan.
*   **ML-Argmin**: Redujo significativamente la latencia (~268 ms) pero en ocasiones puntuales sufrió episodios de "efecto manada" debido a la saturación simultánea del nodo aparentemente más rápido.
*   **ML-Softmax (Propuesto)**: Alcanzó de manera consistente la mejor latencia media (**~222 ms**), superando al baseline predictivo al distribuir de manera probabilística el tráfico y esparcirlo de manera armónica por todo el clúster.

### Escenario 5: Fluctuaciones Dinámicas de Capacidad (Concept Drift)
Este es el escenario clave de la investigación, diseñado para evaluar la resiliencia del sistema ante un fallo abrupto en la capacidad de hardware de un nodo (*noisy neighbor* o estrangulamiento térmico). En el experimento (`run_dynamic_experiment.py`), a los 13 segundos de ejecución constante, el nodo principal (`node-a`) sufrió una degradación drástica de su CPU (de 1.0 a 0.4 CPUs).

*   **Colapso de la Estrategia Estática (WRR)**: Al seguir dependiendo de pesos fijos (configurados bajo la suposición de que el nodo estaba sano), WRR continuó enviando el 50% de la carga al nodo degradado. Esto resultó en un colapso en cascada de la cola de peticiones, disparando la latencia promedio del sistema a **1,814 ms**.
*   **Adaptación Autónoma (ML-Softmax)**: Gracias al modelo predictivo entrenado exclusivamente con variables de congestión dinámicas y sin dependencia de variables estáticas de hardware (previniendo el sesgo *Concept Drift*), el balanceador detectó el aumento de latencia en milisegundos. Automáticamente castigó las probabilidades de enrutamiento del `node-a`, desviando el tráfico de manera inteligente hacia los nodos sanos. Esto contuvo el fallo y estabilizó la latencia en **531 ms**.

## 🏆 Conclusiones

1.  **Reducción Drástica del Impacto**: La estrategia ML-Softmax propuesta reduce la latencia de impacto frente a fallos parciales de infraestructura en un **70%** (531 ms vs 1,814 ms) en comparación directa con el estándar actual de la industria (WRR).
2.  **Mitigación del Concept Drift**: Eliminar las variables de hardware estáticas en el entrenamiento del modelo y basar las inferencias en telemetría de congestión de alta velocidad permite la creación de un balanceador con capacidades intrínsecas de **Self-Healing** (auto-recuperación).
3.  **Prevención del Efecto Manada**: La introducción de un componente probabilístico (Softmax) acoplado a estadísticas de sondeo concurrente es matemáticamente efectivo para prevenir la sobre-saturación de los nodos, solventando los clásicos cuellos de botella de los enrutadores que utilizan `argmin`.

## 💻 Instrucciones de Reproducción

### 1. Calibración y Entrenamiento (Entorno Docker)
*(Nota: Asegúrate de que los puertos 5001, 5002 y 5003 de tu host estén libres).*
```bash
# Iniciar el clúster heterogéneo
docker compose up -d --build

# Calibrar hardware local y entrenar el modelo
python3 calibrate.py --mode docker
python3 train_model.py --mode docker
```

### 2. Ejecutar Experimentos Estadísticos Base
```bash
# IMPORTANTE: Reemplaza <valor_sugerido> con la tasa arrojada por calibrate.py
python3 run_experiment.py --mode docker --reps 10 --n 150 --rate <valor_sugerido>

# Generar reporte ANOVA y T-Test Pareado
python3 analyze_results.py
```
Los reportes estadísticos y gráficos de caja (*boxplots*) se guardarán en la carpeta `results/`.

### 3. Ejecutar el Escenario Dinámico (Simulación de Fallo / Escenario 5)
```bash
python3 run_dynamic_experiment.py
```
Esto aplicará la falla automatizada y generará la comparativa de series de tiempo `results/dynamic_experiment_plot.png`.

```bash
# Al terminar, limpia los contenedores
docker compose down
```

# Balanceador de carga predictivo — código de tesis

Extiende el enfoque de Rahimov & Aghayev (2026) corrigiendo el efecto de
saturación causado por (a) estadísticas desactualizadas (sondeo secuencial)
y (b) selección determinística (argmin), mediante:

1. **Recolección concurrente de métricas** (`common/stats_poller.py`)
2. **Selección probabilística tipo softmax** (`balancers/strategies.py` → `MLSoftmax`)

## Estructura del proyecto

```
worker/              Servicio Flask+gunicorn que simula un nodo de cómputo
balancers/           Las 4 estrategias: RR, WRR, ML-argmin (baseline), ML-softmax (propuesta)
common/              Config de nodos, sondeo de métricas, runtime local y Docker
loadgen/             Generador de carga sintética (log-normal) + cliente de despacho
train_model.py       Recolecta datos y entrena el modelo CatBoost
run_experiment.py    Corre las 4 estrategias bajo la misma carga y compara resultados
docker-compose.yml   Despliegue con 3 nodos heterogéneos REALES (límites de CPU vía cgroups)
data/                Modelo entrenado (model.cbm) y métricas de entrenamiento
results/             Resultados de las corridas comparativas
```

## ✅ Checklist antes de correr Docker (léelo antes de empezar)

1. **Puertos libres**: confirma que nada esté usando 5001-5003 en tu máquina
   (`lsof -i :5001` o similar). Si corriste el modo local antes, asegúrate
   de haber matado esos procesos.
2. **Corre `calibrate.py` primero**, no asumas que mis valores por defecto
   sirven en tu máquina (mi CPU de prueba es distinta a la tuya):
   ```bash
   docker compose up -d --build
   python3 calibrate.py --mode docker
   ```
   Te va a sugerir un `--rate` razonable y si necesitas ajustar `CYCLES_PER_UNIT`.
3. **`STATS_LATENCY_MS` ya viene configurado** (25ms) en `docker-compose.yml`
   en los 3 servicios — es lo que hace que el sondeo secuencial se atrase
   frente al concurrente. Sin esto, tu comparación central saldría plana
   (ya me pasó a mí y lo corregí). Si quieres experimentar con otros
   valores como parte de un análisis de sensibilidad, ajústalo ahí.
4. **Reentrena el modelo en modo Docker**, no reutilices el `model.cbm` que
   viene en este zip — ese se entrenó en mi entorno local (con throttle de
   software), no con los límites de CPU reales de Docker:
   ```bash
   python3 train_model.py --mode docker
   ```
5. **Ten en cuenta la variabilidad esperada**: con límites de CPU reales
   (cgroups), el throttling no es perfectamente suave como en el modo local
   — el kernel puede pausar el proceso en ráfagas dentro de cada período de
   100ms de cuota CFS. Es normal ver algo más de varianza en las latencias
   individuales que en el modo local; no es un bug.

## Cómo se genera la heterogeneidad entre nodos

- **Modo Docker (recomendado para los resultados finales):** cada contenedor
  tiene un límite de CPU real (`cpus: 1.0 / 0.7 / 0.45` en
  `docker-compose.yml`), aplicado por el kernel vía cgroups. No hay ningún
  truco de software — es heterogeneidad de capacidad real, tal como la
  tendrías con VMs o contenedores de distinto tamaño en producción.
- **Modo local (sin Docker, para pruebas rápidas):** no existe un límite de
  CPU real entre procesos, así que se usa `SOFTWARE_THROTTLE` (ver
  `worker/app.py`) como sustituto artificial. Está claramente separado de
  `NODE_SPEED` (que es solo metadato/feature del modelo) y documentado como
  fallback exclusivo de pruebas — **no uses el modo local para los
  resultados finales de tu tesis**, solo para validar que la lógica corre
  bien antes de invertir tiempo en Docker.

## Opción A — Correr con Docker (para los resultados finales de tu tesis)

```bash
docker compose up -d --build
docker compose ps        # confirma que los 3 nodos estén "healthy"

pip install -r requirements.txt   # en tu máquina, fuera de los contenedores
python3 calibrate.py --mode docker        # NUEVO: calibra --rate a tu hardware
python3 train_model.py --mode docker
python3 run_experiment.py --mode docker --reps 10 --n 150 --rate <valor sugerido>
python3 analyze_results.py

docker compose down      # al terminar
```

Los contenedores quedan expuestos en `localhost:5001/5002/5003`, que es lo
que ya usa `common/config.py` — no necesitas cambiar nada más.

**Nota sobre `/stats` y latencia de red:** con Docker en un solo host, la
latencia real entre contenedores es de ~1-2ms (red bridge interna), mucho
menor que en un despliegue multi-host. Si quieres reproducir de forma más
marcada el efecto de staleness que motivó esta tesis, agrega
`STATS_LATENCY_MS=35` (o el valor que prefieras) a cada servicio en
`docker-compose.yml` y documenta esa decisión metodológica en tu tesis
como una simulación de latencia de red inter-host.

## Opción B — Correr localmente sin Docker (solo para pruebas rápidas)

```bash
pip install -r requirements.txt
python3 train_model.py --mode local
python3 run_experiment.py --mode local --reps 2 --n 120
```

Esto lanza los 3 "nodos" como procesos Python en los puertos 5001-5003.
Sirve para validar que la lógica (estrategias, sondeo, modelo) corre sin
errores antes de invertir tiempo en Docker, pero **no debe usarse como
resultado final** — la heterogeneidad ahí es simulada por software, no real.

## Parámetros importantes a calibrar

- `worker/app.py` → `CYCLES_PER_UNIT`: cuánto trabajo real de CPU representa
  una unidad de tarea. Ajústalo para que la latencia de una sola petición
  quede en un rango realista (decenas a cientos de ms).
- `docker-compose.yml` → `cpus:`: la relación de heterogeneidad real entre
  tus 3 nodos. Ajústala si quieres replicar una relación específica de
  algún paper de referencia.
- `run_experiment.py --rate`: tasa de llegada de peticiones. Debe quedar
  por debajo de la capacidad agregada de los 3 nodos para evitar saturación
  total, pero alta para que la estrategia de balanceo sí importe. Se
  recomienda calibrar con una corrida corta de Round Robin primero.
- `balancers/strategies.py` → `MLSoftmax.temperature_fraction`: controla
  qué tan "agresiva" (cercana a argmin) o "explorativa" es la selección
  probabilística. Vale la pena barrer varios valores como parte de un
  análisis de sensibilidad en tu tesis.

## Análisis estadístico (ANOVA + prueba t pareada)

Después de correr `run_experiment.py` (que ahora también guarda
`results/raw_requests.csv` con cada petición individual), corre:

```bash
python3 analyze_results.py
```

Esto genera:
- **Verificación de supuestos**: normalidad (Shapiro-Wilk) y homogeneidad
  de varianza (Levene), con recomendación de usar Kruskal-Wallis/Mann-Whitney
  si no se cumplen.
- **ANOVA de una vía** sobre las 4 estrategias (a nivel de petición individual,
  máximo poder estadístico), con post-hoc de Tukey HSD si resulta significativo.
- **La prueba central de tu tesis (H3)**: prueba t PAREADA entre ML-softmax
  (propuesto) y ML-argmin (baseline), usando la media de latencia por
  repetición. Es pareada porque ambas estrategias reciben la misma secuencia
  de tareas en cada repetición — eso controla la variabilidad entre corridas
  y da más poder estadístico que una prueba no pareada.
- **Cohen's d** (tamaño del efecto) para esa comparación central.
- Un **boxplot** comparando las 4 distribuciones (`results/latency_boxplot.png`).
- Todo el reporte en `results/statistical_report.json`.

**Ya lo probé con una corrida pequeña (3 repeticiones, 60 peticiones) para
confirmar que el script corre sin errores.** El resultado de esa prueba es
un ejemplo perfecto de por qué necesitas más repeticiones: el ANOVA general
SÍ salió significativo (p=0.0002), pero la comparación específica que más te
importa (softmax vs. argmin) NO fue significativa (p=0.88) con solo 3
repeticiones — en esa corrida en particular, softmax incluso salió
ligeramente peor que argmin (307ms vs 296ms), al revés de una corrida
anterior. Esto no invalida tu hipótesis, pero confirma que necesitas las
10+ repeticiones que ya definiste en tu metodología antes de sacar
conclusiones.


Con 2 repeticiones de 120 peticiones cada una:

| Estrategia | Latencia media | Mediana | StdDev carga |
|---|---|---|---|
| Round Robin | 699.9 ± 354.4 ms | 493.8 ms | 0.00 |
| Weighted RR | 658.1 ± 371.6 ms | 489.5 ms | 11.52 |
| ML-argmin (baseline) | 268.9 ± 32.3 ms | 192.7 ms | 37.70 |
| ML-softmax (propuesto) | **222.3 ± 0.8 ms** | **169.5 ms** | 33.50 |

La propuesta superó al baseline en ambas repeticiones (17% menos latencia
media, mucha menor varianza entre corridas). Es evidencia preliminar
alentadora, pero corresponde a modo local (heterogeneidad simulada por
software, no por Docker) — hay que confirmarlo con el modo Docker real.

## Próximos pasos sugeridos

- [ ] Correr `train_model.py --mode docker` y `run_experiment.py --mode docker` con Docker real
- [ ] Aumentar repeticiones a 10+ (`--reps 10`) para el t-test/ANOVA de tu metodología
- [ ] Barrer `temperature_fraction` (análisis de sensibilidad)
- [ ] Probar con distinta cantidad de nodos (escalabilidad)
- [ ] Graficar latencia vs. tiempo para visualizar el "efecto manada" en argmin

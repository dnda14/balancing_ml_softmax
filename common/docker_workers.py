"""
Modo Docker: a diferencia de common/local_workers.py (que lanza procesos
Python locales), estas funciones asumen que los nodos YA están corriendo
como contenedores reales (vía `docker compose up -d`), y solo verifican
que respondan antes de comenzar el experimento.

Esto es lo que se debe usar para los resultados FINALES de la tesis, ya
que Docker aplica límites de CPU reales (cgroups) y aislamiento real entre
nodos -- a diferencia del modo local, limitado por el GIL de Python.
"""
import time
import subprocess
import requests


# Patrón de nombre de contenedor generado por Docker Compose V2.
# Coincide con el nombre del directorio del proyecto ('lb_thesis') + servicio + índice.
# Si tu directorio se llama diferente, ajusta este prefijo.
COMPOSE_PROJECT = "lb_thesis"


def _container_name(node_id: str) -> str:
    """Deriva el nombre del contenedor Docker a partir del node_id."""
    return f"{COMPOSE_PROJECT}-{node_id}-1"


def ensure_docker_nodes(nodes, timeout_s=30):
    for n in nodes:
        ok = False
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                r = requests.get(n["url"] + "/health", timeout=1.0)
                if r.status_code == 200:
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not ok:
            raise RuntimeError(
                f"El nodo {n['id']} no respondió en {n['url']} tras {timeout_s}s.\n"
                f"¿Corriste 'docker compose up -d --build' antes de este script?\n"
                f"Verifica también que common/config.py apunte a los puertos correctos."
            )
    return None  # no hay 'procs' que gestionar; el ciclo de vida lo maneja Docker


def restart_node_containers(nodes, timeout_s=30):
    """Reinicia los contenedores Docker de cada nodo y espera a que estén sanos.

    A diferencia de reset_all_nodes() (que solo limpia contadores Python vía
    el endpoint /reset) o wait_for_node_recovery() (que sondea latencia con
    un timeout fijo), esta función hace un 'docker restart' real que:
      - Mata el proceso Gunicorn y todos sus hilos worker (zombis incluidos)
      - Limpia colas internas, buffers, y cualquier estado en memoria
      - Arranca un proceso Gunicorn fresco con contadores en cero

    Esto garantiza un estado limpio REAL entre bloques de prueba, sin depender
    de que un backlog posiblemente inestable se vacíe solo.
    """
    for n in nodes:
        cname = _container_name(n["id"])
        print(f"  [restart] Reiniciando contenedor {cname}...")
        try:
            subprocess.run(
                ["docker", "restart", cname],
                check=True, capture_output=True, timeout=30,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"No se pudo reiniciar el contenedor '{cname}': {e.stderr.decode().strip()}"
            )

    # Esperar a que todos los contenedores pasen el healthcheck
    print("  [restart] Esperando a que todos los nodos estén sanos...")
    ensure_docker_nodes(nodes, timeout_s=timeout_s)
    print("  [restart] Todos los nodos listos.")


def reset_all_nodes(nodes):
    for n in nodes:
        try:
            requests.post(n["url"] + "/reset", timeout=2.0)
        except Exception as e:
            print(f"Warning: Failed to reset node {n['id']} at {n['url']}: {e}")


def wait_for_node_recovery(nodes, baseline_latency_ms=200, max_wait_s=30):
    """Resetea contadores Y espera a que los hilos zombie de Gunicorn terminen.

    En lugar de un time.sleep(1) fijo, manda una petición de prueba pequeña
    (task_size=1000) a cada nodo y espera a que la latencia real caiga por
    debajo de 'baseline_latency_ms'. Esto garantiza que no queden hilos
    del experimento anterior compitiendo por CPU con el siguiente algoritmo.
    """
    reset_all_nodes(nodes)

    t0 = time.time()
    for n in nodes:
        while time.time() - t0 < max_wait_s:
            try:
                r = requests.post(
                    n["url"] + "/process",
                    json={"task_size": 1000},
                    timeout=5.0,
                )
                latency = r.json().get("latency_ms", 9999)
                if latency < baseline_latency_ms:
                    break
            except Exception:
                pass
            time.sleep(0.5)

    # Limpiar de nuevo tras las peticiones de sondeo
    reset_all_nodes(nodes)


def noop_teardown(_handle):
    """No hace nada: en modo Docker, los contenedores se detienen con
    'docker compose down', no desde este script."""
    pass

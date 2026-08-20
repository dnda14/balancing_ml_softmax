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
import requests


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


def reset_all_nodes(nodes):
    for n in nodes:
        try:
            requests.post(n["url"] + "/reset", timeout=2.0)
        except Exception as e:
            print(f"Warning: Failed to reset node {n['id']} at {n['url']}: {e}")


def noop_teardown(_handle):
    """No hace nada: en modo Docker, los contenedores se detienen con
    'docker compose down', no desde este script."""
    pass

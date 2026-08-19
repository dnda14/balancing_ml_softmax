FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# IMPORTANTE: --workers 1 (no más). Con varios *workers* (procesos) de
# gunicorn, cada uno tendría su propia copia de las variables globales
# (active_requests, recent_latencies), y el endpoint /stats dejaría de
# reflejar el estado real agregado del nodo -- justo la métrica que esta
# tesis necesita medir con precisión para el balanceador predictivo.
# La concurrencia real se logra con --threads (comparten memoria dentro
# del mismo proceso), y el límite de CPU real lo impone Docker (cgroups)
# vía 'cpus:' en docker-compose.yml, no la cantidad de workers/threads.
ENV PORT=5000
EXPOSE 5000
CMD gunicorn --workers 1 --threads 8 --bind 0.0.0.0:${PORT} --timeout 30 app:app

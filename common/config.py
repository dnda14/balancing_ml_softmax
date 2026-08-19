LOCAL_NODES = [
    {"id": "node-a", "url": "http://127.0.0.1:5001", "speed": 1.0},
    {"id": "node-b", "url": "http://127.0.0.1:5002", "speed": 0.7},
    {"id": "node-c", "url": "http://127.0.0.1:5003", "speed": 0.45},
]

NODE_SPEED = {n["id"]: n["speed"] for n in LOCAL_NODES}

# Pesos enteros proporcionales a la velocidad, para Weighted Round Robin
WRR_WEIGHTS = {"node-a": 10, "node-b": 7, "node-c": 5}

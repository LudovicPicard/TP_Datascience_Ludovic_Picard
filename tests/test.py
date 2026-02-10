import requests

resp = requests.get("http://127.0.0.1:8000/dataset/data/324019fa-3c54-4ad2-a602-d6fa68ac6d0")
print(resp.json())  # Vérifie que "target" est bien là

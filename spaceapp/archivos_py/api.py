import requests

url = 'http://127.0.0.1:8000/meteorito'

params = {
    'lat': -5.853268371806371,
    'lon': 43.25798868518021,
    'diametro': 330,
    'densidad': 1900,
    'velocidad': 8000
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
    data = response.json()
    print(data)
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
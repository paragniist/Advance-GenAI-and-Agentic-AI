import requests

# FastAPI endpoint
url = "http://localhost:8000/generate"

# Data to send
json_data = {
    "prompt": "Hello, how are you?",
    "max_length": 50
}

try:
    response = requests.post(url, json=json_data)

    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
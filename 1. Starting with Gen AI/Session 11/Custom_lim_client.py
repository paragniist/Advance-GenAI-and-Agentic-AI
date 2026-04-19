import requests

url = "http://localhost:8000/geminiask"

json_data = {
    "prompt": "How are you?",
    "max_length": 50
}

response = requests.post(url, json=json_data)

print("Status Code:", response.status_code)

if response.status_code == 200:
    print("Generated Text:")
    print(response.json()["generated_text"])
else:
    print("Error:")
    print(response.text)
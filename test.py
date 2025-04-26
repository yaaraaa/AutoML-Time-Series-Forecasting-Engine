import requests
import json

# Define the API endpoint
url = "http://127.0.0.1:5000/forecast"

# Create the payload with extracted sample data
payload = {
    "dataset_id": 21,
    "values": [
        {"time": "2021-07-01", "value": 0.027637},
        {"time": "2021-07-02", "value": -0.989842},
        {"time": "2021-07-04", "value": -1.044431},
        {"time": "2021-07-05", "value": -0.759343},
        {"time": "2021-07-01", "value": 0.027637},
        {"time": "2021-07-02", "value": -0.989842},
        {"time": "2021-07-04", "value": -1.044431},
        {"time": "2021-07-05", "value": -0.759343},
        {"time": "2021-07-01", "value": 0.027637},
        {"time": "2021-07-02", "value": -0.989842},
        {"time": "2021-07-04", "value": -1.044431},
        {"time": "2021-07-05", "value": -0.759343},
        {"time": "2021-07-01", "value": 0.027637},
        {"time": "2021-07-02", "value": -0.989842},
        {"time": "2021-07-04", "value": -1.044431},
        {"time": "2021-07-05", "value": -0.759343},
        {"time": "2021-07-01", "value": 0.027637},
        {"time": "2021-07-02", "value": -0.989842},
        {"time": "2021-07-04", "value": -1.044431}
    ]
}

headers = {'Content-Type': 'application/json'}
response = requests.post(url, json=payload, headers=headers)

# Print the response
print("Response:", response.json())

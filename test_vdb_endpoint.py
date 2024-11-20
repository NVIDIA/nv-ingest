import requests

# Define the endpoint URL
url = "http://ipp1-3304.ipp1u1.colossus.nvidia.com:7670/v1/query"

# payload = {
#     "model": "nvidia/nv-embedqa-e5-v5",
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the driving factor to Nvidia's increased revenue?"}
#     ],
#     "max_tokens": 100,
#     "temperature": 0.7,
#     "top_p": 1.0
# }

payload = {
    "query": "What is the driving factor to Nvidia's increased revenue?",
    "k": 2,
    "job_id": "_7767189e_9f02_4b45_b7d4_0f4d7c5732d6"
}

# Make a POST request to the endpoint
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise an error for bad status codes
    data = response.json()  # Parse the JSON response

    # Print the formatted response
    print(f"Response: {data}")
    
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

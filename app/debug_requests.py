# Create a file debug_requests.py
import os
import requests
from requests.exceptions import RequestException

# Enable request debugging
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"  # Adjust path if needed
requests_log = True

# Try a simple request with verbose output
try:
    print("Attempting connection to Hugging Face...")
    response = requests.get("https://huggingface.co", timeout=10, verify=True)
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Content preview: {response.text[:100]}...")
except RequestException as e:
    print(f"Request failed: {e}")
    
    # If it's an SSL error, try without verification (for debugging only!)
    try:
        print("\nTrying without SSL verification (debug only)...")
        response = requests.get("https://huggingface.co", timeout=10, verify=False)
        print(f"Non-verified response status: {response.status_code}")
    except RequestException as e2:
        print(f"Non-verified request also failed: {e2}")
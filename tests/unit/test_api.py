# MODIFIED FILE: test_api.py

import requests
import json

# --- CONFIGURATION ---
# The URL of our STANDALONE TEST server
BASE_URL = "http://127.0.0.1:8001"  # <-- CHANGE PORT to 8001

# The endpoint we want to test
ENDPOINT = "/protected" # <-- CHANGE ENDPOINT to /protected

def test_protected_route():
    if TOKEN == "ey...":
        print("❌ ERROR: Please paste your actual authentication token into the TOKEN variable.")
        return

    full_url = f"{BASE_URL}{ENDPOINT}"
    
    headers = {
        "Authorization": f"Bearer {TOKEN}"
    }

    print(f"▶️  Sending GET request to: {full_url}")
    
    try:
        response = requests.post(full_url, headers=headers) # CHANGE to GET

        print(f"\n--- RESULTS ---")
        print(f"Status Code: {response.status_code}")
        
        print("\nResponse Body:")
        try:
            print(json.dumps(response.json(), indent=2))
        except json.JSONDecodeError:
            print(response.text)

        if response.status_code == 200:
            print("\n✅ SUCCESS: Authentication dependency works correctly!")
        else:
            print("\n❌ FAILED: Authentication failed. Check status and response.")

    except requests.exceptions.ConnectionError:
        print(f"\n❌ CONNECTION ERROR: Could not connect. Is 'test_auth_standalone.py' running?")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    test_protected_route()
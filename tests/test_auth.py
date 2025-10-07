# FILE: test_api.py

import requests
import json

# --- CONFIGURATION ---
# The URL of your running backend server
BASE_URL = "http://127.0.0.1:8000"

# The endpoint we want to test
ENDPOINT = "/sessions/"

# ==============================================================================
# IMPORTANT: PASTE YOUR FRESH TOKEN HERE
# Follow the "Logout and Refresh" steps to get a new token from your browser's
# Network Tab and paste it between the quotes.
# ==============================================================================
TOKEN = "eyJhbGciOiJSUzI1NiIsImNhdCI6ImNsX0I3ZDRQRDExMUFBQSIsImtpZCI6Imluc18zMm8ya0NWbTNhelg1bkhtRW5oNjlraEdmbTciLCJ0eXAiOiJKV1QifQ.eyJhenAiOiJodHRwOi8vbG9jYWxob3N0OjMwMDEiLCJleHAiOjE3NTk4NjA0MzksImZ2YSI6WzIsLTFdLCJpYXQiOjE3NTk4NjAzNzksImlzcyI6Imh0dHBzOi8vc3VwcmVtZS1jYXJpYm91LTk1LmNsZXJrLmFjY291bnRzLmRldiIsIm5iZiI6MTc1OTg2MDM2OSwic2lkIjoic2Vzc18zM2tUanl2bWlHeFlldTR4Zm54TUVFTjRLUkUiLCJzdHMiOiJhY3RpdmUiLCJzdWIiOiJ1c2VyXzMybzNvU1FxQk1RNkgwU0VwS0FNaWt3azQ5aSIsInYiOjJ9.0Jjlva1h_b5fZCKb7YpvZzEkev2wsm708CBJmL2Cy3kV0iC0Rl3Vy5gpACEULVsrUeuHtWSlpr2JGiV5BRTLIRpYc0CugnUCuaw-M9ipvCIMRu4KXSITTxPlRC652MiP5_k5oR5H3EB11T4muKNLLbzH3PXlucrcyj758SSzqYoMPbyLwOXlgCX1byC2dsbhQPWWEJBy8nQxwzSJFbd4CrfDxvvrd0_hR_oJfPFPuhsNCbkuLQHGP4c95pupK3K4qn3_2PswYQ7kzZ9T2ebr0N_qzWSPg4bk-w5dhi3U8tKE3dzXONOMd-dlAgo2YVv21glvCgLYw1BiSjgc0ILmNw"  # <--- PASTE YOUR TOKEN HERE

# --- SCRIPT LOGIC ---

def test_create_session():
    """
    Sends an authenticated POST request to create a new chat session.
    """
    if TOKEN == "ey...":
        print("❌ ERROR: Please paste your actual authentication token into the TOKEN variable.")
        return

    full_url = f"{BASE_URL}{ENDPOINT}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    }
    
    payload = {
        "framework": "langchain"
    }

    print(f"▶️  Sending POST request to: {full_url}")
    
    try:
        response = requests.post(full_url, headers=headers, json=payload)

        # Print the results
        print(f"\n--- RESULTS ---")
        print(f"Status Code: {response.status_code}")
        
        print("\nResponse Body:")
        try:
            # Try to print formatted JSON
            print(json.dumps(response.json(), indent=2))
        except json.JSONDecodeError:
            # If response is not JSON, print as text
            print(response.text)

        if response.status_code == 200:
            print("\n✅ SUCCESS: The API call was successful and the session was created.")
        else:
            print("\n❌ FAILED: The API call failed. Check the status code and response body for details.")

    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ CONNECTION ERROR: Could not connect to the server at {BASE_URL}.")
        print("   Is your Uvicorn server running?")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    test_create_session()
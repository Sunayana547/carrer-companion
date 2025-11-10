import requests
import json

def test_crop_yield():
    # Test data
    test_data = {
        "crop": "wheat",
        "temp": 22,
        "rain": 100,
        "fertilizer": 50,
        "nitrogen": 50,
        "phosphorus": 30,
        "potassium": 20
    }
    
    try:
        # Send POST request to the API
        response = requests.post(
            'http://localhost:8000/api/predict',
            json=test_data
        )
        
        # Print the response
        print("Status Code:", response.status_code)
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")

if __name__ == "__main__":
    test_crop_yield()

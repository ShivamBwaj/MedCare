import requests
import json

def test_disease_prediction():
    url = "http://localhost:8000/ai/disease/predict"
    
    # Test data
    test_data = {
        "symptoms": "I have cough and vomiting and chest pain"
    }
    
    try:
        print("Testing disease prediction API...")
        print(f"URL: {url}")
        print(f"Data: {json.dumps(test_data, indent=2)}")
        print("-" * 50)
        
        response = requests.post(url, json=test_data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("-" * 50)
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! Response:")
            print(json.dumps(result, indent=2))
        else:
            print("ERROR Response:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Is it running on localhost:8000?")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_disease_prediction()

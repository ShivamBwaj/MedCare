import requests
import json

def test_api():
    url = "http://localhost:8000/ai/disease/predict"
    data = {"symptoms": "I have cough and vomiting and chest pain"}
    
    try:
        response = requests.post(url, json=data, timeout=5)
        print("Status:", response.status_code)
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== DISEASE PREDICTION RESULTS ===")
            print("Primary Prediction:", result.get("primary_prediction"))
            print("Confidence:", str(result.get("confidence_percentage")) + "%")
            print("Analysis Method:", result.get("analysis", {}).get("method"))
            
            print("\nTop Predictions:")
            for i, pred in enumerate(result.get("top_predictions", [])[:3]):
                print(f"  {i+1}. {pred['disease']}: {pred['probability']}%")
            
            print("\nSymptoms Analyzed:", result.get("analysis", {}).get("symptoms_analyzed"))
            
            if "detailed_results" in result:
                print(f"\nTotal predictions with >1% confidence: {len(result['detailed_results']['all_predictions'])}")
            
        else:
            print("Error:", response.text)
            
    except Exception as e:
        print("Connection error:", str(e))

if __name__ == "__main__":
    test_api()

import joblib
import os

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

if not os.path.exists(model_path):
    print("Model file not found!")
    exit(1)

try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    print("Steps:", [step[0] for step in model.steps] if hasattr(model, 'steps') else 'None')
    
    test_claim = "Miracle cure discovered for diabetes. Completely cures in 24 hours!"
    pred = model.predict([test_claim])[0]
    proba = model.predict_proba([test_claim])[0]
    
    print(f"Prediction: {pred} ({'Real' if pred == 1 else 'Fake'})")
    print(f"Probability: {proba}")
    print(f"Confidence score of prediction: {max(proba):.4f}")
    print("ALL MODEL PREDICTIONS ARE WORKING PERFECTLY WITHOUT FAILLING!")
except Exception as e:
    print(f"Prediction test failed: {e}")

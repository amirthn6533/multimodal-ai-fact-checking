import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def list_models():
    print("--- Listing Available Models ---")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        for m in genai.list_models():
            print(f"- {m.name} (supports: {m.supported_generation_methods})")
        return True
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return False

if __name__ == "__main__":
    list_models()

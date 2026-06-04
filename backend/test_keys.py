import google.generativeai as genai
import requests
import logging

# Load keys from main.py
GEMINI_API_KEY = "AIzaSyAJh4b3DhPwwSjaP5hZYunF_zThEiGcmTI"
GOOGLE_FACT_CHECK_API_KEY = "AIzaSyDqIa5yKoxjyh-QcTnvDHzRgsKXkd1L_Tg"

def test_gemini():
    print("--- Testing Gemini AI ---")
    # Try different model names
    models_to_try = ['gemini-2.0-flash']
    
    for m_name in models_to_try:
        try:
            print(f"Trying model: {m_name}...")
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(m_name)
            response = model.generate_content("Say 'OK'")
            print(f"Success with {m_name}: {response.text}")
            return True
        except Exception as e:
            print(f"Failed with {m_name}: {str(e)}")
    
    return False

def test_fact_check():
    print("\n--- Testing Fact Check API ---")
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": "covid", "key": GOOGLE_FACT_CHECK_API_KEY}
    try:
        res = requests.get(url, params=params)
        if res.status_code == 200:
            print("Fact Check API: Success!")
            return True
        else:
            print(f"Fact Check API Error: {res.status_code} - {res.text}")
            return False
    except Exception as e:
        print(f"Fact Check API Error: {str(e)}")
        return False

if __name__ == "__main__":
    g_ok = test_gemini()
    f_ok = test_fact_check()
    
    if g_ok and f_ok:
        print("\nBOTH KEYS ARE WORKING PERFECTLY!")
    else:
        print("\nSOMETHING IS WRONG. See errors above.")

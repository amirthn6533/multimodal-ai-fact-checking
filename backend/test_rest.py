import os
import requests
import json

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
headers = {'Content-Type': 'application/json'}
data = {
    "contents": [{"parts": [{"text": "Say 'REST API working'"}]}]
}

print("Testing Gemini via REST API...")
response = requests.post(url, headers=headers, data=json.dumps(data))

if response.status_code == 200:
    print("Success!")
    print(response.json()['candidates'][0]['content']['parts'][0]['text'])
else:
    print(f"Error {response.status_code}: {response.text}")

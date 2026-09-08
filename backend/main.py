from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import logging
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
from io import BytesIO
from langdetect import detect
from PIL import Image
from transformers import pipeline
import os
import google.generativeai as genai
import datetime

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------
# AI Setup (Environment Variables)
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")

gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        logging.warning(f"Could not configure Gemini model: {e}")

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="AI Fact Checker ULTIMATE API")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Global State (For Trending)
# -----------------------------
trending_checks = []

# -----------------------------
# Load Models
# -----------------------------
try:
    model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        logging.info("Text model loaded successfully")
    else:
        model = None
        logging.warning("model.pkl not found at path")
except Exception as e:
    logging.error("Error loading model: " + str(e))
    model = None

try:
    image_detector = pipeline(
        "image-classification",
        model="dima806/deepfake_vs_real_image_detection"
    )
    logging.info("Image model loaded successfully")
except Exception as e:
    logging.error("Error loading image model: " + str(e))
    image_detector = None

# -----------------------------
# Helpers
# -----------------------------
def search_google_fact_check(query):
    """Fetch real-time fact-checks from Google Fact Check Explorer API"""
    if not GOOGLE_FACT_CHECK_API_KEY or GOOGLE_FACT_CHECK_API_KEY == "YOUR_GOOGLE_CLOUD_API_KEY_HERE":
        return None
    
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query,
        "key": GOOGLE_FACT_CHECK_API_KEY,
        "languageCode": "en-US",
        "pageSize": 3
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if 'claims' in data:
            return data['claims']
    except Exception as e:
        logging.error(f"Fact Check API Error: {str(e)}")
    return None

def get_citations(text, is_fake, fact_check_data=None):
    """Generate citations using Gemini and search results"""
    context = ""
    if fact_check_data:
        context = "Reference these real fact-check reports: " + str(fact_check_data)
    
    if not is_fake:
        prompt = f"Claim: {text[:200]}\n{context}\nProvide 2-3 credible links or sources that confirm this claim. If real-time sources were provided, use them."
    else:
        prompt = f"Claim: {text[:200]}\n{context}\nThis claim is debunked. Provide 2-3 links or sources that prove it is misinformation."
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini Citation Error: {str(e)}")
        return "Sources not available at this time."

def analyze_with_gemini(text, fact_check_data=None):
    """Analyze claim using Gemini with real-time fact-check context"""
    try:
        lang = detect(text)
        context = ""
        if fact_check_data:
            context = f"\n\nReal-time Fact-Check Data found: {str(fact_check_data)}"
            
        prompt = (
            f"You are a professional fact-checker. The input language is {lang}. "
            f"Analyze the following claim: '{text}'{context}\n\n"
            "Provide a final verdict (Real, Fake, or Mixed) and a detailed explanation based on available evidence."
        )
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini Analysis Error: {str(e)}")
        return "AI analysis failed."

# -----------------------------
# Endpoints
# -----------------------------

class NewsInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "online", "message": "ULTIMATE AI Fact Checker API"}

@app.post("/predict")
async def predict(data: NewsInput):
    if not model:
        raise HTTPException(status_code=500, detail="Text model not loaded")
    
    text = data.text.strip()
    prediction = model.predict([text])[0]
    label = "Fake News ❌" if prediction == 0 else "Real News ✅"
    
    try:
        confidence = float(model.predict_proba([text]).max())
    except:
        confidence = 0.5

    # Real-time search
    fact_check_data = search_google_fact_check(text)
    
    analysis = analyze_with_gemini(text, fact_check_data)
    sources = get_citations(text, prediction == 0, fact_check_data)

    # Save to trending
    res = {"prediction": label, "confidence": confidence, "ai_analysis": analysis, "sources": sources}
    trending_checks.insert(0, {"text": text[:50] + "...", "prediction": label, "time": datetime.datetime.now().strftime("%H:%M")})
    if len(trending_checks) > 20: trending_checks.pop()

    return res

@app.post("/analyze-url")
async def analyze_url(data: dict):
    url = data.get("url")
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        # Simple extraction: title + main paragraphs
        text = soup.title.string + " " + " ".join([p.text for p in soup.find_all('p')[:5]])
        return await predict(NewsInput(text=text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to scrape URL: {str(e)}")

@app.post("/analyze-doc")
async def analyze_doc(file: UploadFile = File(...)):
    content = await file.read()
    text = ""
    
    if file.filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        for page in pdf_reader.pages[:5]: # Limit to 5 pages
            text += page.extract_text()
    elif file.filename.endswith(".docx"):
        doc = docx.Document(BytesIO(content))
        text = " ".join([p.text for p in doc.paragraphs[:50]])
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in document")
        
    return await predict(NewsInput(text=text))

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not image_detector:
        raise HTTPException(status_code=500, detail="Image model not loaded")
    
    image = Image.open(file.file)
    result = image_detector(image)
    return {"prediction": result[0]["label"], "confidence": float(result[0]["score"])}

@app.get("/trending")
async def get_trending():
    return trending_checks

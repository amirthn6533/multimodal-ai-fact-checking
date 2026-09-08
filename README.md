# 🔍 Multimodal AI Fact-Checking & Deepfake Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-Deepfake_Detection-blueviolet?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-Claim_Verification-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**A multimodal machine learning system designed to detect synthetic media (deepfakes), cross-verify text claims against visual evidence, and combat digital misinformation.**

[Overview](#-overview) • [Key Capabilities](#-key-capabilities) • [System Architecture](#-system-architecture) • [Quick Start](#-quick-start) • [Verification Tests](#-verification-tests) • [License](#-license)

</div>

---

## 📌 Overview

With the rapid proliferation of generative artificial intelligence and synthetic media, verifying online claims requires analyzing both textual arguments and associated imagery. **Multimodal AI Fact-Checking** combines computer vision models for visual manipulation forensics with NLP transformers for claim extraction and contextual truthfulness scoring.

---

## 🚀 Key Capabilities

- 🖼️ **Deepfake & Image Forensics:** Inspects images for GAN artifacts, face swapping boundaries, spatial inconsistencies, and EXIF manipulation.
- 📝 **Natural Language Claim Extraction:** Isolates factual claims from raw text paragraphs and evaluates semantic consistency.
- 🔄 **Cross-Modal Consistency Scoring:** Validates whether the visual context directly supports, contradicts, or is unrelated to the written narrative.
- ⚡ **High-Throughput REST API:** FastAPI backend serving model inferences with sub-second latency.
- 🧪 **Comprehensive Automated Test Suite:** Integrated unit tests verifying model loading, image preprocessing, and API endpoints.

---

## 🏗️ System Architecture

```text
┌────────────────────────┐      ┌────────────────────────┐
│      Input Image       │      │       Input Text       │
│  (PNG / JPG / WebP)    │      │    (Claim / News Body) │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
            ▼                               ▼
┌────────────────────────┐      ┌────────────────────────┐
│  Vision Transformer /  │      │  NLP Fact Verification │
│  CNN Deepfake Detector │      │  Transformer Pipeline  │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
            ┌───────────────────────────────┐
            │ Multimodal Cross-Verification │
            │      Fusion & Confidence      │
            └───────────────┬───────────────┘
                            ▼
            ┌───────────────────────────────┐
            │   Verdict: Real / Manipulated │
            │   + Confidence Breakdown %    │
            └───────────────────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- PyTorch & CUDA (optional for GPU acceleration)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/amirthn6533/multimodal-ai-fact-checking.git
cd multimodal-ai-fact-checking

# 2. Navigate to backend
cd backend

# 3. Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the FastAPI server
uvicorn main:app --reload --port 8000
```

---

## 🧪 Verification Tests

Run the built-in test suite to validate model integrity and endpoints:

```bash
# Test model imports and dependencies
python test_imports.py

# Test image preprocessing & manipulation detection
python test_image.py

# Test prediction pipeline
python test_prediction.py

# Test REST API responses
python test_rest.py
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

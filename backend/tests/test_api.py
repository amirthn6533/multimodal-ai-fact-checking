import sys, os
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['status'] == 'online'

def test_empty_predict_rejected():
    response = client.post('/predict', json={})
    assert response.status_code == 422

def test_predict_structure():
    response = client.post('/predict', json={'text': 'Test claim about climate change.'})
    assert response.status_code in [200, 500]

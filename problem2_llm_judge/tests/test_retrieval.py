import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add problem1_rag to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_retrieve_endpoint():
    response = client.post(
        "/retrieve",
        json={
            "question": "What is semantic search?",
            "top_k": 3
        }
    )

    assert response.status_code == 200
    assert isinstance(response.json(), dict)
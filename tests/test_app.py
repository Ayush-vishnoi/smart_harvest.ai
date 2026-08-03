import io
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import app_v2


@pytest.fixture
def client():
    app_v2.app.config.update(TESTING=True)
    return app_v2.app.test_client()


def yield_payload(client):
    options = client.get("/api/options").get_json()
    return {
        "state": options["states"][0], "crop": options["crops"][0],
        "season": options["seasons"][0], "crop_year": 2020, "area": 10,
        "annual_rainfall": 1247.6, "fertilizer": 1444.9, "pesticide": 2.7,
    }


def test_health_reports_models(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["yield_model_loaded"] is True
    assert "disease_model_loaded" in response.json
    assert response.json["database_connected"] is True
    assert response.json["database_backend"] in {"sqlite", "postgresql"}
    assert isinstance(response.json["chatbot_configured"], bool)
    assert isinstance(response.json["chatbot_groq_configured"], bool)
    assert isinstance(response.json["chatbot_pinecone_configured"], bool)
    assert isinstance(response.json["chatbot_missing_environment_variables"], list)


def test_dashboard_renders_analytics(client):
    response = client.get("/dashboard")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Analytics Dashboard" in page
    assert "19,689" in page
    assert "82.68%" in page
    assert 'id="trendChart"' in page
    assert 'id="modelChart"' in page
    assert "Production feature removed" in page


def test_yield_prediction_validates_categories(client):
    payload = yield_payload(client)
    response = client.post("/api/predict/yield", json=payload)
    assert response.status_code == 200
    assert "raw_prediction" in response.json
    assert response.json["yield_prediction"] >= 0

    recent = client.get("/api/recent?limit=1")
    assert recent.status_code == 200
    assert recent.json["recent"][0]["type"] == "yield"
    assert recent.json["recent"][0]["state"] == payload["state"]

    payload["state"] = "not-a-real-state"
    response = client.post("/api/predict/yield", json=payload)
    assert response.status_code == 400


def test_disease_prediction(client):
    from PIL import Image

    image = io.BytesIO()
    Image.new("RGB", (192, 192), color=(80, 140, 60)).save(image, format="JPEG")
    image.seek(0)
    response = client.post(
        "/api/predict/disease",
        data={"image": (image, "leaf.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert response.json["prediction"]["label"]
        assert response.json["top_predictions"]


def test_disease_requires_upload(client):
    response = client.post("/api/predict/disease")
    assert response.status_code in (400, 503)


def test_chat_requires_message(client):
    response = client.post("/chat", json={"history": []})
    assert response.status_code == 400
    assert response.json["error"] == "message is required"


def test_chat_reports_missing_groq_configuration(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.post("/chat", json={"message": "How should I water tomatoes?"})
    assert response.status_code == 503
    assert response.json["error"] == "Chat service is not configured"


def test_chat_success(client, monkeypatch):
    chunks = [
        {"text": "Title: Soil health", "metadata": {"title": "Soil Health"}, "score": 0.9},
        {"text": "Title: Compost", "metadata": {"title": "Soil Health"}, "score": 0.8},
    ]

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Add compost after a soil test."}}]}

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("routes.chatbot.retrieve_relevant_chunks", lambda query, top_k=3: chunks)
    monkeypatch.setattr("routes.chatbot.requests.post", lambda *args, **kwargs: FakeResponse())

    response = client.post(
        "/chat",
        json={"message": "How can I improve soil?", "history": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert response.json == {"reply": "Add compost after a soil test.", "sources": ["Soil Health"]}

    from utils.database import ChatInteraction, database

    with database.sessions() as db_session:
        saved = db_session.query(ChatInteraction).order_by(ChatInteraction.id.desc()).first()
        assert saved.message == "How can I improve soil?"
        assert saved.reply == "Add compost after a soil test."


def test_chat_handles_retrieval_failure(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def fail_retrieval(query, top_k=3):
        raise ConnectionError("pinecone unavailable")

    monkeypatch.setattr("routes.chatbot.retrieve_relevant_chunks", fail_retrieval)
    response = client.post("/chat", json={"message": "Help my crop"})
    assert response.status_code == 502
    assert "error" in response.json


def test_chat_handles_groq_timeout(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("routes.chatbot.retrieve_relevant_chunks", lambda query, top_k=3: [])

    def timeout(*args, **kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setattr("routes.chatbot.requests.post", timeout)
    response = client.post("/chat", json={"message": "What should I plant?"})
    assert response.status_code == 504
    assert "timed out" in response.json["error"].lower()

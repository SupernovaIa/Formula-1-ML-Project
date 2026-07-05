from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_chat_without_api_key_returns_a_clean_500(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.post("/chat", json={"messages": [{"role": "user", "content": "Who won?"}]})

    assert response.status_code == 500
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_chat_with_no_messages_is_422(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    response = client.post("/chat", json={"messages": []})

    assert response.status_code == 422


def test_chat_last_message_not_from_user_is_422(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    response = client.post(
        "/chat", json={"messages": [{"role": "assistant", "content": "Hi there!"}]}
    )

    assert response.status_code == 422

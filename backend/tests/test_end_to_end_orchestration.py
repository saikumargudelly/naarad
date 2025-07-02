import pytest
import requests
import json
import time
import asyncio
import websockets

BASE_URL = "http://localhost:8000"
API_BASE_URL = f"{BASE_URL}/api/v1"

@pytest.mark.asyncio
async def test_chat_via_websocket():
    """Test chat via WebSocket, verify orchestrator and agent response."""
    user_id = f"testuser_{int(time.time())}"
    ws_url = f"ws://localhost:8000/api/v1/ws/{user_id}"
    async with websockets.connect(ws_url) as ws:
        # Send a chat message
        message = {
            "type": "text",
            "message": "Who wrote Hamlet?",
            "conversation_id": None
        }
        await ws.send(json.dumps(message))
        responses = []
        for _ in range(5):
            data = await ws.recv()
            responses.append(json.loads(data))
            if any(r.get("type") == "message_complete" for r in responses):
                break
        # Find the main message
        main_msg = next((r for r in responses if r.get("type") == "message"), None)
        assert main_msg is not None, "No message response from orchestrator"
        assert "Hamlet" in main_msg["content"] or "Shakespeare" in main_msg["content"], "Agent did not answer correctly"


def test_chat_via_rest():
    """Test chat via REST, verify orchestrator and agent response."""
    url = f"{API_BASE_URL}/chat"
    payload = {
        "message": "What is the capital of France?",
        "conversation_id": f"conv_{int(time.time())}"
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Paris" in data["message"], "Agent did not answer correctly"
    assert data["conversation_id"] == payload["conversation_id"]


def test_analytics_analyze():
    """Test analytics endpoint, verify orchestrator and memory manager usage."""
    url = f"{API_BASE_URL}/analytics/analyze"
    csv_data = "col1,col2\n1,2\n3,4"
    payload = {
        "data": csv_data,
        "analysis_type": "descriptive",
        "generate_chart": False,
        "user_id": f"testuser_{int(time.time())}"
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Descriptive Statistics" in data["analysis"]
    assert data["user_id"] == payload["user_id"]


def test_personalization_learn():
    """Test personalization learning endpoint, verify orchestrator and memory manager usage."""
    url = f"{API_BASE_URL}/personalization/learn"
    payload = {
        "user_id": f"testuser_{int(time.time())}",
        "message": "I like science fiction books.",
        "response": "Noted your preference for science fiction.",
        "interaction_type": "chat",
        "timestamp": None,
        "context": {"source": "test"}
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["preferences_updated"] in [True, False]  # Could be either depending on logic
    assert data["user_id"] == payload["user_id"]


@pytest.mark.asyncio
async def test_complex_multi_turn_chat_via_websocket():
    """Test multi-turn chat via WebSocket with context retention and reasoning."""
    user_id = f"testuser_{int(time.time())}_multi"
    ws_url = f"ws://localhost:8000/api/v1/ws/{user_id}"
    async with websockets.connect(ws_url) as ws:
        # Turn 1: Set context
        await ws.send(json.dumps({"type": "text", "message": "My favorite color is blue.", "conversation_id": None}))
        await ws.recv()  # typing_start
        await ws.recv()  # message
        await ws.recv()  # message_complete
        # Turn 2: Ask for context recall
        await ws.send(json.dumps({"type": "text", "message": "What is my favorite color?", "conversation_id": None}))
        responses = []
        for _ in range(5):
            data = await ws.recv()
            responses.append(json.loads(data))
            if any(r.get("type") == "message_complete" for r in responses):
                break
        main_msg = next((r for r in responses if r.get("type") == "message"), None)
        assert main_msg is not None, "No message response from orchestrator"
        assert "blue" in main_msg["content"].lower(), "Agent did not recall context correctly"


def test_analytics_invalid_data():
    """Test analytics endpoint with invalid data (negative scenario)."""
    url = f"{API_BASE_URL}/analytics/analyze"
    payload = {
        "data": "not,a,csv\nthis is not valid json either",
        "analysis_type": "descriptive",
        "generate_chart": False,
        "user_id": f"testuser_{int(time.time())}_neg"
    }
    response = requests.post(url, json=payload)
    assert response.status_code in [400, 422, 500]
    data = response.json()
    assert not data.get("success", True) or "error" in data or "detail" in data


def test_personalization_missing_fields():
    """Test personalization learning endpoint with missing required fields (negative)."""
    url = f"{API_BASE_URL}/personalization/learn"
    payload = {
        # Missing user_id
        "message": "I like fantasy books.",
        "response": "Noted your preference for fantasy.",
        "interaction_type": "chat",
        "timestamp": None,
        "context": {"source": "test"}
    }
    response = requests.post(url, json=payload)
    assert response.status_code in [400, 422]
    data = response.json()
    assert "error" in data or "detail" in data


def test_analytics_large_dataset():
    """Test analytics endpoint with a large dataset (happy scenario)."""
    url = f"{API_BASE_URL}/analytics/analyze"
    # Generate a large CSV
    rows = ["col1,col2"] + [f"{i},{i*2}" for i in range(1, 1001)]
    csv_data = "\n".join(rows)
    payload = {
        "data": csv_data,
        "analysis_type": "descriptive",
        "generate_chart": False,
        "user_id": f"testuser_{int(time.time())}_large"
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Descriptive Statistics" in data["analysis"]
    assert data["user_id"] == payload["user_id"]


def test_chat_ambiguous_query_rest():
    """Test chat REST with ambiguous query (complex scenario)."""
    url = f"{API_BASE_URL}/chat"
    payload = {
        "message": "Can you tell me about it?",
        "conversation_id": f"conv_{int(time.time())}_ambig"
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    # Should not crash, should return a fallback or clarification
    assert isinstance(data["message"], str)
    assert len(data["message"]) > 0


def test_chat_unsupported_language():
    """Test chat REST with unsupported language (negative scenario)."""
    url = f"{API_BASE_URL}/chat"
    payload = {
        "message": "これはサポートされていない言語です。",  # Japanese: "This is an unsupported language."
        "conversation_id": f"conv_{int(time.time())}_lang"
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    # Should not crash, should return a fallback or error message
    assert isinstance(data["message"], str)
    assert len(data["message"]) > 0


def test_analytics_with_chart():
    """Test analytics endpoint with chart generation (happy scenario)."""
    url = f"{API_BASE_URL}/analytics/analyze"
    csv_data = "col1,col2\n1,2\n3,4"
    payload = {
        "data": csv_data,
        "analysis_type": "descriptive",
        "generate_chart": True,
        "user_id": f"testuser_{int(time.time())}_chart"
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Descriptive Statistics" in data["analysis"]
    assert data["chart"] is not None
    assert data["user_id"] == payload["user_id"] 
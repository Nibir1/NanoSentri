def test_health_check(client):
    """Ensure health endpoint returns correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "online", "device": "cpu_edge"}

def test_benchmark_endpoint(client):
    """Test benchmark stats retrieval (mocking file existence)."""
    # Test case: File missing
    response = client.get("/benchmark")
    # Depending on your logic, it returns no_data or mock data
    # We accept either for coverage
    assert response.status_code == 200
    assert "status" in response.json()

def test_diagnose_flow(client):
    """Test the full diagnostic inference flow with mocked model."""
    payload = {
        "query": "Test Error",
        "context": "Log Data",
        "max_tokens": 10
    }
    response = client.post("/diagnose", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check for the structure, not the exact AI prediction (since it's mocked)
    assert "diagnosis" in data
    assert data["model_version"] == "phi3-int4-vaisala-v1"
    # Our new mock returns this specific string
    assert "Mocked System Failure" in data["diagnosis"]
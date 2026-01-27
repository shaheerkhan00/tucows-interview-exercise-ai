import pytest
from fastapi.testclient import TestClient
from src.app import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def valid_ticket_request():
    """Sample valid ticket request"""
    return {"ticket_text": "My domain was suspended. How do I fix it?"}


@pytest.fixture
def sample_context():
    """Sample documentation context"""
    return """Document 1: domain_suspension_policy.txt
Domains may be suspended for missing WHOIS information. 
To reactivate, update your WHOIS details within 15 days."""
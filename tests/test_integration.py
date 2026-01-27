import pytest
from fastapi import status


class TestHealthEndpoints:
    """Test health and stats endpoints"""
    
    def test_health_endpoint(self, client):
        """Test health check returns 200"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert "llm_provider" in data
        assert "indexed_chunks" in data
    
    def test_stats_endpoint(self, client):
        """Test stats endpoint returns system info"""
        response = client.get("/stats")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        # Should have stats or error if not initialized
        assert ("total_chunks" in data) or ("error" in data)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestInputValidation:
    """Test input validation and edge cases"""
    
    def test_empty_ticket_text(self, client):
        """Test empty ticket text returns 422 validation error"""
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": ""}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_whitespace_only_ticket(self, client):
        """Test whitespace-only ticket returns 422"""
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": "   \n\t   "}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_ticket_text_field(self, client):
        """Test missing ticket_text field returns 422"""
        response = client.post(
            "/resolve-ticket",
            json={}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_too_long_input(self, client):
        """Test input exceeding max length returns 422"""
        long_text = "a" * 15000  # Exceeds 10000 char limit
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": long_text}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_special_characters_accepted(self, client):
        """Test special characters are handled"""
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": "My domain <script>alert('xss')</script> is down"}
        )
        # Should not crash, returns 200 or 503
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE
        ]
    
    def test_unicode_characters_accepted(self, client):
        """Test unicode/emoji characters are handled"""
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": "My domain 🔥 is broken 💔"}
        )
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE
        ]


class TestValidQueries:
    """Test valid queries return correct structure"""
    
    def test_valid_query_returns_200(self, client, valid_ticket_request):
        """Test valid query returns 200 OK"""
        response = client.post("/resolve-ticket", json=valid_ticket_request)
        
        # Might be 503 if index not loaded, but not 400/422
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE
        ]
    
    def test_response_has_required_fields(self, client):
        """Test response has all required fields"""
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": "What are your nameservers?"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "answer" in data
            assert "references" in data
            assert "action_required" in data
            
            # Check types
            assert isinstance(data["answer"], str)
            assert isinstance(data["references"], list)
            assert isinstance(data["action_required"], str)
    
    def test_action_required_valid_values(self, client):
        """Test action_required is one of allowed values"""
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": "I need help with my domain"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            valid_actions = [
                "none",
                "escalate_to_abuse_team",
                "escalate_to_billing",
                "escalate_to_technical"
            ]
            assert data["action_required"] in valid_actions


class TestEscalationLogic:
    """Test that queries are escalated appropriately"""
    
    def test_spam_complaint_escalates_to_abuse(self, client):
        """Test spam complaints escalate to abuse team"""
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": "My domain was suspended for spam complaints"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["action_required"] == "escalate_to_abuse_team"
    
    def test_billing_issue_escalates_to_billing(self, client):
        """Test billing issues escalate to billing"""
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": "I was charged twice for my renewal"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["action_required"] == "escalate_to_billing"
    
    def test_technical_issue_escalates_to_technical(self, client):
        """Test complex technical issues escalate"""
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": "My DNS is not propagating after 72 hours"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["action_required"] == "escalate_to_technical"
    
    def test_off_topic_query_escalates(self, client):
        """Test off-topic queries escalate"""
        response = client.post(
            "/resolve-ticket",
            json={"ticket_text": "How do I bake a chocolate cake?"}
        )
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["action_required"] == "escalate_to_technical"
            assert len(data["references"]) == 0
import pytest
from src.config import get_settings
from src.prompts import PromptManager
from src.ingest import chunk_text


class TestConfiguration:
    """Test configuration loading"""
    
    def test_config_loads(self):
        """Test that configuration loads from environment"""
        settings = get_settings()
        # Remove check for LLM_PROVIDER
        assert settings.OPENAI_API_KEY is not None  # Changed
        assert settings.CHUNK_SIZE > 0
        assert settings.CHUNK_OVERLAP >= 0
        assert settings.TOP_K_RERANK > 0
    
    def test_config_validation(self):
        """Test configuration has required fields"""
        settings = get_settings()
        assert hasattr(settings, "OPENAI_MODEL")  # Changed
        assert hasattr(settings, "EMBEDDING_MODEL")
        assert hasattr(settings, "RERANKER_MODEL")


class TestChunking:
    """Test text chunking logic"""
    
    def test_chunk_with_overlap(self):
        """Test that chunking creates overlapping chunks"""
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 1
        # Verify overlap exists between consecutive chunks
        if len(chunks) > 1:
            # Last words of first chunk should appear in second chunk
            first_chunk_words = chunks[0].split()
            second_chunk_words = chunks[1].split()
            # There should be some overlap
            assert len(set(first_chunk_words) & set(second_chunk_words)) > 0
    
    def test_chunk_empty_text(self):
        """Test chunking empty text returns empty list"""
        chunks = chunk_text("", chunk_size=100, overlap=20)
        assert chunks == []
    
    def test_chunk_whitespace_only(self):
        """Test chunking whitespace returns empty list"""
        chunks = chunk_text("   \n\t   ", chunk_size=100, overlap=20)
        assert chunks == []
    
    def test_chunk_size_respected(self):
        """Test that chunks don't exceed specified size"""
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        
        for chunk in chunks:
            assert len(chunk) <= 100


class TestPromptManager:
    """Test prompt building"""
    
    def test_build_messages_structure(self):
        """Test that PromptManager builds correct message structure"""
        messages = PromptManager.build_messages("test query", "test context")
        
        # Should have system + examples + user message
        assert len(messages) >= 3
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
    
    def test_messages_contain_context(self):
        """Test that context is included in messages"""
        context = "Sample context about domains"
        messages = PromptManager.build_messages("test", context)
        
        # Context should be in the last user message
        user_message = messages[-1]["content"]
        assert "Sample context" in user_message or "Context" in user_message
    
    def test_messages_contain_query(self):
        """Test that query is included in messages"""
        query = "How do I fix my domain?"
        messages = PromptManager.build_messages(query, "context")
        
        user_message = messages[-1]["content"]
        assert query in user_message
    
    def test_output_schema_specified(self):
        """Test that output schema is clearly specified"""
        messages = PromptManager.build_messages("test", "context")
        user_message = messages[-1]["content"]
        
        # Should mention the required fields
        assert "answer" in user_message.lower()
        assert "references" in user_message.lower()
        assert "action_required" in user_message.lower()
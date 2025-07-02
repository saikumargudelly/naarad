"""Test suite for voice features with real audio input simulation."""

import pytest
import asyncio
import base64
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from io import BytesIO

from main import app
from agent.agents.voice_agent import VoiceAgent, AgentConfig

client = TestClient(app)

class TestVoiceFeatures:
    """Test class for voice features."""
    
    @pytest.fixture
    def voice_agent(self):
        """Create a test voice agent."""
        return VoiceAgent(AgentConfig(
            name="test_voice_agent",
            description="Test voice agent",
            model_name="llama3-70b-8192"
        ))
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data for testing."""
        # Create a minimal WAV file header (44 bytes)
        wav_header = (
            b'RIFF' +  # Chunk ID
            (36).to_bytes(4, 'little') +  # Chunk size
            b'WAVE' +  # Format
            b'fmt ' +  # Subchunk1 ID
            (16).to_bytes(4, 'little') +  # Subchunk1 size
            (1).to_bytes(2, 'little') +   # Audio format (PCM)
            (1).to_bytes(2, 'little') +   # Number of channels
            (16000).to_bytes(4, 'little') +  # Sample rate
            (32000).to_bytes(4, 'little') +  # Byte rate
            (2).to_bytes(2, 'little') +   # Block align
            (16).to_bytes(2, 'little') +  # Bits per sample
            b'data' +  # Subchunk2 ID
            (4).to_bytes(4, 'little') +   # Subchunk2 size
            b'\x00\x00\x00\x00'  # Sample data (silence)
        )
        return base64.b64encode(wav_header).decode('utf-8')
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for speech synthesis testing."""
        return "Hello, this is a test message for speech synthesis."
    
    def test_voice_health_check(self):
        """Test voice health check endpoint."""
        response = client.get("/api/v1/voice/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert data["service"] == "voice-processing"
    
    def test_get_available_voices(self):
        """Test getting available voices."""
        response = client.get("/api/v1/voice/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert "formats" in data
        assert "default_voice" in data
        assert isinstance(data["voices"], list)
        assert isinstance(data["formats"], list)
    
    @pytest.mark.asyncio
    async def test_voice_agent_initialization(self, voice_agent):
        """Test voice agent initialization."""
        assert voice_agent.name == "test_voice_agent"
        assert voice_agent.description == "Test voice agent"
        assert voice_agent.model_name == "llama3-70b-8192"
        assert hasattr(voice_agent, 'speech_recognition')
        assert hasattr(voice_agent, 'text_to_speech')
    
    # NOTE: Audio transcription features are coming in a future update.
    # The following tests related to audio transcription are commented out until the feature is available.

    # @pytest.mark.asyncio
    # async def test_speech_recognition_simulation(self, voice_agent, sample_audio_data):
    #     """Test speech recognition with simulated audio."""
    #     with patch.object(voice_agent.speech_recognition, '_run') as mock_run:
    #         mock_run.return_value = "Hello, this is a test transcription."
    #         result = voice_agent.speech_recognition._run(sample_audio_data)
    #         assert result == "Hello, this is a test transcription."
    #         mock_run.assert_called_once_with(sample_audio_data)
    
    @pytest.mark.asyncio
    async def test_text_to_speech_simulation(self, voice_agent, sample_text):
        """Test text-to-speech with simulated response."""
        with patch.object(voice_agent.text_to_speech, '_run') as mock_run:
            mock_run.return_value = "data:audio/mp3;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
            
            result = await voice_agent.text_to_speech_response(
                text=sample_text,
                voice="alloy",
                format="mp3"
            )
            
            assert result["success"] is True
            assert "audio_data" in result
            assert result["voice"] == "alloy"
            assert result["format"] == "mp3"
    
    # def test_voice_process_endpoint(self, sample_audio_data):
    #     """Test voice processing endpoint."""
    #     with patch('routers.voice.voice_agent.process_voice_input') as mock_process:
    #         mock_process.return_value = {
    #             "success": True,
    #             "transcribed_text": "Hello, this is a test.",
    #             "response_text": "Hello! I heard you say: Hello, this is a test.",
    #             "audio_response": "data:audio/mp3;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT",
    #             "metadata": {"voice_used": "alloy"}
    #         }
    #         
    #         response = client.post(
    #             "/api/v1/voice/process",
    #             json={
    #                 "audio_data": sample_audio_data,
    #                 "user_id": "test_user",
    #                 "voice_preference": "alloy",
    #                 "generate_audio": True
    #             }
    #         )
    #         
    #         assert response.status_code == 200
    #         data = response.json()
    #         assert data["success"] is True
    #         assert "transcribed_text" in data
    #         assert "response_text" in data
    #         assert "audio_response" in data
    #         assert "processing_time" in data
    
    def test_voice_synthesize_endpoint(self, sample_text):
        """Test speech synthesis endpoint."""
        with patch('routers.voice.voice_agent.text_to_speech_response') as mock_synthesize:
            mock_synthesize.return_value = {
                "success": True,
                "audio_data": "data:audio/mp3;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT",
                "voice": "alloy",
                "format": "mp3",
                "text_length": len(sample_text)
            }
            
            response = client.post(
                "/api/v1/voice/synthesize",
                data={
                    "text": sample_text,
                    "voice": "alloy",
                    "format": "mp3"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "audio_data" in data
            assert data["voice"] == "alloy"
            assert data["format"] == "mp3"
    
    def test_voice_test_endpoint(self, sample_audio_data, sample_text):
        """Test voice testing endpoint."""
        with patch('routers.voice.voice_agent.speech_recognition._run') as mock_recognition:
            with patch('routers.voice.voice_agent.text_to_speech_response') as mock_synthesis:
                with patch('routers.voice.voice_agent.process_voice_input') as mock_process:
                    mock_recognition.return_value = "Test transcription"
                    mock_synthesis.return_value = {
                        "success": True,
                        "audio_data": "data:audio/mp3;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT",
                        "voice": "alloy"
                    }
                    mock_process.return_value = {
                        "success": True,
                        "transcribed_text": "Test transcription",
                        "response_text": "Test response",
                        "audio_response": "data:audio/mp3;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
                    }
                    
                    # Test recognition
                    response = client.post(
                        "/api/v1/voice/test",
                        json={
                            "test_type": "recognition",
                            "audio_data": sample_audio_data
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["test_type"] == "recognition"
                    assert "results" in data
                    assert "recognition" in data["results"]
                    
                    # Test synthesis
                    response = client.post(
                        "/api/v1/voice/test",
                        json={
                            "test_type": "synthesis",
                            "text_data": sample_text
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["test_type"] == "synthesis"
                    assert "results" in data
                    assert "synthesis" in data["results"]
                    
                    # Test full pipeline
                    response = client.post(
                        "/api/v1/voice/test",
                        json={
                            "test_type": "full",
                            "audio_data": sample_audio_data
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["test_type"] == "full"
                    assert "results" in data
                    assert "full_pipeline" in data["results"]
    
    def test_voice_upload_endpoint(self):
        """Test voice upload endpoint."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Write minimal WAV header
            wav_header = (
                b'RIFF' +  # Chunk ID
                (36).to_bytes(4, 'little') +  # Chunk size
                b'WAVE' +  # Format
                b'fmt ' +  # Subchunk1 ID
                (16).to_bytes(4, 'little') +  # Subchunk1 size
                (1).to_bytes(2, 'little') +   # Audio format (PCM)
                (1).to_bytes(2, 'little') +   # Number of channels
                (16000).to_bytes(4, 'little') +  # Sample rate
                (32000).to_bytes(4, 'little') +  # Byte rate
                (2).to_bytes(2, 'little') +   # Block align
                (16).to_bytes(2, 'little') +  # Bits per sample
                b'data' +  # Subchunk2 ID
                (4).to_bytes(4, 'little') +   # Subchunk2 size
                b'\x00\x00\x00\x00'  # Sample data (silence)
            )
            temp_file.write(wav_header)
            temp_file.flush()
            
            with patch('routers.voice.voice_agent.process_voice_input') as mock_process:
                mock_process.return_value = {
                    "success": True,
                    "transcribed_text": "Uploaded audio transcription",
                    "response_text": "Response to uploaded audio",
                    "audio_response": "data:audio/mp3;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
                }
                
                with open(temp_file.name, 'rb') as audio_file:
                    response = client.post(
                        "/api/v1/voice/upload",
                        files={"file": ("test_audio.wav", audio_file, "audio/wav")},
                        data={
                            "user_id": "test_user",
                            "voice_preference": "alloy",
                            "generate_audio": True
                        }
                    )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "transcribed_text" in data
                assert "response_text" in data
                assert "audio_response" in data
                assert data["filename"] == "test_audio.wav"
            
            # Clean up
            os.unlink(temp_file.name)
    
    def test_voice_error_handling(self):
        """Test voice error handling."""
        # Test with invalid audio data
        response = client.post(
            "/api/v1/voice/process",
            json={
                "audio_data": "invalid_base64_data",
                "user_id": "test_user"
            }
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 500]
    
    @pytest.mark.asyncio
    async def test_voice_agent_error_handling(self, voice_agent):
        """Test voice agent error handling."""
        # Test speech recognition error
        with patch.object(voice_agent.speech_recognition, '_run') as mock_run:
            mock_run.side_effect = Exception("Recognition error")
            
            result = voice_agent.speech_recognition._run("invalid_data")
            assert "Error" in result
        
        # Test text-to-speech error
        with patch.object(voice_agent.text_to_speech, '_run') as mock_run:
            mock_run.side_effect = Exception("Synthesis error")
            
            result = await voice_agent.text_to_speech_response(
                text="Test text",
                voice="alloy"
            )
            assert result["success"] is False
            assert "error" in result

if __name__ == "__main__":
    pytest.main([__file__]) 
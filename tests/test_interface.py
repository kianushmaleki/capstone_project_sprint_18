import pytest
from unittest.mock import patch, MagicMock


def test_chat_returns_string():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello!")]

    with patch("src.app.get_client") as mock_client:
        mock_client.return_value.messages.create.return_value = mock_response
        from src.app import chat
        result = chat("Hi")
        assert isinstance(result, str)
        assert result == "Hello!"

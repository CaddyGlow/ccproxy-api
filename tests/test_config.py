"""Test configuration module."""

import os
import tempfile
from pathlib import Path

import pytest

from claude_proxy.config import Settings, get_settings


class TestSettings:
    """Test Settings class."""

    def test_settings_default_values(self):
        """Test that Settings has correct default values."""
        settings = Settings(anthropic_api_key="test-key")
        
        assert settings.anthropic_api_key == "test-key"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.log_level == "INFO"
        assert settings.workers == 1
        assert settings.reload is False
        assert settings.rate_limit_requests == 100
        assert settings.rate_limit_window == 60
        assert settings.request_timeout == 300
        assert settings.cors_origins == ["*"]
        assert settings.config_file is None

    def test_settings_from_env_vars(self):
        """Test loading settings from environment variables."""
        # Set environment variables
        env_vars = {
            "ANTHROPIC_API_KEY": "env-test-key",
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "LOG_LEVEL": "debug",
            "WORKERS": "2",
            "RELOAD": "true",
            "RATE_LIMIT_REQUESTS": "50",
            "CORS_ORIGINS": "https://example.com,https://test.com",
        }
        
        # Temporarily set environment variables
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            settings = Settings()
            
            assert settings.anthropic_api_key == "env-test-key"
            assert settings.host == "127.0.0.1"
            assert settings.port == 9000
            assert settings.log_level == "DEBUG"  # Should be normalized to uppercase
            assert settings.workers == 2
            assert settings.reload is True
            assert settings.rate_limit_requests == 50
            assert settings.cors_origins == ["https://example.com", "https://test.com"]
            
        finally:
            # Restore original environment variables
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_settings_properties(self):
        """Test Settings properties."""
        settings = Settings(anthropic_api_key="test-key", host="localhost", port=8080)
        
        assert settings.server_url == "http://localhost:8080"
        assert settings.is_development is False
        
        # Test development mode detection
        debug_settings = Settings(anthropic_api_key="test-key", log_level="DEBUG")
        assert debug_settings.is_development is True
        
        reload_settings = Settings(anthropic_api_key="test-key", reload=True)
        assert reload_settings.is_development is True

    def test_model_dump_safe(self):
        """Test that model_dump_safe masks sensitive information."""
        settings = Settings(anthropic_api_key="secret-key")
        
        safe_data = settings.model_dump_safe()
        
        assert safe_data["anthropic_api_key"] == "***MASKED***"
        assert safe_data["host"] == "0.0.0.0"
        assert safe_data["port"] == 8000

    def test_config_file_validation(self):
        """Test config file path validation."""
        # Test string path conversion
        settings = Settings(anthropic_api_key="test-key", config_file="config.json")
        assert isinstance(settings.config_file, Path)
        assert settings.config_file == Path("config.json")
        
        # Test Path object
        path_obj = Path("test.json")
        settings = Settings(anthropic_api_key="test-key", config_file=path_obj)
        assert settings.config_file == path_obj
        
        # Test None
        settings = Settings(anthropic_api_key="test-key", config_file=None)
        assert settings.config_file is None

    def test_cors_origins_validation(self):
        """Test CORS origins validation."""
        # Test string input
        settings = Settings(
            anthropic_api_key="test-key",
            cors_origins="https://example.com,https://test.com"
        )
        assert settings.cors_origins == ["https://example.com", "https://test.com"]
        
        # Test list input
        origins_list = ["https://example.com", "https://test.com"]
        settings = Settings(anthropic_api_key="test-key", cors_origins=origins_list)
        assert settings.cors_origins == origins_list

    def test_field_validation(self):
        """Test field validation."""
        # Test port validation
        with pytest.raises(ValueError):
            Settings(anthropic_api_key="test-key", port=0)
            
        with pytest.raises(ValueError):
            Settings(anthropic_api_key="test-key", port=70000)
        
        # Test workers validation
        with pytest.raises(ValueError):
            Settings(anthropic_api_key="test-key", workers=0)
            
        with pytest.raises(ValueError):
            Settings(anthropic_api_key="test-key", workers=50)

    def test_get_settings_function(self):
        """Test get_settings function."""
        # Set a valid API key in environment
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        
        try:
            settings = get_settings()
            assert isinstance(settings, Settings)
            assert settings.anthropic_api_key == "test-key"
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_get_settings_missing_api_key(self):
        """Test get_settings with missing API key."""
        # Ensure API key is not in environment
        os.environ.pop("ANTHROPIC_API_KEY", None)
        
        with pytest.raises(ValueError, match="Configuration error"):
            get_settings()

    def test_dotenv_file_loading(self):
        """Test loading from .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("ANTHROPIC_API_KEY=dotenv-test-key\nPORT=7000\n")
            
            # Change to temp directory to test .env loading
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                settings = Settings()
                assert settings.anthropic_api_key == "dotenv-test-key"
                assert settings.port == 7000
            finally:
                os.chdir(original_cwd)
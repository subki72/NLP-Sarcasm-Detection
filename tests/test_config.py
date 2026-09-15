"""
Unit tests for configuration module and environment variable overrides.
"""

import importlib

from src import config


def test_default_config_values():
    """Verify default paths and hyperparameters."""
    assert "data" in config.DATA_PATH
    assert "models" in config.MODEL_PATH
    assert config.MAX_LENGTH == 128
    assert config.LABEL_MAP == {0: "GENUINE", 1: "SARCASTIC"}


def test_env_var_overrides(monkeypatch):
    """Verify environment variables properly override config constants."""
    custom_path = "/custom/models/path"
    custom_length = "256"

    monkeypatch.setenv("SARCASM_MODEL_PATH", custom_path)
    monkeypatch.setenv("SARCASM_MAX_LENGTH", custom_length)

    importlib.reload(config)

    assert config.MODEL_PATH == custom_path
    assert config.MAX_LENGTH == 256

    # Clean up reload
    monkeypatch.delenv("SARCASM_MODEL_PATH", raising=False)
    monkeypatch.delenv("SARCASM_MAX_LENGTH", raising=False)
    importlib.reload(config)

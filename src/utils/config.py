"""Configuration management for experiments."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List


class Config:
    """Manages experiment configuration and API keys."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml file. If None, uses default.
        """
        # Load environment variables
        env_path = Path(__file__).parent.parent.parent / "config" / ".env"
        load_dotenv(env_path)

        # Load YAML config
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # API keys
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    def get_models(self, provider: str = None) -> List[str]:
        """Get list of models to test."""
        if provider:
            return self.config["models"].get(provider, [])

        # Return all models from all providers
        all_models = []
        for models in self.config["models"].values():
            all_models.extend(models)
        return all_models

    def get_step1_config(self) -> Dict[str, Any]:
        """Get Step 1 (classification) configuration."""
        # breakpoint()
        return self.config["step1"]

    def get_step2_config(self) -> Dict[str, Any]:
        """Get Step 2 (articulation) configuration."""
        return self.config["step2"]

    def get_step3_config(self) -> Dict[str, Any]:
        """Get Step 3 (faithfulness) configuration."""
        return self.config["step3"]

    def get_data_config(self) -> Dict[str, Any]:
        """Get data generation configuration."""
        return self.config["data"]

    def get_api_config(self) -> Dict[str, Any]:
        """Get API settings."""
        return self.config["api"]

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config["logging"]


# Global config instance
_config = None

def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

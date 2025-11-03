"""Factory for creating LLM clients."""

from typing import Optional
from .base import BaseLLMClient
from .anthropic_client import AnthropicClient
from .openai_client import OpenAIClient
from .openrouter_client import OpenRouterClient
import logging

logger = logging.getLogger(__name__)


def get_client(
    model: str,
    anthropic_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    openrouter_key: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """
    Get appropriate LLM client based on model identifier.

    Args:
        model: Model identifier (e.g., 'claude-3-5-sonnet-20241022', 'gpt-4o')
        anthropic_key: Anthropic API key
        openai_key: OpenAI API key
        openrouter_key: OpenRouter API key
        **kwargs: Additional arguments for client initialization

    Returns:
        Appropriate LLM client instance

    Raises:
        ValueError: If model is not recognized or API key is missing
    """
    # Anthropic models
    if model.startswith("claude-"):
        if not anthropic_key:
            raise ValueError("Anthropic API key required for Claude models")
        logger.info(f"Creating Anthropic client for {model}")
        return AnthropicClient(anthropic_key, model, **kwargs)

    # OpenAI models (direct)
    elif model.startswith("gpt-"):
        if not openai_key:
            raise ValueError("OpenAI API key required for GPT models")
        logger.info(f"Creating OpenAI client for {model}")
        return OpenAIClient(openai_key, model, **kwargs)

    # OpenRouter models (format: provider/model-name)
    elif "/" in model:
        if not openrouter_key:
            raise ValueError("OpenRouter API key required for OpenRouter models")
        logger.info(f"Creating OpenRouter client for {model}")
        return OpenRouterClient(openrouter_key, model, **kwargs)

    else:
        raise ValueError(f"Unknown model format: {model}")


def get_client_from_config(model: str, config) -> BaseLLMClient:
    """
    Get client using configuration object.

    Args:
        model: Model identifier
        config: Config object with API keys

    Returns:
        Appropriate LLM client instance
    """
    api_config = config.get_api_config()

    return get_client(
        model,
        anthropic_key=config.anthropic_api_key,
        openai_key=config.openai_api_key,
        openrouter_key=config.openrouter_api_key,
        max_retries=api_config.get("max_retries", 3),
        retry_delay=api_config.get("retry_delay", 1.0)
    )

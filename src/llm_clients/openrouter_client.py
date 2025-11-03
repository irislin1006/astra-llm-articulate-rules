"""OpenRouter API client."""

from typing import Dict, Any
import requests
from .base import BaseLLMClient
import logging

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API."""

    BASE_URL = "https://openrouter.ai/api/v1"

    # Approximate pricing per 1M tokens (input, output)
    # OpenRouter has dynamic pricing, these are estimates
    PRICING = {
        "anthropic/claude-3.5-sonnet": (3.0, 15.0),
        "anthropic/claude-3-opus": (15.0, 75.0),
        "openai/gpt-4-turbo": (10.0, 30.0),
        "openai/gpt-4o": (5.0, 15.0),
        "meta-llama/llama-3-70b-instruct": (0.9, 0.9),
        "google/gemini-pro": (0.125, 0.375),
    }

    def __init__(self, api_key: str, model: str, **kwargs):
        """Initialize OpenRouter client."""
        super().__init__(api_key, model, **kwargs)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Generate response using OpenRouter.

        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt

        Returns:
            Dict with response, tokens, and cost
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        data = response.json()

        # Extract response text
        response_text = data["choices"][0]["message"]["content"]

        # Calculate tokens (OpenRouter provides usage stats)
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        # Calculate approximate cost
        pricing = self.PRICING.get(self.model, (5.0, 15.0))  # Default pricing
        cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000

        logger.debug(f"Generated response: {len(response_text)} chars, {total_tokens} tokens, ${cost:.6f}")

        return {
            "response": response_text,
            "tokens": total_tokens,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

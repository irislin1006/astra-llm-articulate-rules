"""Anthropic API client."""

from typing import Dict, Any
import anthropic
from .base import BaseLLMClient
import logging

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic's Claude models."""

    # Pricing per 1M tokens (input, output) - approximate as of 2024
    PRICING = {
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
        "claude-3-opus-20240229": (15.0, 75.0),
        "claude-3-sonnet-20240229": (3.0, 15.0),
        "claude-3-haiku-20240307": (0.25, 1.25),
    }

    def __init__(self, api_key: str, model: str, **kwargs):
        """Initialize Anthropic client."""
        super().__init__(api_key, model, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Generate response using Claude.

        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt

        Returns:
            Dict with response, tokens, and cost
        """
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        # Extract response text
        response_text = response.content[0].text

        # Calculate tokens and cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        pricing = self.PRICING.get(self.model, (3.0, 15.0))  # Default to Sonnet pricing
        cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000

        logger.debug(f"Generated response: {len(response_text)} chars, {total_tokens} tokens, ${cost:.6f}")

        return {
            "response": response_text,
            "tokens": total_tokens,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

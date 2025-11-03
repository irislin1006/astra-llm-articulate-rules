"""OpenAI API client."""

from typing import Dict, Any
from openai import OpenAI
from .base import BaseLLMClient
import logging

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models."""

    # Pricing per 1M tokens (input, output) - approximate as of 2024
    PRICING = {
        "gpt-4-turbo-preview": (10.0, 30.0),
        "gpt-4o": (5.0, 15.0),
        "gpt-4": (30.0, 60.0),
        "gpt-3.5-turbo": (0.5, 1.5),
    }

    # Models that use max_completion_tokens instead of max_tokens
    USES_MAX_COMPLETION_TOKENS = [
        "o1-preview", "o1-mini", "o1",  # o1 series
        "gpt-5", "gpt-5-nano", "gpt-5-mini"  # gpt-5 series (future-proofing)
    ]

    # Models that don't support custom temperature (only default=1)
    NO_TEMPERATURE_CONTROL = [
        "o1-preview", "o1-mini", "o1",  # o1 series
        "gpt-5", "gpt-5-nano", "gpt-5-mini"  # gpt-5 series
    ]

    def __init__(self, api_key: str, model: str, **kwargs):
        """Initialize OpenAI client."""
        super().__init__(api_key, model, **kwargs)
        self.client = OpenAI(api_key=api_key)

        # Determine if this model uses max_completion_tokens
        self.uses_max_completion_tokens = any(
            model_prefix in self.model for model_prefix in self.USES_MAX_COMPLETION_TOKENS
        )

        # Determine if this model supports temperature control
        self.supports_temperature = not any(
            model_prefix in self.model for model_prefix in self.NO_TEMPERATURE_CONTROL
        )

        if self.uses_max_completion_tokens:
            logger.info(f"Model {self.model} uses 'max_completion_tokens' parameter")
        if not self.supports_temperature:
            logger.info(f"Model {self.model} does not support custom temperature (uses default=1)")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Generate response using OpenAI.

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

        # Build API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
        }

        # Only add temperature if the model supports it
        if self.supports_temperature:
            api_params["temperature"] = temperature

        # Use appropriate max tokens parameter based on model
        if self.uses_max_completion_tokens:
            api_params["max_completion_tokens"] = max_tokens
        else:
            api_params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**api_params)

        # Extract response text
        response_text = response.choices[0].message.content
        # breakpoint()
        # Calculate tokens and cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        # Calculate cost
        pricing = self.PRICING.get(self.model, (10.0, 30.0))  # Default to GPT-4 Turbo pricing
        cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000

        logger.debug(f"Generated response: {len(response_text)} chars, {total_tokens} tokens, ${cost:.6f}")

        return {
            "response": response_text,
            "tokens": total_tokens,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

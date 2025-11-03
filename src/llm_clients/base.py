"""Base interface for LLM clients."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import time
import logging

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, api_key: str, model: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize LLM client.

        Args:
            api_key: API key for the service
            model: Model identifier
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.total_tokens = 0
        self.total_cost = 0.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt

        Returns:
            Dict with 'response', 'tokens', and 'cost' keys
        """
        pass

    def generate_with_retry(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """Generate with automatic retry on failure."""
        for attempt in range(self.max_retries):
            try:
                result = self.generate(prompt, temperature, max_tokens, system_prompt)
                self.total_tokens += result.get('tokens', 0)
                self.total_cost += result.get('cost', 0.0)
                return result
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost
        }

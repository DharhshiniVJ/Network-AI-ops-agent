"""
LLM Provider Factory.
"""
import os
from typing import Tuple
from openai import OpenAI

class LLMProviderFactory:
    """Factory to get the right OpenAI client and model based on provider name."""

    @staticmethod
    def get_client(provider: str) -> Tuple[OpenAI, str]:
        """
        Get the OpenAI client and model string for a given provider.

        Args:
            provider: The provider name ("gemini", "openai", "ollama").

        Returns:
            A tuple of (OpenAI client, model string).

        Raises:
            ValueError: If the provider is unknown.
        """
        provider = provider.lower()
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            return client, "gemini-2.0-flash"
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            client = OpenAI(api_key=api_key)
            return client, "gpt-4o-mini"
        elif provider == "ollama":
            client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1"
            )
            return client, "qwen2.5:7b-instruct"
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

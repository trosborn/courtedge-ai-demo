"""
LiteLLM client wrapper providing ChatAnthropic-compatible interface.

This module provides a clean async wrapper around LiteLLM that mimics
the LangChain ChatAnthropic interface for easy drop-in replacement.

Supports both direct provider calls and LiteLLM proxy usage:
- Direct: Set ANTHROPIC_API_KEY (or provider-specific key)
- Proxy: Set LITELLM_API_KEY and LITELLM_API_BASE
"""

from typing import List, Optional, Any
from litellm import acompletion
import logging
import os

logger = logging.getLogger(__name__)


class Message:
    """Base message class."""

    def __init__(self, content: str):
        self.content = content
        self.role = "user"


class SystemMessage(Message):
    """System message for providing context/instructions to the LLM."""

    def __init__(self, content: str):
        super().__init__(content)
        self.role = "system"


class HumanMessage(Message):
    """Human message representing user input."""

    def __init__(self, content: str):
        super().__init__(content)
        self.role = "user"


class AIMessage(Message):
    """AI message representing LLM response."""

    def __init__(self, content: str):
        super().__init__(content)
        self.role = "assistant"


class Response:
    """Response object that mimics LangChain's response structure."""

    def __init__(self, content: str, raw_response: Optional[Any] = None):
        self.content = content
        self.raw_response = raw_response


class LiteLLMClient:
    """
    Async LiteLLM client that provides ChatAnthropic-compatible interface.

    This client wraps LiteLLM's acompletion function and provides the same
    interface as LangChain's ChatAnthropic for easy migration.

    Args:
        model: Model name (e.g., "claude-sonnet-4-20250514")
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in response (default: 4096)
        **kwargs: Additional parameters passed to LiteLLM

    Example:
        ```python
        client = LiteLLMClient(model="claude-sonnet-4-20250514", temperature=0.7)
        response = await client.ainvoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!")
        ])
        print(response.content)
        ```
    """

    def __init__(
        self,
        model: str = "claude-4-5-sonnet",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

        # Support LiteLLM proxy or direct provider calls
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")

        logger.info(
            f"Initialized LiteLLMClient with model={model}, "
            f"temperature={temperature}, max_tokens={max_tokens}, "
            f"api_base={self.api_base or 'default'}"
        )

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        """
        Convert Message objects to LiteLLM format.

        Args:
            messages: List of Message objects (SystemMessage, HumanMessage, AIMessage)

        Returns:
            List of dicts in LiteLLM format: [{"role": "...", "content": "..."}, ...]
        """
        converted = []
        for msg in messages:
            converted.append({
                "role": msg.role,
                "content": msg.content
            })
        return converted

    async def ainvoke(self, messages: List[Message]) -> Response:
        """
        Async invocation of the LLM with a list of messages.

        Args:
            messages: List of Message objects forming the conversation

        Returns:
            Response object with .content attribute containing the LLM's response

        Raises:
            Exception: If LiteLLM call fails
        """
        try:
            # Convert messages to LiteLLM format
            litellm_messages = self._convert_messages(messages)

            logger.debug(
                f"Invoking LiteLLM with model={self.model}, "
                f"message_count={len(messages)}"
            )

            # Prepare completion arguments
            completion_kwargs = {
                "model": self.model,
                "messages": litellm_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **self.extra_params
            }

            # Add API key and base if configured (for LiteLLM proxy)
            if self.api_key:
                completion_kwargs["api_key"] = self.api_key
            if self.api_base:
                # LiteLLM proxy acts as OpenAI-compatible endpoint
                completion_kwargs["api_base"] = self.api_base
                completion_kwargs["custom_llm_provider"] = "openai"

            # Call LiteLLM's async completion
            response = await acompletion(**completion_kwargs)

            # Extract content from response
            content = response.choices[0].message.content

            logger.debug(f"LiteLLM response received, length={len(content)}")

            # Return Response object with .content attribute
            return Response(content=content, raw_response=response)

        except Exception as e:
            logger.error(f"LiteLLM invocation failed: {str(e)}", exc_info=True)
            raise

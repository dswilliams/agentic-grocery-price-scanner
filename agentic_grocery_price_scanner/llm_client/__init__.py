"""LLM Client module for local inference with Ollama."""

from .ollama_client import OllamaClient
from .prompt_templates import PromptTemplates

__all__ = ["OllamaClient", "PromptTemplates"]
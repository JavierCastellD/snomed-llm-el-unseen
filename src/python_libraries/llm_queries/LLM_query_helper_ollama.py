import os
import warnings

import ollama

from .LLM_query_helper import LLMQueryHelper


class OllamaQueryHelper(LLMQueryHelper):
    """Class wrapper that uses Ollama to perform queries to locally-served open-source LLMs.

    Attributes:
        _ollama_model_name (str):
            Original model name as passed to Ollama (e.g. "llama3.1:8b").
        temperature (float):
            Value that controls the determinism of the response.
        client (ollama.Client):
            Ollama client used to send chat requests.
    """
    def __init__(self, model_name: str, temperature: float = 1.0,
                 host: str = "http://localhost:11434",
                 folder_cache_path: str = None):
        """Initializes the Ollama client and the base cache machinery.

        Parameters:
            model_name (str):
                Name of the model as known to Ollama (e.g. "llama3.1:8b").
            temperature (float):
                Value that controls the determinism of the response. Defaults to 1.0.
            host (str):
                URL of the Ollama server. Defaults to "http://localhost:11434".
            folder_cache_path (str):
                Path to the folder that will contain the cache. Optional.
        """
        self._ollama_model_name = model_name
        # Colons in model tags (e.g. "llama3.1:8b") are invalid in Windows filenames.
        safe_name = model_name.replace(":", "_")
        # OLLAMA_HOST env var takes priority (e.g. when set by a SLURM job script)
        effective_host = os.environ.get('OLLAMA_HOST')
        if effective_host:
            if not effective_host.startswith('http'):
                effective_host = 'http://' + effective_host
        else:
            effective_host = host
        super().__init__(api_key='', endpoint=effective_host, model_name=safe_name,
                         temperature=temperature, folder_cache_path=folder_cache_path)
        self.client = ollama.Client(host=effective_host)

    def _send_request(self, payload: dict, return_response: bool = False):
        """Send the request via Ollama and return the completion string.

        Parameters:
            payload (dict):
                Dictionary built by the base class. Only 'messages' is used.
            return_response (bool):
                Whether to return the raw Ollama response object instead of the
                message string.

        Returns:
            A string with the message content, the raw response object if
            return_response is True, or None if the request failed.
        """
        try:
            response = self.client.chat(
                model=self._ollama_model_name,
                messages=payload['messages'],
                options={"temperature": self.temperature}
            )
        except ollama.ResponseError as e:
            warnings.warn(f"Ollama error: {e}. Returning None.")
            return None

        if return_response:
            return response
        return response.message.content

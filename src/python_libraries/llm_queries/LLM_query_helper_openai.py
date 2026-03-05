import time
import warnings

from openai import AzureOpenAI, RateLimitError, BadRequestError
from openai.types.chat.chat_completion import ChatCompletion

from .LLM_query_helper import LLMQueryHelper

class LLMQueryHelperOpenAI(LLMQueryHelper):
    """Class wrapper that uses OpenAI to simplify performing queries to Microsoft's Azure OpenAI for using LLMs.
    
    Attributes:
        api_key (str):
            API key to Microsoft's service.
        endpoint (str):
            URL of the endpoint provided by Microsoft.
        model_name (str):
            Name of the LLM model type deployed at the endpoint.
        deployment (str):
            Name given to the LLM model deployed at the endpoint. If set to None, uses model_name.
        temperature (float):
            Value between 0 and 2 that defines the determinism of the response from the LLM.
        api_version (str):
            Version of the API of Microsoft Azure.
        client (AzureOpenAI):
            AzureOpenAI client to send the chat completion requests.
    """
    def __init__(self, api_key : str, endpoint : str, model_name : str = "gpt-4o-mini", deployment : str = None,
                 temperature : float = 0.5, api_version : str = "2024-12-01-preview"):
        """Initializes the class by setting up the API key, endpoint, model, and headers.
        
        Parameters:
            api_key (str):
                API key to Microsoft's service.
            endpoint (str):
                URL of the endpoint provided by Microsoft.
            model_name (str):
                Name of the LLM model type deployed at the endpoint. By default it uses gpt-4o-mini.
            deployment (str):
                Name given to the LLM model deployed at the endpoint. If set to None, uses model_name.
            temperature (float):
                Value between 0 and 2 that defines the determinism of the response from the LLM.
            api_version (str):
                Version of the API of Microsoft Azure. By default uses 2024-12-01-preview.
        """
        super().__init__(api_key=api_key, endpoint=endpoint, model_name=model_name, temperature=temperature)
        self.deployment = deployment if deployment is not None else model_name
        self.api_version = api_version
        self.client = AzureOpenAI(
                        api_version=api_version,
                        azure_endpoint=endpoint,
                        api_key=api_key,
        )

    def _send_request(self, payload : dict[str], return_response : bool = False) -> str | ChatCompletion:
        """Send the request with the payload provided as argument. If return_response is set to True, returns the
        ChatCompletion object from the request. Otherwise, it just returns the message.
        
        Parameters:
            payload (dict[str]):
                Dictionary that contains the information for the request. It should at least have: 'messages'.
            return_response (bool):
                Whether to return the ChatCompletion response object or just a string with the message.

        Returns:
            A string with the message from the response or the ChatCompletion object from the request.
        """
        response = None
        while response is None:
            try:
                response = self.client.chat.completions.create(messages=payload['messages'],
                                                            temperature=self.temperature,
                                                            model=self.deployment)
            except RateLimitError as e:
                warnings.warn("Rate limit per minute exceeded. Waiting 60 seconds...")
                time.sleep(60)
            except BadRequestError as e:
                warnings.warn("Error. Returning None... Message: " + str(payload['messages']))
                return None
            
        if return_response:
            return response
        else:
            return response.choices[0].message.content
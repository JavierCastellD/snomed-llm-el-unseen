import requests
from requests import Response
import time
import warnings

from .LLM_query_helper import LLMQueryHelper

STATUS_CODE_CORRECT = 200
STATUS_CODE_TOO_MANY_REQUESTS = 429
STATUS_INVALID_REQUEST = 424

class LLMQueryHelperRequest(LLMQueryHelper):
    """Class wrapper that uses HTTP requests to simplify performing queries to Microsoft's Azure OpenAI for using LLMs.
    
    Attributes:
        api_key (str):
            API key to Microsoft's service.
        endpoint (str):
            URL of the endpoint provided by Microsoft.
        model_name (str):
            Name of the LLM model type deployed at the endpoint.
        temperature (float):
            Value between 0 and 2 that defines the determinism of the response from the LLM.
        headers (dict[str]):
            Headers of the request.
    """
    def __init__(self, api_key : str, endpoint : str, model_name : str = "DeepSeek-V3", temperature : float = 0.5, headers : dict[str] = None):
        """Initializes the class by setting up the API key, endpoint, model, and headers.
        
        Parameters:
            api_key (str):
                API key to Microsoft's service.
            endpoint (str):
                URL of the endpoint provided by Microsoft.
            model_name (str):
                Name of the LLM model type deployed at the endpoint. By default it uses DeepSeek-V3.
            temperature (float):
                Value between 0 and 2 that defines the determinism of the response from the LLM.
            headers (dict[str]):
                Dictionary that serves as headers. By default it sets the content-type as a json and adds the API key.
        """
        super().__init__(api_key=api_key, endpoint=endpoint, model_name=model_name, temperature=temperature)
        self.headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        } if headers is None else headers

    def _send_request(self, payload : dict[str], return_response : bool = False) -> str | Response | None:
        """Send the request with the payload provided as argument and returns the message. If return_response is set to True, returns the
        Response object from the request.
        
        Parameters:
            payload (dict[str]):
                Dictionary that contains the information for the request.
            return_response (bool):
                Whether to return the Response object or just a string with the message.

        Returns:
            A string with the message from the response or the Response object from the request. 
        """
        response_status_code = None

        while (response_status_code is None or response_status_code in [STATUS_CODE_TOO_MANY_REQUESTS, STATUS_INVALID_REQUEST]):
            response = requests.post(self.endpoint, headers=self.headers, json=payload)
            response_status_code = response.status_code

            if response_status_code == STATUS_CODE_TOO_MANY_REQUESTS:
                warnings.warn("Rate limit per minute exceeded. Waiting 60 seconds...")
                time.sleep(60)
            elif response_status_code == STATUS_INVALID_REQUEST:
                warnings.warn("424 error. Trying again in 1 second...")
                time.sleep(1)
                
        if return_response:
            return response
        else:
            if response.status_code == STATUS_CODE_CORRECT:
                return response.json()['choices'][0]['message']['content']
            else:
                error_message = f"""Status code {response.status_code}. {response.json()['error']['message']}.\nCaused by: {payload['messages'][1]['content']}"""
                warnings.warn(error_message)
                return None
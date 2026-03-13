from abc import ABC, abstractmethod
import hashlib
import json
import os

# Base directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_prompt(path: str) -> str:
    with open(os.path.join(BASE_DIR, path), "r", encoding="utf-8") as f:
        return f.read()

def get_hash(*values : str):
    """Auxiliary function to get the hash of a set of strings.
    
    Parameters:
        values (str):
            Strings to be hashed.
    
    Returns:
        A string value that represents the hash of the values provided as parameters in MD5.
    """
    combined_string = "-".join(values)
    return hashlib.md5(combined_string.encode('utf-8')).hexdigest()

# PROMPT FILES
ENTITY_LINKING_SYS_ES = "prompts/system/entity_linking_es.txt"
ENTITY_LINKING_USER_ES = "prompts/user/entity_linking_es.txt"
ENTITY_LINKING_SYS_EN = "prompts/system/entity_linking_en.txt"
ENTITY_LINKING_USER_EN = "prompts/user/entity_linking_en.txt"

ABBREVIATION_DISAMBIGUATION_SYS_ES = "prompts/system/abbreviation_disambiguation_es.txt"
ABBREVIATION_DISAMBIGUATION_USER_ES = "prompts/user/abbreviation_disambiguation_es.txt"
ABBREVIATION_DISAMBIGUATION_SYS_EN = "prompts/system/abbreviation_disambiguation_en.txt"
ABBREVIATION_DISAMBIGUATION_USER_EN = "prompts/user/abbreviation_disambiguation_en.txt"

REPHRASING_SYS_EN = "prompts/system/rephrasing_en.txt"
REPHRASING_USER_EN = "prompts/user/rephrasing_en.txt"

SYNONYMS_SYS_EN = "prompts/system/synonyms_en.txt"
SYNONYMS_USER_EN = "prompts/user/synonyms_en.txt"

class LLMQueryHelper(ABC):
    """Abstract class that defines a wrapper to simplify performing queries to Microsoft's Azure OpenAI for using LLMs.
    
    Attributes:
        api_key (str):
            API key to Microsoft's service.
        endpoint (str):
            URL of the endpoint provided by Microsoft.
        model_name (str):
            Name of the LLM model type deployed at the endpoint.
        temperature (float):
            Value between 0 and 2 that defines the determinism of the response from the LLM.
        folder_cache_path (str):
            Path to the folder that will contain the cache to prevent repeated requests.
        cache (dict[str, dict]):
            Dictionary used to store the requests and responses.
    """
    def __init__(self, api_key : str, endpoint : str, model_name : str, temperature : float = 0.5,
                 folder_cache_path : str = None):
        """Initializes the class by setting up the API key, endpoint, model, and temperature.
        
        Parameters:
            api_key (str):
                API key to Microsoft's service.
            endpoint (str):
                URL of the endpoint provided by Microsoft.
            model_name (str):
                Name of the LLM model type deployed at the endpoint.
            temperature (float):
                Value between 0 and 2 that defines the determinism of the response from the LLM.
            folder_cache_path (str):
                Path to the folder that will contain the cache to prevent repeated requests.
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.model_name = model_name
        self.temperature = temperature

        if folder_cache_path is None:
            folder_cache_path = ".llm_query_helper_" + self.model_name.replace(" ", "_") + "_cache"

        self.folder_cache_path = folder_cache_path

        if not os.path.exists(self.folder_cache_path):
            os.makedirs(self.folder_cache_path)
    
        self.cache = {'EL' : {},
                    'AB' : {},
                    'RP' : {},
                    'SYN' : {},
                    'AB_SP' : {},
                    'EL_SP' : {}
                    }
    
        # We load each cache file
        for key in self.cache.keys():
            dic_path = os.path.join(self.folder_cache_path, key + '_' + self.model_name.replace(" ", "_") + '.json')
            if os.path.exists(dic_path):
                with open(dic_path, 'r', encoding='utf-8') as dic_file:
                    self.cache[key] = json.load(dic_file)

    @abstractmethod
    def _send_request(self, payload : dict[str], return_response : bool = False) -> str | None:
        """Send the request with the payload provided as argument and returns the message. If return_response is set to True, returns the
        original response object from the request. The type of the response object might vary depending on the implementation.
        
        Parameters:
            payload (dict[str]):
                Dictionary that contains the information for the request.
            return_response (bool):
                Whether to return the response object or just a string with the message.

        Returns:
            A string with the message from the response or the response object from the request. If there is an error, None is returned instead.
        """
        pass

    def send_request_EL_es(self, sentence : str, span : str, options : list[str], return_response : bool = False) -> str | None:
        """Creates the query for entity linking and returns the response message. If return_response is set to True, returns the response object
        from the request. The type of the response object might vary depending on the implementation of _send_request.
        
        Parameters:
            sentence (str):
                Sentence that provides context for the entity linking.
            span (str):
                String detected as an entity.
            options (list[str]):
                List of options for the LLM to choose from.
            return_response (bool):
                Whether to return the response object or just a string with the message.

        Returns:
            A string with the message from the response or the response object from the request. If there is an error, None is returned instead.
        """
        payload = self.create_EL_es_payload(sentence=sentence, span=span, options=options)

        hash = get_hash(sentence, span, str(options), str(return_response), str(payload['messages']))

        if hash in self.cache["EL_SP"]:
            response = self.cache["EL_SP"][hash]
        else:
            response = self._send_request(payload, return_response)

            self.cache["EL_SP"][hash] = response

        return response

    def send_request_AB_es(self, sentence : str, span : str, return_response : bool = False) -> str | None:
        """Creates the query for resolving abbreviations and returns the response message. If return_response is set to True, returns the response object
        from the request. The type of the response object might vary depending on the implementation of _send_request.
        
        Parameters:
            sentence (str):
                Sentence that provides context for the entity linking.
            span (str):
                String detected as an entity.
            return_response (bool):
                Whether to return the response object or just a string with the message.

        Returns:
            A string with the message from the response or the response object from the request. If there is an error, None is returned instead.
        """
        payload = self.create_AB_es_payload(sentence=sentence, span=span)
        hash = get_hash(sentence, span, str(return_response), str(payload['messages']))

        if hash in self.cache["AB_SP"]:
            response = self.cache["AB_SP"][hash]
        else:
            response = self._send_request(payload, return_response)

            self.cache["AB_SP"][hash] = response

        return response

    def send_request_EL(self, sentence : str, span : str, section : str,
                        options : list[str], return_response : bool = False) -> str | None:
        """Creates the query for entity linking and returns the response message. If return_response is set to True, returns the response object
        from the request. The type of the response object might vary depending on the implementation of _send_request.
        
        Parameters:
            sentence (str):
                Sentence that provides context for the entity linking.
            span (str):
                String detected as an entity.
            section (str):
                String that denotes the section where the sentence is found.
            options (list[str]):
                List of options for the LLM to choose from.
            return_response (bool):
                Whether to return the response object or just a string with the message.

        Returns:
            A string with the message from the response or the response object from the request. If there is an error, None is returned instead.
        """
        payload = self.create_EL_payload(sentence=sentence, span=span, section=section, options=options)

        hash = get_hash(sentence, span, section, str(options), str(return_response), str(payload['messages']))

        if hash in self.cache["EL"]:
            response = self.cache["EL"][hash]
        else:
            response = self._send_request(payload, return_response)

            self.cache["EL"][hash] = response

        return response

    def send_request_AB(self, sentence : str, span : str, section : str, 
                        return_response : bool = False) -> str | None:
        """Creates the query for resolving abbreviations and returns the response message. If return_response is set to True, returns the response object
        from the request. The type of the response object might vary depending on the implementation of _send_request.
        
        Parameters:
            sentence (str):
                Sentence that provides context for the entity linking.
            span (str):
                String detected as an entity.
            section (str):
                String that denotes the section where the sentence is found.
            return_response (bool):
                Whether to return the response object or just a string with the message.

        Returns:
            A string with the message from the response or the response object from the request. If there is an error, None is returned instead.
        """
        payload = self.create_AB_payload(sentence=sentence, span=span, section=section)
        hash = get_hash(sentence, span, section, str(return_response), str(payload['messages']))

        if hash in self.cache["AB"]:
            response = self.cache["AB"][hash]
        else:
            response = self._send_request(payload, return_response)

            self.cache["AB"][hash] = response

        return response

    def send_request_RP(self, sentence : str, span : str, section : str, 
                        return_response : bool = False) -> str | None:
        """Creates the query for rephrasing spans and returns the response message. If return_response is set to True, returns the response object
        from the request. The type of the response object might vary depending on the implementation of _send_request.
        
        Parameters:
            sentence (str):
                Sentence that provides context for the entity linking.
            span (str):
                String detected as an entity.
            section (str):
                String that denotes the section where the sentence is found.
            return_response (bool):
                Whether to return the response object or just a string with the message.

        Returns:
            A string with the message from the response or the response object from the request. If there is an error, None is returned instead.
        """
        payload = self.create_RP_payload(sentence=sentence, span=span, section=section)
        hash = get_hash(sentence, span, section, str(return_response), str(payload['messages']))

        if hash in self.cache["RP"]:
            response = self.cache["RP"][hash]
        else:
            response = self._send_request(payload, return_response)

            self.cache["RP"][hash] = response

        return response

    def send_request_SYN(self, clinical_term : str, synonyms : list[str],
                        return_response : bool = False) -> str | None:
        """Creates the query for rephrasing spans and returns the response message. If return_response is set to True, returns the response object
        from the request. The type of the response object might vary depending on the implementation of _send_request.
        
        Parameters:
            clinical_term (str):
                Clinical term for which to obtain the synonyms.

        Returns:
            A string with the message from the response or the response object from the request. If there is an error, None is returned instead.
        """
        payload = self.create_SYN_payload(clinical_term=clinical_term, synonyms=synonyms)
        hash = get_hash(clinical_term, str(synonyms), str(return_response), str(payload['messages']))

        if hash in self.cache["SYN"]:
            response = self.cache["SYN"][hash]
        else:
            response = self._send_request(payload, return_response)

            self.cache["SYN"][hash] = response

        return response

    def create_EL_es_payload(self, sentence : str, span : str, options : list[str]) -> dict[str]:
        """Creates the payload to be sent for the entity linking request in Spanish.
        
        Parameters:
            sentence (str):
                Sentence that provides context for the entity linking.
            span (str):
                String detected as an entity.
            section (str):
                String that denotes the section where the sentence is found.
            options (list[str]):
                List of options for the LLM to choose from.
        Returns:
            A dictionary that contains the payload for the request.
        """
        system_prompt = load_prompt(ENTITY_LINKING_SYS_ES)
        user_prompt = load_prompt(ENTITY_LINKING_USER_ES)
        payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt.format(sentence=sentence, span=span, options=options)
                        }
                    ],
                    "temperature": self.temperature,
                    "model" : f"{self.model_name}"
                }
        
        return payload
    
    def create_AB_es_payload(self, sentence : str, span : str) -> dict[str]:
        """Creates the payload to be sent for the abbreviation request in Spanish.
        
        Parameters:
            sentence (str):
                Sentence that provides context for the entity linking.
            span (str):
                String detected as an entity.
            section (str):
                String that denotes the section where the sentence is found.
        Returns:
            A dictionary that contains the payload for the request.
        """
        system_prompt = load_prompt(ABBREVIATION_DISAMBIGUATION_SYS_ES)
        user_prompt = load_prompt(ABBREVIATION_DISAMBIGUATION_USER_ES)
        payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt.format(sentence=sentence, span=span)
                        }
                    ],
                    "temperature": self.temperature,
                    "model" : f"{self.model_name}"
                }
        
        return payload

    def create_EL_payload(self, sentence : str, span : str, section : str, options : list[str]) -> dict[str]:
        """Creates the payload to be sent for the entity linking request.
        
        Parameters:
            sentence (str):
                Sentence that provides context for the entity linking.
            span (str):
                String detected as an entity.
            section (str):
                String that denotes the section where the sentence is found.
            options (list[str]):
                List of options for the LLM to choose from.
        Returns:
            A dictionary that contains the payload for the request.
        """
        system_prompt = load_prompt(ENTITY_LINKING_SYS_EN)
        user_prompt = load_prompt(ENTITY_LINKING_USER_EN)
        payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt.format(sentence=sentence, span=span, section=section, options=options)
                        }
                    ],
                    "temperature": self.temperature,
                    "model" : f"{self.model_name}"
                }
        
        return payload
    
    def create_AB_payload(self, sentence : str, span : str, section : str) -> dict[str]:
        """Creates the payload to be sent for the abbreviation request.
        
        Parameters:
            sentence (str):
                Sentence that provides context for the entity linking.
            span (str):
                String detected as an entity.
            section (str):
                String that denotes the section where the sentence is found.
        Returns:
            A dictionary that contains the payload for the request.
        """
        system_prompt = load_prompt(ABBREVIATION_DISAMBIGUATION_SYS_EN)
        user_prompt = load_prompt(ABBREVIATION_DISAMBIGUATION_USER_EN)
        payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt.format(sentence=sentence, span=span, section=section)
                        }
                    ],
                    "temperature": self.temperature,
                    "model" : f"{self.model_name}"
                }
        
        return payload

    def create_RP_payload(self, sentence : str, span : str, section : str) -> dict[str]:
        """Creates the payload to be sent for the rephrasing request.

        Parameters:
            sentence (str):
                Sentence that provides context for the entity linking.
            span (str):
                String detected as an entity.
            section (str):
                String that denotes the section where the sentence is found.
        Returns:
            A dictionary that contains the payload for the request.
        """
        system_prompt = load_prompt(REPHRASING_SYS_EN)
        user_prompt = load_prompt(REPHRASING_USER_EN)
        payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt.format(sentence=sentence, span=span, section=section)
                        }
                    ],
                    "temperature": self.temperature,
                    "model" : f"{self.model_name}"
                }
        
        return payload
    
    def create_SYN_payload(self, clinical_term : str, synonyms : list[str]) -> dict[str]:
        """Creates the payload to be sent for the synonym request.

        Parameters:
            clinical_term (str):
                Clinical term for which to obtain the synonyms.
        Returns:
            A dictionary that contains the payload for the request.
        """
        system_prompt = load_prompt(SYNONYMS_SYS_EN)
        user_prompt = load_prompt(SYNONYMS_USER_EN)
        payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt.format(clinical_term=clinical_term, synonyms=synonyms)
                        }
                    ],
                    "temperature": self.temperature,
                    "model" : f"{self.model_name}"
                }
        
        return payload

    def save_cache(self):
        """Method to save the cache of the queries."""
        # For each cache dictionary, we obtain the path and save the current dictionary there
        for key in self.cache.keys():
            dic_path = os.path.join(self.folder_cache_path, key + '_' + self.model_name.replace(" ", "_") + '.json')
            with open(dic_path, 'w', encoding='utf-8') as dic_file:
                json.dump(self.cache[key], dic_file, indent=4)


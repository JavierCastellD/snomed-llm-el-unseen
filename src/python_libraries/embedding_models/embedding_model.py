from abc import ABC, abstractmethod
import warnings

import numpy as np

def save_embeddings(embedding_dictionary : dict[str|int, list|np.ndarray], filename : str):
    '''Function to save a embedding dictionary into a compressed numpy format (npz).
    
    Parameters:
        embedding_dictionary (dict):
            Dictionary that has IDs as keys and embeddings as values.
        
        filename (str):
            Name of the filename saved.
    '''
    ids = np.array(list(embedding_dictionary.keys()))
    vectors = np.array(list(embedding_dictionary.values()), dtype=np.float32)
    np.savez_compressed(filename, ids=ids, vectors=vectors)

def load_embeddings(filename : str) -> dict[str|int, list|np.ndarray]:
    '''Function to load the embedding dictionary.
    
    Parameters:
        filename (str):
            Name of the filename that contains the dictionary.
    Returns:
        A python dictionary that links ID to embeddings.
    '''
    data = np.load(filename, allow_pickle=True)
    ids, vectors = data["ids"], data["vectors"]
    embedding_dictionary = {id_: vec for id_, vec in zip(ids, vectors)}
    
    return embedding_dictionary 

class EmbeddingModel(ABC):
    '''Abstract class that represents an embedding model, such as BERT, FastText, etc. 
    The methods to be implemented are train_model, save_model, load_model and get_embedding.
    
    Attributes:
        model:
            Language model prepared to be used to obtain embeddings from it.
    '''
    def __init__(self, model_path : str = None, corpora : list[str] = None):
        '''Loads the model from a certain path if it was provided. Otherwise, if a corpora was given, trains the model. 
        The method expects either model_path or corpora to be provided.
        
        Parameters:
            model_path (str):
                Path to the language model to be loaded.
            corpora (list):
                List that contains the path to the corpora from which the model will be trained.
        '''
        if model_path is None and corpora is None:
            raise ValueError("Either 'model_path' or 'corpora' must be provided.")

        if model_path is not None and corpora is not None:
            warnings.warn(
                "Both 'model_path' and 'corpora' were provided. "
                "'model_path' will be used and 'corpora' will be ignored.",
                UserWarning
            )

        if model_path is not None:
            self.model = self.load_model(model_path)
        elif corpora is not None: 
            self.model = self.train_model(corpora)

    @abstractmethod
    def train_model(self, corpora : list[str]):
        '''Method to train the model from a list of corpus.
        
        Parameters:
            corpora (list):
                List of paths from which to train the model.
        
        Returns:
            A trained language model.
        '''
        pass

    @abstractmethod
    def save_model(self, model_path : str):
        '''Method to save the model.
        
        Parameters:
            model_path (str):
                Path to where the model should be saved.
        '''
        pass

    @abstractmethod
    def load_model(self, model_path : str):
        '''Method to load the model.
        
        Parameters:
            model_path (str):
                Path from where the model to be loaded can be found.
        '''
        pass

    @abstractmethod
    def get_embedding(self, word : str):
        '''Method to obtain an embedding for a given string, which can be either a single word or multiple ones.
        
        Parameters:
            word (str):
                String that represents the word(s) from which to obtain the embedding.
        
        Returns:
            An embedding.
        '''
        pass

    def get_embeddings(self, words_list : list[str]):
        '''Method to obtain the embeddings for each string in a list.
        
        Parameters:
            words_list (list):
                List of strings of which to obtain the embeddings.
        
        Returns:
            A list of embeddings. The list will be of the same length of words_list.
        '''
        embeddings = []
        for word in words_list:
            embeddings.append(self.get_embedding(word))
        
        return embeddings


    def get_embedding_from_list(self, names_list : list[str], agg_average : bool = True) -> list:
        '''Method to obtain an embedding by aggregating the embedding of different words. If agg_average is 
        set to True, the aggregation is done by averaging the embeddings. Otherwise, the aggregation is done
        by a sum.
        
        Parameters:
            names_list (list):
                List of Strings that represents the words from which to obtain the embedding.
            
            agg_average (bool):
                Whether to aggregate by averaging or summing.
        Returns:
            An embedding.
        '''
        vectors = []

        if len(names_list) == 0:
            return []

        for name in names_list:
            vectors.append(self.get_embedding(name))
        
        if agg_average:
            return sum(vectors)/len(vectors)
        else:
            return sum(vectors)
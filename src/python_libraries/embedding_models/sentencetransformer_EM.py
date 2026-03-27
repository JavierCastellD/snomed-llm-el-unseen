import warnings

from sentence_transformers import SentenceTransformer

from .embedding_model import EmbeddingModel

class SentenceTransformerEM(EmbeddingModel):
    '''Class that represents a SentenceTransformer embedding model. It works over the implementation of SBERT described here: https://www.sbert.net/.
    
    Attributes:
        model:
            Language model prepared to be used to obtain embeddings from it.
    '''
    def __init__(self, model_path : str = None, corpora : list[str] = None):
        '''Method to load the SentenceTransformer model. If model_path was given as a parameter, the model will be loaded from 
        sentence-transformers or from the disk. Otherwise, it will train from corpora. The method expects to receive
        at least one of those parameters.
        
        Parameters:
            model_path (str):
                Path to the language model to be loaded or name in sentence-transformers hub.
            corpora (list):
                List that contains the path to the corpora from which the model will be trained.
        '''
        if model_path is not None:
            self.model = self.load_model(model_path)
        elif corpora is not None: 
            self.model = self.train_model(corpora)
        else:
            warnings.warn('Neither model path to load nor corpora to train were given.')

    def train_model(self, corpora : list[str]):
        '''Method to train the model from a list of corpus.
        
        Parameters:
            corpora (list):
                List of paths from which to train the model.
        
        Returns:
            A trained language model.
        '''
        raise NotImplementedError('Training not implemented yet for SentenceTransformersEM')

    def save_model(self, model_path : str):
        '''Method to save the model.
        
        Parameters:
            model_path (str):
                Path to where the model should be saved.
        '''
        raise NotImplementedError('Saving not implemented yet for SentenceTransformersEM')

    def load_model(self, model_path : str):
        '''Method to load the model.
        
        Parameters:
            model_path (str):
                Path from where the model to be loaded can be found.
        '''
        return SentenceTransformer(model_path)

    def get_embedding(self, word : str):
        '''Method to obtain an embedding for a given string, which can be either a single word or multiple ones.
        
        Parameters:
            word (str):
                String that represents the word(s) from which to obtain the embedding.
        
        Returns:
            An embedding.
        '''
        return self.model.encode(word)

    def get_embeddings(self, words_list: list[str]):
        '''Method to obtain the embeddings for each string in a list.
        
        Parameters:
            words_list (list):
                List of strings of which to obtain the embeddings.
        
        Returns:
            A list of embeddings. The list will be of the same length of words_list.
        '''
        return list(self.model.encode(words_list))
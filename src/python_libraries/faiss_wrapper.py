import faiss
import numpy as np

class FaissWrapper:
    """Class to simplify using FAISS.
    
    Attributes:
        index2entity (list):
            Used to transform from an index to its corresponding entity ID.
        
        faiss_index:
            Faiss index object.
        
    """
    def __init__(self, embedding_dictionary : dict[str, list]):
        """Initializes the Faiss index object and prepares the index2entity property to transform from index to an ID.
        
        Parameters:
            embedding_dictionary (dict):
                Dictionary where the keys are IDs and the values are embeddings.
        """
        # Extract the embeddings
        embedding_values = list(embedding_dictionary.values())
        embedding_size = len(embedding_values[0])

        # To transform from an index to an entity ID
        self.index2entity = list(embedding_dictionary.keys())
                
        # We use Faiss for fast similarity search, so we need to normalize the embeddings first
        faiss_embeddings = np.array(embedding_values).astype(np.float32)
        faiss.normalize_L2(faiss_embeddings)
        
        # We set up faiss to use cosine similarity
        self.faiss_index = faiss.index_factory(embedding_size, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.ntotal
        
        # Add the normalized embeddings
        self.faiss_index.add(faiss_embeddings)

    def search_similar(self, embeddings : list, n : int = 10):
        """Method that returns the most similar entities for each of the embeddings.
        
        Parameters:
            embeddings (list):
                Embeddings from which to obtain the most similar concepts.
            n (int):
                Number of entities to return for each embedding.

        Returns:
            A tuples of lists where the first elements are the similarity values, and the second one are the entities IDs.
        """
        # Normalize the embeddings
        q_vectors = np.array(embeddings).astype(np.float32)
        faiss.normalize_L2(q_vectors)

        # Obtain the most similar entities
        sim_values, index = self.faiss_index.search(q_vectors, n)

        entities = []
        # Transform from index to entities
        for i in range(len(index)):
            sim_entities = [self.index2entity[i_entity] for i_entity in index[i]]

            entities.append(sim_entities)
        
        return (sim_values, entities)
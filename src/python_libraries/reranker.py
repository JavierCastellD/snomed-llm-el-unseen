from sentence_transformers.cross_encoder import CrossEncoder

class Reranker():
    """Wrap class for a CrossEncoder from the sentence_transformers library.
    
    Attributes:
        cross_encoder (CrossEncoder):
            CrossEncoder used to rerank the concepts.
    """
    def __init__(self, cross_encoder : CrossEncoder):
        """Initializes the class by assigning the CrossEncoder as an attribute.
        
        Parameters:
            cross_encoder (CrossEncoder):
                CrossEncoder from sentence_transformers used to rerank concepts.
        """
        self.cross_encoder = cross_encoder
    

    def rerank_concepts(self, sentence : str, options : list[str], options_ids : list[int], n : int = 10) -> tuple[list[str], list[int], list[float]]:
        """Method that reorders the concepts according to their similarity to the sentence and the reranker.
        
        Parameters:
            sentence (str):
                Text to be used as basis for the reranking.
            options (list[str]):
                List of options to rerank.
            option_ids (list[int]):
                List of corresponding IDs to the options.
            n (int):
                Number of concepts to return.

        Returns:
            A tuple of the top n made up by the options, option_ids, and scores according to the reranker.
        """
        # Rerank the options
        ranks = self.cross_encoder.rank(query=sentence, documents=options)

        # Extract the corresponding options, IDs and scores
        top_n_options = []
        top_n_options_ids = []
        top_n_scores = []

        for rank in ranks[:n]:
            top_n_options.append(options[rank['corpus_id']])
            top_n_options_ids.append(options_ids[rank['corpus_id']])
            top_n_scores.append(rank['score'])
        
        return top_n_options, top_n_options_ids, top_n_scores
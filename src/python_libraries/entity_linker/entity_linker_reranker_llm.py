from .entity_linker_llm import EntityLinkerLLM
from ..llm_queries import LLMQueryHelper
from ..sentence_ner import Entity
from ..reranker import Reranker
from ..snomed import Snomed, SnomedEmbedder

class EntityLinkerRerankerLLM(EntityLinkerLLM):
    """Entity Linker LLM subclass that uses LLMs and embeddings to link entities to the corresponding concepts and a Reranker to order the potential candidates.
    
    Attributes:
        reranker (Reranker):
            Reranker object to order the potential candidates obtained from the embeddings.
        number_of_options (int):
            Number of candidates for the reranker to reorder.
        rerank_top_n (int):
            Number of candidates to return.
    """
    def __init__(self, snomed : Snomed, snomed_embedder : SnomedEmbedder, llm_query : LLMQueryHelper, reranker : Reranker, 
                 number_of_options : int = 50, rerank_top_n : int = 10, disambiguate_abbreviations : bool = True, 
                 llm_for_el : bool = True, rephrase : bool = False, replace_span : bool = True, use_fsn : bool = False, 
                 choose_by_frequency : bool = False, threshold : float = None, spanish_version : bool = False):
        """Initializes the attributes of the class.
        
        Parameters:
            snomed (Snomed):
                Snomed object that contains information about the terminology.
            snomed_embedder (SnomedEmbedder):
                SnomedEmbedder object that uses embeddings for SNOMED CT.
            llm_query (LLMQueryHelper):
                LLMQueryHelper object that helps perform the queries to the LLMs.
            reranker (Reranker):
                Reranker object to order the potential candidates obtained from the embeddings.
            number_of_options (int):
                Number of candidates for the reranker to reorder.
            rerank_top_n (int):
                Number of candidates to return.
            disambiguate_abbreviations (bool):
                Whether or not to disambiguate the abbreviations spans found.
            llm_for_el (bool):
                Whether or not to use the LLM for choosing the corresponding concept. If set to false, the most similar one according to cosine similarity will be selected.
            rephrase (bool):
                Whether or not to use the LLM to rephrase the spans detected according to the sentence. Defaults to False.
            replace_span (bool):
                Whether to replace the disambiguated or rephrased span in the original sentence text for the LLM. Defaults to True.
            use_fsn (bool):
                Whether or not to use the FSN of the most similar concept instead of the detected synonym. Defaults to False.
            choose_by_frequency (bool):
                Whether to choose the concept as the most frequent among the top 10. This is only used when the LLM is not used for EL or when it does not choose an option. Defaults to False.
            threshold (float): 
                Threshold value to ignore entities. Value between 0 and 1.0. Defaults to None.
            spanish_version (bool):
                Whether to use the Spanish version of the prompts. Otherwise, it uses the English prompts. Defaults to False.    
        """
        super().__init__(snomed=snomed, snomed_embedder=snomed_embedder, llm_query=llm_query, disambiguate_abbreviations=disambiguate_abbreviations, llm_for_el=llm_for_el,
                         rephrase=rephrase, replace_span=replace_span, use_fsn=use_fsn, choose_by_frequency=choose_by_frequency, threshold=threshold, spanish_version=spanish_version)
        self.reranker = reranker
        self.number_of_options = number_of_options
        self.rerank_top_n = rerank_top_n

    def _obtain_concepts_options(self, span : str, entity : Entity):
        """Method that returns a list of the most appropiate concepts to be chosen
        for a given entity and span.
        
        Parameters:
            span (str):
                String that represents the span for which to find the most appropiate concepts.
            entity (Entity):
                Entity related to the span.
        
        Returns:
            Returns a list of triples (SCT-ID, sim_value, name), where the first element is the ID of the SNOMED concept, the second is the similarity score, and the third one is the corresponding name for the ID.
        """
        return self.snomed_embedder.get_most_similar_concept(span, n=self.number_of_options)

    def _reorder_ids_names(self, concept_ids : list[int], concept_names : list[str], span : str, entity : Entity = None,
                          return_top_scores : bool = False) -> tuple[list[int]|list[str]]|tuple[list[int]|list[str]|list[float]]:
        """Method that orders the list of IDs and list of names.
        
        Parameters:
            concept_ids (list[int]):
                List of IDs of the concepts.
            concept_names (list[str]):
                List of names of the concepts.
            span (str):
                String that represents the span for which to find the most appropiate concepts.
            entity (Entity):
                Entity related to the span. It can be None.
            return_top_scores (bool):
                Whether to return the top scores according to the reranker or not. Defaults to False.

        Returns:
            A list of concept_ids and a list of concept_names. If return_top_scores is set to True, an additional list
            of scores is returned.
        """
        top_fsns, top_ids, top_scores = self.reranker.rerank_concepts(span, concept_names, concept_ids, n=50)

        # Set the reranker options
        if entity is not None:
            entity.other['reranker'] = top_ids
            entity.other['reranker_scores'] = top_scores
        
        if return_top_scores:
            return top_ids[:self.rerank_top_n], top_fsns[:self.rerank_top_n], top_scores[:self.rerank_top_n]
        else:
            return top_ids[:self.rerank_top_n], top_fsns[:self.rerank_top_n]
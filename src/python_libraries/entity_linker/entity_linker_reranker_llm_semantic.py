from .entity_linker_llm_semantic import EntityLinkerLLMSemantic
from .entity_linker_reranker_llm import EntityLinkerRerankerLLM
from ..llm_queries import LLMQueryHelper
from ..sentence_ner import Entity
from ..reranker import Reranker
from ..snomed import Snomed, SnomedEmbedder

class EntityLinkerRerankerLLMSemantic(EntityLinkerRerankerLLM, EntityLinkerLLMSemantic):
    """Entity Linker subclass that uses LLMs and embeddings to link entities to the corresponding concepts and a Reranker to order the potential candidates. 
    Additionally, it uses the semantic type or hierarchy to help when searching for similar concepts.
    
    Attributes:
        ner_type2hierarchy (dict[str, str]):
            Dictionary to map the NER type assigned to the Entity to the concept type assigned in the SnomedEmbedder. 
            By default assigns 'body', 'fin', and 'pro' tags to the corresponding names in SNOMED CT.
    """
    def __init__(self, snomed : Snomed, snomed_embedder : SnomedEmbedder, llm_query : LLMQueryHelper, reranker : Reranker,
                 number_of_options : int = 50, rerank_top_n : int = 10,  ner_type2hierarchy : dict[str, str] = None,
                 disambiguate_abbreviations : bool = True, llm_for_el : bool = True, rephrase : bool = False, 
                 replace_span : bool = True, use_fsn : bool = False, choose_by_frequency : bool = False, threshold : float = None,
                 spanish_version : bool = False):
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
            ner_type2hierarchy (dict[str, str]):
                Dictionary to map the NER type assigned to the Entity to the concept type assigned in the SnomedEmbedder. 
                By default assigns 'body', 'fin', and 'pro' tags to the corresponding names in SNOMED CT. 
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
        """
        super().__init__(snomed=snomed, snomed_embedder=snomed_embedder, llm_query=llm_query, reranker=reranker, disambiguate_abbreviations=disambiguate_abbreviations, 
                         llm_for_el=llm_for_el, rephrase=rephrase, replace_span=replace_span, use_fsn=use_fsn, choose_by_frequency=choose_by_frequency,
                         threshold=threshold, number_of_options=number_of_options, rerank_top_n=rerank_top_n, spanish_version=spanish_version)
        
        if ner_type2hierarchy is not None:
            self.ner_type2hierarchy = ner_type2hierarchy
        else:
            self.ner_type2hierarchy = {'body' : 'Body structure', 
                                       'fin' : 'Clinical finding', 
                                       'pro' : 'Procedure'} 

    def _obtain_concepts_options(self, span : str, entity : Entity):
        """Method that returns a list of the most appropiate concepts to be chosen for a given entity and span.
        
        Parameters:
            span (str):
                String that represents the span for which to find the most appropiate concepts.
            entity (Entity):
                Entity related to the span.
        
        Returns:
            Returns a list of triples (SCT-ID, sim_value, name), where the first element is the ID of the SNOMED concept, the second is the similarity score, and the third one is the corresponding name for the ID.
        """
        hierarchy = self.ner_type2hierarchy[entity.ner_type] 
            
        entity.other['hierarchy'] = hierarchy

        return self.snomed_embedder.get_most_similar_concept(span, n=self.number_of_options, concept_type=hierarchy)
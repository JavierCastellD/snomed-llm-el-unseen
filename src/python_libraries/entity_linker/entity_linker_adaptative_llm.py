import re
from typing import Any

from .entity_linker_reranker_llm_semantic import EntityLinkerRerankerLLMSemantic
from ..llm_queries import LLMQueryHelper
from ..sentence_ner import Sentence, Entity
from ..reranker import Reranker
from ..snomed import Snomed, SnomedEmbedder

TRAINING_BIAS = 0.05

def static_preprocess(text):
    t = text.lower()
    t = t.replace("\n", "")
    t = re.sub("[^a-z]", "", t)
    t = re.sub("\s+", " ", t)
    t = t.strip()
    return t

class EntityLinkerAdaptiveLLM(EntityLinkerRerankerLLMSemantic):
    """Entity Linker subclass that uses embeddings, a cross-encoder reranker, and an LLM to link
    entities to the corresponding concepts. The LLM step is skipped adaptively when the embedding
    confidence exceeds a configurable threshold, reducing API calls for high-confidence predictions.

    Attributes:
        threshold_for_llm (float):
            Embedding confidence threshold above which the LLM step is skipped. None to disable.
        apply_threshold_for_llm (bool):
            Whether the LLM threshold is active.
    """
    def __init__(self, snomed : Snomed, snomed_embedder : SnomedEmbedder, llm_query : LLMQueryHelper, reranker : Reranker, dictionary_options : dict[str],
                 disambiguate_abbreviations : bool = True, llm_for_el : bool = True, rephrase : bool = False, replace_span : bool = True, use_fsn : bool = False, 
                 threshold : float = None, number_of_options : int = 50, rerank_top_n : int = 10,  ner_type2hierarchy : dict[str, str] = None,
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
            use_reranker (bool):
                Whether to use the reranker or not for the entity linking.
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
            threshold (float): 
                Threshold value to ignore entities. Value between 0 and 1.0. Defaults to None.
            number_of_options (int):
                Number of candidates for the reranker to reorder.
            rerank_top_n (int):
                Number of candidates to return.
            ner_type2hierarchy (dict[str, str]):
                Dictionary to map the NER type assigned to the Entity to the concept type assigned in the SnomedEmbedder. 
                By default assigns 'body', 'fin', and 'pro' tags to the corresponding names in SNOMED CT.
            spanish_version (bool):
                Whether to use the Spanish version of the prompts. Otherwise, it uses the English prompts. Defaults to False. 
        """
        super().__init__(snomed=snomed, snomed_embedder=snomed_embedder, llm_query=llm_query, reranker=reranker, disambiguate_abbreviations=disambiguate_abbreviations, 
                         llm_for_el=llm_for_el, rephrase=rephrase, replace_span=replace_span, use_fsn=use_fsn, threshold=threshold, number_of_options=number_of_options, 
                         rerank_top_n=rerank_top_n, ner_type2hierarchy=ner_type2hierarchy, spanish_version=spanish_version)
        
        self.use_reranker = True
        self.threshold_for_llm = dictionary_options['threshold_llm'] if 'threshold_llm' in dictionary_options else None 
        self.apply_threshold_for_llm = self.threshold_for_llm is not None


    def link_entities_from_sentence(self, sentence : Sentence):
        """Method that finds the relevant concept in the KG, ontology or vocabulary for the Entities
        in the Sentence. The concepts are assigned to the Entities.
        
        Parameters:
            sentence (Sentence):
                Sentence object that contains the entities to be linked, as well as additional information about the sentence.
        """
        # Obtain the section
        section = sentence.section

        # Obtain the sentence text
        sentence_text = sentence.text

        for entity in sentence.entities:
            # Obtain the span
            span = entity.text

            # If the abbreviation disambiguation flag is set to true
            if self.disambiguate_abbreviations:
                new_span = self._resolve_disambiguation(sentence_text, span, section)
                entity.other['disabbrevation'] = new_span
            else:
                new_span = span

            if self.replace_span:
                start = entity.start
                end = entity.end
                new_sentence_text = sentence_text[:start] + new_span + sentence_text[end:]
                
                entity.other['new_context'] = new_sentence_text
            else:
                new_sentence_text = sentence_text

            entity.other['context'] = sentence_text

            if self.rephrase:
                 new_span = self._rephrase_span(new_sentence_text, span, section)
                 entity.other['rephrasing'] = new_span
            
            # Obtain the set of similar concepts through cosine similarity
            concepts = self._obtain_concepts_options(span=new_span, entity=entity)
            
            concept_ids_orig = []
            sim_vals = []
            names = []
            for sct_id, sim_val, name in concepts:
                concept_ids_orig.append(sct_id)
                sim_vals.append(sim_val)
                names.append(name)

            if self.use_fsn:
                names = [self.snomed.get_fsn(sct_id) for sct_id in concept_ids_orig]

            # Set the options
            entity.other['original_concepts'] = concept_ids_orig
            entity.other['original_names'] = names
            entity.set_options(concept_ids_orig)

            # Get the confidence value for the embedding option
            confidence_embedding = sim_vals[0]

            entity.set_confidence(confidence_embedding)

            # If we are using a threshold and the embedding it is not good enough
            # we skip it
            if self.apply_threshold and confidence_embedding < self.threshold:
                entity.set_label("-1")
            else:
                # If we want to find the option, we might want to use a reranker to
                # reorder the options
                if self.use_reranker:
                    concept_ids, concept_names, top_scores = self._reorder_ids_names(concept_ids_orig, names, new_span, entity, return_top_scores=True)
                    confidence_reranker = top_scores[0]
                else:
                    concept_ids = concept_ids_orig
                    concept_names = names
                    confidence_reranker = confidence_embedding

                # Set confidence reranker
                entity.other['confidence_reranker'] = confidence_reranker

                # And we either let the LLM choose or take the top option
                if self.llm_for_el:
                    needs_disambiguation = sim_vals[0] == sim_vals[1]

                    # If the confidence of the embedding is greater than the threshold for LLM, we ignore the LLM
                    if not needs_disambiguation and (self.apply_threshold_for_llm and confidence_embedding >= self.threshold_for_llm):
                        concept_id = None
                    else:
                        # Disambiguate potential duplicates, i.e, infiltration (procedure) vs infiltration (morphologic abnormality)
                        concept_names = self._disambiguate_concepts_names(concept_ids, concept_names)
                        entity.other['names_for_el'] = concept_names
                        entity.other['LLM_chose'] = 'YES'

                        # Find the most similar concept by querying the LLM
                        concept_id = self._find_concept_id(new_sentence_text, span=new_span, section=section, concept_ids=concept_ids, concept_names=concept_names)
                        
                        # Set the options
                        entity.set_options(concept_ids)
                        entity.set_label(concept_id)

                    # If the LLM did not choose or we do not want to use an LLM
                    if not self.llm_for_el or concept_id is None:
                        entity.set_label(concept_ids_orig[0]) # Using embedding option
                        # entity.set_label(concept_ids[0]) # Using reranker option
                        entity.other['LLM_chose'] = 'NO'

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
        if self.spanish_version:
            return self.snomed_embedder.get_most_similar_concept(span, n=self.number_of_options, concept_type=None)
        else:
            hierarchy = self.ner_type2hierarchy[entity.ner_type] 
                
            entity.other['hierarchy'] = hierarchy

            return self.snomed_embedder.get_most_similar_concept(span, n=self.number_of_options, concept_type=hierarchy)
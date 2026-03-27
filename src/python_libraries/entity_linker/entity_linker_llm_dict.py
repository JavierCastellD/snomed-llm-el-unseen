import re
from typing import Any

from .entity_linker_reranker_llm_semantic import EntityLinkerRerankerLLMSemantic
from ..llm_queries import LLMQueryHelper
from ..annotated_datasets.MIMIC_IV_annotated_dataset import get_deanonymized_section_name
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

class EntityLinkerLLMDictionary(EntityLinkerRerankerLLMSemantic):
    """Entity Linker subclass that uses LLMs, embeddings, and a reranker to link entities to the corresponding concepts by also using a dictionary-based approach.
    
    Attributes:
        span_dictionary (dict[str, Any]):
            Dictionary that serves as knowledge base to guide the entity linking process.
        dictionary_with_sections (bool):
            Whether the dictionary is divided into sections.
        dictionary_with_multiple_options (bool):
            Whether each entry in the dictionary is associated with multiple options or not.
        use_reranker (bool):
            Whether to use the reranker or not for the entity linking.
        trust_training (bool):
            Whether to add a training bias to the confidence value of the predictions.
        threshold_for_dictionary (float):
            Threshold to apply to use the prediction of the regular pipeline over the dictionary's prediction. It can be None.
        apply_threshold_for_dictionary (bool):
            Whether to apply threshold for the dictionary or not.
        training_concepts (list[int]):
            List of concepts seen during training. It is extracted from the span_dictionary.
        dictionary_options (dict[str]):
            Dictionary that contains the configuration for the dictionary approach.
    """
    def __init__(self, snomed : Snomed, snomed_embedder : SnomedEmbedder, llm_query : LLMQueryHelper, reranker : Reranker, span_dictionary : dict[str, Any], dictionary_options : dict[str],
                 trust_training : bool = False, threshold_for_dictionary : float = None, use_reranker : bool = True,
                 disambiguate_abbreviations : bool = True, llm_for_el : bool = True, rephrase : bool = False, replace_span : bool = True, use_fsn : bool = False, 
                 threshold : float = None, number_of_options : int = 50, rerank_top_n : int = 10,  ner_type2hierarchy : dict[str, str] = None,
                 training_concepts : list[int] = None, spanish_version : bool = False):
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
            span_dictionary (dict[str, Any]):
                Dictionary that serves as knowledge base to guide the entity linking process.
            trust_training (bool):
                Whether to add a training bias to the confidence value of the predictions.
            threshold_for_dictionary (float):
                Threshold to apply to use the prediction of the regular pipeline over the dictionary's prediction. It can be None.
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
        """
        super().__init__(snomed=snomed, snomed_embedder=snomed_embedder, llm_query=llm_query, reranker=reranker, disambiguate_abbreviations=disambiguate_abbreviations, 
                         llm_for_el=llm_for_el, rephrase=rephrase, replace_span=replace_span, use_fsn=use_fsn, threshold=threshold, number_of_options=number_of_options, 
                         rerank_top_n=rerank_top_n, ner_type2hierarchy=ner_type2hierarchy, spanish_version=spanish_version)
        
        self.use_reranker = use_reranker
        self.trust_training = trust_training

        self.threshold_for_dictionary = threshold_for_dictionary
        self.apply_threshold_for_dictionary = threshold_for_dictionary is not None
        self.threshold_for_reranker = dictionary_options['threshold_reranker']
        self.apply_threshold_for_reranker = self.threshold_for_reranker is not None
        self.threshold_for_llm = dictionary_options['threshold_llm'] if 'threshold_llm' in dictionary_options else None 
        self.apply_threshold_for_llm = self.threshold_for_llm is not None
        self.span_dictionary = span_dictionary
        self.dictionary_with_sections = False
        self.dictionary_with_multiple_options = False

        if training_concepts is None:
            # This is when we are using a dictionary with multiple options per span
            if isinstance(list(span_dictionary.values())[0], list):
                self.dictionary_with_multiple_options = True
                self.training_concepts = list(set(cid for cids in span_dictionary.values() for cid in cids))
            # This is when we are using a dictionary separated per section
            elif isinstance(list(span_dictionary.values())[0], dict):
                self.dictionary_with_sections = True

                # Which may contain multiple options per span
                span_subdictionary = list(span_dictionary.values())[0]
                if isinstance(list(span_subdictionary.values())[0], list):
                    self.dictionary_with_multiple_options = True
                    self.training_concepts = list(set([cid for dictionary in span_dictionary.values() for cids in dictionary.values() for cid in cids]))
                # or just the most frequent option per span
                else:
                    self.training_concepts = list(set([cid for dictionary in span_dictionary.values() for cid in dictionary.values()]))
            # This is when we are using a dictionary with the most frequent option per span
            else:
                self.training_concepts = list(set(span_dictionary.values()))
        else:
            self.training_concepts = training_concepts

        # Extract the option configuration from the options
        self.dictionary_options = dictionary_options

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
            # If it is part of the training set, we might want to add some bias to its confidence
            confidence_embedding = sim_vals[0] + TRAINING_BIAS if self.trust_training else sim_vals[0]

            entity.set_confidence(confidence_embedding)

            # Find if it is worth to use the dictionary, as the option from the embedding
            # might be the most interesting
            if self.apply_threshold_for_dictionary and confidence_embedding >= self.threshold_for_dictionary:
                dictionary_option = None
            else:
                new_span_for_method = new_span

                # Whether to use the new span or not
                if self.dictionary_options['use_new_span']:
                    span_for_method = new_span
                    new_span_for_method = None
                else:
                    span_for_method = span

                # Choose which dictionary option to apply
                match self.dictionary_options['method']:
                    case 'frequent':
                        dictionary_option = self._use_most_frequent_dictionary(span=span_for_method, section=section, new_span=new_span_for_method,
                                                                               use_both_spans_for_dict=self.dictionary_options['use_both_spans_for_dict'])
                    case 'llm':
                        dictionary_option = self._use_dictionary_with_llm(span=span_for_method, sentence_text=new_sentence_text, 
                                                                          section=section, new_span=new_span_for_method, 
                                                                          use_both_spans_for_dict=self.dictionary_options['use_both_spans_for_dict'],
                                                                          use_synonyms=self.dictionary_options['use_synonyms'],
                                                                          use_new_span_for_llm=self.dictionary_options['use_new_span_for_llm'])
                    case 'emb_llm':
                        dictionary_option = self._use_dictionary_with_embeddings_and_llms(span=span_for_method, sentence_text=new_sentence_text,
                                                                                          section=section, new_span=new_span_for_method,
                                                                                          use_both_spans_for_dict=self.dictionary_options['use_both_spans_for_dict'],
                                                                                          top_concepts_embeddings=self.dictionary_options['top_concepts_embeddings'], 
                                                                                          use_synonyms=self.dictionary_options['use_synonyms'],
                                                                                          trust_validation=self.dictionary_options['trust_validation'],
                                                                                          use_new_span_for_validation_and_llm=self.dictionary_options['use_new_span_for_validation_and_llm'],
                                                                                          rerank_validation=self.dictionary_options['rerank_validation'],
                                                                                          top_validation_reranked=self.dictionary_options['top_validation_reranked'])
                    case _:
                        print('No valid option detected for dictionary')
                        dictionary_option = None
            
            # dictionary_option = self._use_dictionary_with_embeddings_and_reranker(span, sentence_text, section) # Similar to the previous one, but looking at less concepts using a reranker
            # dictionary_option = self._use_dictionary_type(span, sentence_text, section) # The dictionary is segmentized by hierarchy
            # dictionary_option = self._use_dictionary_with_reranker(span, sentence_text, section) # If there is more than one option, the reranker chooses

            # If the span was not found in the dictionary or if it was deemed not worth using
            # we ignore the option from the dictionary
            entity.other['dictionary_chose'] = 'YES' if dictionary_option is not None else 'NO'

            # If the dictionary did not found an option, we need to use another one
            if dictionary_option is None:
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

                    if self.apply_threshold_for_reranker and confidence_reranker < self.threshold_for_reranker:
                        entity.set_label("-1")
                    else:
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
                            #entity.set_label(concept_ids[0])
                            entity.other['LLM_chose'] = 'NO'
            else:
                entity.set_label(dictionary_option)    


        # Those that did not meet the threshold, are removed
        if self.apply_threshold:
            for entity in sentence.entities.copy():
                if entity.label == "-1":
                    sentence.entities.remove(entity)
                    
    def _prepare_names_from_snomed_ids(self, option_ids : list[int], use_synonyms : bool = True) -> tuple[list[int], list[str]]:
        """Auxiliary method to obtain the names of a list of SNOMED CT IDs. If use_synonyms is set to True, all the synonyms
        of each concept are used, rather than just the Fully Specified Name.
        
        Parameters:
            option_ids (list[int]):
                List of SNOMED CT ids.
            use_synonyms (bool):
                Whether to use the synonyms of the concepts or just the FSN.
        
        Returns:
            A tuple of two lists: one with SNOMED CT ids, and another one with the corresponding name. If using synonyms, the list
            with IDs will contain duplicates, although there won't be duplicate pairs (sct_id, sct_name).
        """
        concept_names_tuples = []
        for cid in option_ids:
            if use_synonyms:
                concept_names_tuples += [(cid, syn) for syn in self.snomed.get_descriptions(cid)]
            else:
                snomed_fsn = self.snomed.get_fsn(cid)
                if snomed_fsn != '':
                    concept_names_tuples.append((cid, snomed_fsn))

        # Remove duplicate pairs
        concept_names_tuples = list(set(concept_names_tuples))

        # Separate into two lists
        concept_ids, concept_names = zip(*concept_names_tuples)
        concept_ids = list(concept_ids)
        concept_names = list(concept_names)

        return concept_ids, concept_names

    def _let_llm_choose(self, span : str, sentence_text : str, section : str,
                        option_ids : list[int], use_synonyms : bool = True) -> int|None:
        """Auxiliary method to use an LLM to choose the option among a set of SNOMED CT concepts that best
        corresponds to a span considering the sentence and section as context.
        
        Parameters:
            span (str):
                Span to link to a concept.
            sentence_text (str):
                Sentence where the span was found to serve as context.
            section (str):
                Section where the sentence and span were found.
            option_ids (list[int]):
                List of SNOMED CT IDs from which to choose the corresponding concept.
            use_synonyms (bool):
                Whether to use the synonyms or just the FSN of the list of concepts.

        Returns:
            An int that represents the ID of a SNOMED CT concept or None.
        """
        # Obtain the names from the SNOMED CT ids provided from the dictionary
        concept_ids, concept_names = self._prepare_names_from_snomed_ids(option_ids, use_synonyms)        

        # Disambiguate terms
        concept_names = self._disambiguate_concepts_names(concept_ids, concept_names)

        # Find the most similar concept by querying the LLM
        concept_id = self._find_concept_id(sentence_text=sentence_text, span=span, section=section, 
                                           concept_ids=concept_ids, concept_names=concept_names)
        
        # This concept_id might be None if the LLM did not found a valid option 
        return concept_id

    def _let_reranker_choose(self, span : str, option_ids : list[int], use_synonyms : bool = True) -> int:
        """Auxiliary method to use a reranker to choose the option among a set of SNOMED CT concepts that best
        corresponds to a span.
        
        Parameters:
            span (str):
                Span to link to a concept.
            option_ids (list[int]):
                List of SNOMED CT IDs from which to choose the corresponding concept.
            use_synonyms (bool):
                Whether to use the synonyms or just the FSN of the list of concepts.
                
        Returns:
            An int that represents the ID of a SNOMED CT concept.
        """
        # Obtain the names from the SNOMED CT ids provided from the dictionary
        concept_ids, concept_names = self._prepare_names_from_snomed_ids(option_ids=option_ids, use_synonyms=use_synonyms)

        # Reorder them according to the reranker
        concept_ids, concept_names, top_scores = self._reorder_ids_names(concept_ids=concept_ids, concept_names=concept_names, span=span, return_top_scores=True)

        return concept_ids[0]

    def _get_dictionary_entry(self, span : str, section : str, new_span : str = None) -> list[int]:
        """Auxiliary method to get the entry from the dictionary for a given span. If the dictionary is divided into sections,
        the section parameter is used. If new_span is not None, it will return the list of IDs associated to both span and
        new_span. There are no duplicates in the returned list.
        
        Parameters:
            span (str):
                Span to use to get the dictionary values.
            section (str):
                Section associated to the span.
            new_span (str):
                Additional span to get dictionary values.
        
        Returns:
            A list of int that represents the SNOMED CT IDs associated with the span and, optionally, to new_span. An empty
            list will be returned if there is no entry in the dictionary associated with the span(s).
        """
        span_entry = []
        # We might have a dictionary that uses sections
        if self.dictionary_with_sections:
            deanonymized_section_name = get_deanonymized_section_name(section)
            if deanonymized_section_name in self.span_dictionary:
                span_dictionary = self.span_dictionary[deanonymized_section_name]
            else:
                return []
        else:
            span_dictionary = self.span_dictionary 

        span = span.lower() if not self.dictionary_options['preprocess_span'] else static_preprocess(span)
        if span in span_dictionary:
            if self.dictionary_with_multiple_options:
                span_entry += span_dictionary[span]
            else:
                span_entry.append(span_dictionary[span])

        # We might want to use both spans (normal and abbreviated) to generate options
        if new_span is not None:
            new_span = new_span.lower() if not self.dictionary_options['preprocess_new_span'] else static_preprocess(new_span)
            if new_span in span_dictionary:
                if self.dictionary_with_multiple_options:
                    span_entry += span_dictionary[new_span]
                else:
                    span_entry.append(span_dictionary[new_span])

        return list(set(span_entry))

    def _validate_dictionary_entry(self, span_entry : list[int], span : str, top_concepts_embedding : int,
                                   use_reranker : bool = False, top_concepts_reranker : int = 100) -> list[int]:
        """Auxiliary function to validate the values returned from the dictionary by using embeddings. Any concept id that
        is present among the most similar concepts from the embeddings is considered validated. If use_reranker is set to True,
        the most similar concepts are reordered.
        
        Parameters:
            span_entry (list[int]):
                List of SNOMED CT IDs obtained from the dictionary.
            span (str):
                String that represents the span used for the dictionary.
            top_concepts_embeddings (int):
                Number of similar concepts to obtain.
            use_reranker (bool):
                Whether to use the reranker for the validation.
            top_concepts_reranker (int):
                Number of concepts to look at after they are reordered by the reranker.

        Returns:
            A list of int that represents the concepts validated by the embeddings. The list will be empty if no concept was validated.
        """
        # We want to use the original span for the dictionary and the expanded
        # to get the similar concepts
        concepts_info = [(sct_id, name) for sct_id, _, name in self.snomed_embedder.get_most_similar_concept(span, n=top_concepts_embedding)]
        
        # Extract the IDs and names
        concepts = [sct_id for sct_id, _ in concepts_info]
        names = [name for _, name in concepts_info]

        if use_reranker:
            concepts, names = self._reorder_ids_names(concept_ids=concepts, concept_names=names, span=span, return_top_scores=False)

            concepts = concepts[:top_concepts_reranker]

        # We only consider the options from the dictionary that are in the top X
        # most similar concepts according to the embeddings
        concepts_in_most_similar = []
        for concept_id in span_entry:
            if concept_id in concepts:
                concepts_in_most_similar.append(concept_id)
        
        return concepts_in_most_similar

    def _use_most_frequent_dictionary(self, span : str, section : str, new_span : str = None,
                                      use_both_spans_for_dict : bool = False) -> int|None:
        """Auxiliary function to use an embedding approach based on a dictionary that only contains
        the most frequent SNOMED CT ID from the training set associated with a span.
        
        Parameters:
            span (str):
                Span for the dictionary approach.
            section (str):
                Section where the span was found.
            new_span (str):
                Additional span for the dictionary approach. It can be None.

        Returns:
            An int that represents the SNOMED CT ID of the concept or None if no valid entry was found.
        """
        if not use_both_spans_for_dict:
            new_span = None

        span_entry = self._get_dictionary_entry(span=span, section=section, new_span=new_span)

        if len(span_entry) == 0:
            return None
        return span_entry[0]

    def _use_dictionary_with_llm(self, span : str, sentence_text : str, section : str,
                                 new_span : str = None, use_both_spans_for_dict : bool = False,
                                 use_synonyms : bool = True, use_new_span_for_llm : bool = True):
        """Auxiliary function to use an embedding approach based on a dictionary that might contain
        multiple IDs associated with an entry. To resolve that disambiguation, an LLM is asked to choose
        from those concepts.
        
        Parameters:
            span (str):
                Span for the dictionary approach.
            sentence_text (str):
                String that represents the sentence where the span was found.
            section (str):
                Section where the span was found.
            new_span (str):
                Additional expanded version of the span. It can be None.
            use_both_spans_for_dict (bool):
                Whether to use span and new_span for the dictionary entry.
            use_synonyms (bool):
                Whether to use the synonyms for the IDs from the dictionary or just the Fully Specified Name.
            use_new_span_for_llm (bool):
                Whether to use new_span for the LLM disambiguation rather than span. Defaults to True. If new_span
                is None, it works as if set to False.

        Returns:
            An int that represents the SNOMED CT ID of the concept or None if no ID was found.
        """
        # We might want to have one span for the dictionary, and a different one for the LLM
        if use_both_spans_for_dict:
            new_span_for_dict = new_span
        else:
            new_span_for_dict = None

        # Obtain the entry for the span
        span_entry = self._get_dictionary_entry(span=span, section=section, new_span=new_span_for_dict)
        
        if len(span_entry) == 0:
            return None
        elif len(span_entry) == 1:
            return span_entry[0]
        else:
            # This is in case we want to use the original span for the dictionary
            # and the new span for the llm
            if use_new_span_for_llm and new_span is not None:
                span_for_llm = new_span
            else:
                span_for_llm = span

            return self._let_llm_choose(span=span_for_llm, sentence_text=sentence_text, section=section,
                                        option_ids=span_entry, use_synonyms=use_synonyms)

    def _use_dictionary_with_embeddings_and_llms(self, span : str, sentence_text : str, section : str,
                                                new_span : str = None, use_both_spans_for_dict : bool = False,
                                                top_concepts_embeddings : int = 100, use_synonyms : bool = True, 
                                                trust_validation : bool = True, use_new_span_for_validation_and_llm : bool = True,
                                                rerank_validation : bool = False, top_validation_reranked : int = 100):
        """Auxiliary function to use an embedding approach based on a dictionary that might contain multiple IDs. Only those
        concepts validated are considered and any disambiguation is resolved using an LLM. A concept is validated if it is among
        the most similar concepts according to the embeddings.
        
        Parameters:
            span (str):
                Span for the dictionary approach.
            sentence_text (str):
                String that represents the sentence where the span was found.
            section (str):
                Section where the span was found.
            new_span (str):
                Additional expanded version of the span. It can be None.
            use_both_spans_for_dict (bool):
                Whether to use span and new_span for the dictionary entry.
            top_concepts_embeddings (int):
                Number of similar concepts to obtain from embeddings.
            use_synonyms (bool):
                Whether to use the synonyms for the IDs from the dictionary or just the Fully Specified Name.
            trust_validation (bool):
                Whether to trust the validation when none of the concepts from the dictionary are among the most similars.
            use_new_span_for_validation_and_llm (bool):
                Whether to use new_span for the LLM disambiguation rather than span, as well as for the validation. Defaults to True.
                If new_span is None, it works as if set to False.
            rerank_validation (bool):
                Whether to use a reranker for the validation. If set to True, the most similar concepts are reordered.
            top_validation_reranked (int):
                Number of concepts to look at after they are reordered by the reranker.

        Returns:
            An int that represents the SNOMED CT ID of the concept or None if no ID was found.
        """
        # We might want to have one span for the dictionary, and a different one for the LLM
        if use_both_spans_for_dict:
            new_span_for_dict = new_span
        else:
            new_span_for_dict = None

        # Obtain the entry for the span
        span_entry = self._get_dictionary_entry(span=span, section=section, new_span=new_span_for_dict)
        
        if len(span_entry) == 0:
            return None
        
        # We use the embeddings to validate those concepts
        if use_new_span_for_validation_and_llm and new_span is not None:
            span_for_validation = new_span
        else:
            span_for_validation = span

        concepts_in_most_similar = self._validate_dictionary_entry(span_entry=span_entry, span=span_for_validation, top_concepts_embedding=top_concepts_embeddings,
                                                                   use_reranker=rerank_validation, top_concepts_reranker=top_validation_reranked)

        if len(concepts_in_most_similar) == 1:
            return concepts_in_most_similar[0]
        elif len(concepts_in_most_similar) > 1:
            return self._let_llm_choose(span=span_for_validation, sentence_text=sentence_text, section=section,
                                            option_ids=concepts_in_most_similar, use_synonyms=use_synonyms)

        else:
            # If we trust the validation, we return None so that other method is used to obtain
            # the concept id
            if trust_validation:
                return None
            # Otherwise, we let the LLM choose from among the dictionary entries
            else:
                return self._let_llm_choose(span=span_for_validation, sentence_text=sentence_text, section=section,
                                            option_ids=span_entry, use_synonyms=use_synonyms)
    
    def _use_dictionary_with_embeddings_and_reranker(self, span : str, section : str, new_span : str = None, 
                                                    use_both_spans_for_dict : bool = False, top_concepts_embeddings : int = 100, 
                                                    use_synonyms : bool = True, trust_validation : bool = True,
                                                    use_new_span_for_validation_and_reranker : bool = True,
                                                    rerank_validation : bool = False, top_validation_reranked : int = 100):
        """Auxiliary function to use an embedding approach based on a dictionary that might contain multiple IDs. Only those
        concepts validated are considered and any disambiguation is resolved using a reranker. A concept is validated if it is among
        the most similar concepts according to the embeddings.
        
        Parameters:
            span (str):
                Span for the dictionary approach.
            section (str):
                Section where the span was found.
            new_span (str):
                Additional expanded version of the span. It can be None.
            use_both_spans_for_dict (bool):
                Whether to use span and new_span for the dictionary entry.
            top_concepts_embeddings (int):
                Number of similar concepts to obtain from embeddings.
            use_synonyms (bool):
                Whether to use the synonyms for the IDs from the dictionary or just the Fully Specified Name.
            trust_validation (bool):
                Whether to trust the validation when none of the concepts from the dictionary are among the most similars.
            use_new_span_for_validation_and_reranker (bool):
                Whether to use new_span for the reranker disambiguation rather than span, as well as for the validation. Defaults to True.
                If new_span is None, it works as if set to False.
            rerank_validation (bool):
                Whether to use a reranker for the validation. If set to True, the most similar concepts are reordered.
            top_validation_reranked (int):
                Number of concepts to look at after they are reordered by the reranker.

        Returns:
            An int that represents the SNOMED CT ID of the concept or None if no ID was found.
        """
        if use_both_spans_for_dict:
            new_span_for_dict = new_span
        else:
            new_span_for_dict = None

        # Obtain the entry for the span
        span_entry = self._get_dictionary_entry(span=span, section=section, new_span=new_span_for_dict)
        
        if len(span_entry) == 0:
            return None
        
        # We use the embeddings to validate those concepts
        if use_new_span_for_validation_and_reranker and new_span is not None:
            span_for_validation = new_span
        else:
            span_for_validation = span

        concepts_in_most_similar = self._validate_dictionary_entry(span_entry=span_entry, span=span_for_validation, top_concepts_embedding=top_concepts_embeddings,
                                                                   use_reranker=rerank_validation, top_concepts_reranker=top_validation_reranked)

        if len(concepts_in_most_similar) == 1:
            return concepts_in_most_similar[0]
        elif len(concepts_in_most_similar) > 1:
            
            return self._let_reranker_choose(span=span_for_validation, option_ids=concepts_in_most_similar, use_synonyms=use_synonyms)

        else:
            # If we trust the validation, we return None so that other method is used to obtain
            # the concept id
            if trust_validation:
                return None
            # Otherwise, we let the LLM choose from among the dictionary entries
            else:
                return self._let_reranker_choose(span=span_for_validation, option_ids=span_entry, use_synonyms=use_synonyms)

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
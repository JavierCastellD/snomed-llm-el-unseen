import re
import warnings

from .entity_linker import EntityLinker
from ..llm_queries.LLM_query_helper import LLMQueryHelper
from ..ner_model.sentence_ner import Sentence, Entity
from ..snomed.snomed import Snomed
from ..snomed.snomed_embedder import SnomedEmbedder

def truncate_sentence_text(sentence_text : str, start : int, end : int, max_chars : int = 128):
    """Method that truncates a sentence so that it has close to the maximum number of characters. The sentence is
    truncated around the span defined by start and end and tries not to prevent cutting words in half.
    
    Parameters:
        sentence_text (str):
            Sentence to truncate.
        start (int):
            Start of the span of interest from the sentence.
        end (int):
            End of the span of interest from the sentence.
        max_chars (int):
            Maximum number of characters desired. By default is 512.
    """
    span = sentence_text[start:end]

    # This should not happen, but just in case the span is too big
    if len(span) > max_chars:
        return span
    
    # Split the remaining characters between pre-span and post-span
    remaining_chars = max_chars - len(span)
    pre_chars = remaining_chars // 2
    post_chars = remaining_chars - pre_chars

    # Calculate the start and end of the context
    # We try to ensure that as much characters as possible are used
    context_start = max(0, start - pre_chars)

    # If some characters exceed from the start, we add it to the end
    if context_start == 0:
        exceeding_chars = pre_chars - start
        context_end = min(len(sentence_text), end + post_chars + (pre_chars - start))
    else:
        context_end = min(len(sentence_text), end + post_chars)

    # If some characters exceed from the end, we add it to the start
    if context_end == len(sentence_text) and context_start != 0:
        exceeding_chars = post_chars - (len(sentence_text) - end)
        context_start = max(0, start - (pre_chars + exceeding_chars))
    
    # We adjust so that there is no word cut
    # If the previous character is alphanumeric, then we are probably cutting some word
    while (context_start > 0 and sentence_text[context_start -1].isalnum()):
        context_start -= 1
    
    # The same applies to the character after the end
    while context_end < len(sentence_text) and sentence_text[context_end].isalnum():
        context_end += 1
    
    return sentence_text[context_start:context_end]

class EntityLinkerLLM(EntityLinker):
    """Entity Linker subclass that uses LLMs and embeddings to link entities to the corresponding concepts.
    
    Attributes:
        snomed (Snomed):
            Snomed object that contains information about the terminology.

        snomed_embedder (SnomedEmbedder):
            SnomedEmbedder object that uses embeddings for SNOMED CT.
        
        llm_query (LLMQueryHelper):
            LLMQueryHelper object that helps perform the queries to the LLMs. 

        disambiguate_abbreviations (bool):
            Whether or not to disambiguate the abbreviations spans found.

        llm_for_el (bool):
            Whether or not to use the LLM for choosing the corresponding concept. If set to false, the most similar one according to the embeddings' cosine similarity will be selected.

        rephrase (bool):
            Whether or not to use the LLM to rephrase the spans detected according to the sentence. Defaults to False.
        
        replace_span (bool):
            Whether to replace the disambiguated or rephrased span in the original sentence text for the LLM. Defaults to True.

        use_fsn (bool):
            Whether or not to use the FSN of the most similar concept instead of the detected synonym. Defaults to False.

        choose_by_frequency (bool):
            Whether to choose the concept as the most frequent among the top 10. This is only used when the LLM is not used for EL or when it does not choose an option. Defaults to False.

        apply_threshold (bool):
            Whether to apply a threshold to ignore those entities that do not meet it (and remove them from the sentence). Defaults to False.

        threshold (float): 
            Threshold value to ignore entities. Value between 0 and 1.0. Defaults to None.        
    """
    def __init__(self, snomed : Snomed, snomed_embedder : SnomedEmbedder, llm_query : LLMQueryHelper, 
                 disambiguate_abbreviations : bool = True, llm_for_el : bool = True, rephrase : bool = False,
                 replace_span : bool = True, use_fsn : bool = False, choose_by_frequency : bool = False,
                 threshold : float = None, spanish_version : bool = False):
        """Initializes the attributes of the class.
        
        Parameters:
            snomed (Snomed):
                Snomed object that contains information about the terminology.
            snomed_embedder (SnomedEmbedder):
                SnomedEmbedder object that uses embeddings for SNOMED CT.
            llm_query (LLMQueryHelper):
                LLMQueryHelper object that helps perform the queries to the LLMs. 
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
        self.snomed = snomed
        self.snomed_embedder = snomed_embedder
        self.llm_query = llm_query
        self.disambiguate_abbreviations = disambiguate_abbreviations
        self.llm_for_el = llm_for_el
        self.apply_threshold = threshold is not None
        self.threshold = threshold
        self.rephrase = rephrase
        self.replace_span = replace_span
        self.use_fsn = use_fsn
        self.choose_by_frequency = choose_by_frequency
        self.spanish_version = spanish_version

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

            # Find the most similar concept through cosine similarity
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

            # Set the options names
            entity.other['options names'] = names

            # Set the options
            entity.set_options(concept_ids_orig)

            # Set the similarity value
            entity.set_confidence(sim_vals[0])
            
            if self.apply_threshold and entity.confidence < self.threshold:
                entity.set_label("-1")
                continue

            # Get the IDs and FSNs
            concept_ids, concept_names = self._reorder_ids_names(concept_ids_orig, names, new_span, entity)

            if self.llm_for_el:
                # Disambiguate potential duplicates, i.e, infiltration (procedure) vs infiltration (morphologic abnormality)
                concept_names = self._disambiguate_concepts_names(concept_ids, concept_names)

                # Find the most similar concept by querying the LLM
                concept_id = self._find_concept_id(new_sentence_text, span=new_span, section=section, concept_ids=concept_ids, concept_names=concept_names)

                # If concept_id is None, it means that the LLM made up the response
                if concept_id is None:
                    # warnings.warn(f"LLM made up or could not find an adequate concept: {llm_concept_fsn} for span {new_span} in sentence {sentence_text}")
                    if self.choose_by_frequency:
                        counts = {}
                        for cid in concept_ids:
                            if cid not in counts:
                                counts[cid] = 0
                            counts[cid] += 1
                        counts_list = list(counts.items())
                        counts_list.sort(key=lambda x : x[1], reverse=True)
                        
                        entity.set_label(counts_list[0][0])
                    else:
                        entity.set_label(concept_ids[0])

                    entity.other['llm_chose'] = False
                else:
                    # Add the concept id to the entity
                    entity.set_label(concept_id)

                    entity.other['llm_chose'] = True
            else:
                if self.choose_by_frequency:
                    counts = {}
                    for cid in concept_ids:
                        if cid not in counts:
                            counts[cid] = 0
                        counts[cid] += 1
                    counts_list = list(counts.items())
                    counts_list.sort(key=lambda x : x[1], reverse=True)
                    
                    entity.set_label(counts_list[0][0])
                else:
                    entity.set_label(concept_ids[0])
        
        if self.apply_threshold:
            for entity in sentence.entities.copy():
                if entity.label == "-1":
                    sentence.entities.remove(entity)

    def _reorder_ids_names(self, concept_ids : list[int], concept_names : list[str], span : str, entity : Entity = None) -> tuple[list[int]|list[str]]:
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

        Returns:
            A list of concept_ids and a list of concept_names
        """
        return concept_ids, concept_names

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
        return self.snomed_embedder.get_most_similar_concept(span, n=10)

    def _find_concept_id(self, sentence_text : str, span : str, section : str, concept_ids : list[int], concept_names : list[str]) -> int:
        """Method used to identify the corresponding concept for a given span from a list of concept_ids considering the context
        and the section where it was identified.

        Parameters:
            sentence_text (str):
                String that represents the sentence that serves as context for the span.
            span (str):
                String that represents the span for which we want to identify the concept from SNOMED CT.
            section (str):
                String that represents the section from the text where the span was found.
            concept_ids (list[int]):
                List of potential concept IDs from which to select the concept.
            concept_names (list[str]):
                Corresponding list of names for the concept_ids.
        
        Returns:
            An int that represents the ID of the concept or None if none of the options was valid for the LLM or if there was an error.
        """
        if self.spanish_version:
            response_el = self.llm_query.send_request_EL_es(sentence=sentence_text, span=span, options=concept_names)
        else:
            response_el = self.llm_query.send_request_EL(sentence_text, span, section, concept_names)
        # print(f'EL request: span={span} | response={response_el} | options={concept_names} | sentence={sentence_text}')

        # If there is an error, None is returned
        if response_el is None:
            return None

        # Find the corresponding concept_id
        for cid, name in zip(concept_ids, concept_names):
            if name.lower() == response_el.lower():
                return cid
            
        # print(f'Not found {response_el} among options: {concept_names}')
        return None

    def _resolve_disambiguation(self, sentence_text : str, span : str, section : str) -> str:
        """Method used to disambiguate an abbreviation given the context provided by sentence_text and section.

        Parameters:
            sentence_text (str):
                String that represents the sentence that serves as context for the span.
            span (str):
                String that represents the span for which we want to identify the concept from SNOMED CT.
            section (str):
                String that represents the section from the text where the span was found.

        Returns:
            A string that represents the disambiguated version of the span if there is any. Otherwise, the span is returned instead.
        """
        # Find if it is an abbreviation by querying the LLM
        if self.spanish_version:
            response_ab = self.llm_query.send_request_AB_es(sentence_text, span)
        else:
            response_ab = self.llm_query.send_request_AB(sentence_text, span, section)
        
        # If there is an error, None is returned
        if response_ab is None:
            return span
        
        # Use regex to extract the new span from the response
        if '"' in response_ab:
            match = re.search(r'"(.*)"', response_ab)

            if match is None:
                #warnings.warn(f"Match not found in LLM response for abbreviation: {response_ab} for span {new_span} in sentence {sentence_text}")
                return span
            
            return match.group(1)
        
        return response_ab

    def _rephrase_span(self, sentence_text : str, span : str, section : str) -> str:
        """Method used to rephrase an span given the context provided by sentence_text and section.

        Parameters:
            sentence_text (str):
                String that represents the sentence that serves as context for the span.
            span (str):
                String that represents the span for which we want to identify the concept from SNOMED CT.
            section (str):
                String that represents the section from the text where the span was found.

        Returns:
            A string that represents the disambiguated version of the span if there is any. Otherwise, the span is returned instead.
        """
        # Find if it is an abbreviation by querying the LLM
        response_rp = self.llm_query.send_request_RP(sentence_text, span, section)
        
        # If there is an error, None is returned
        if response_rp is None:
            return span
        
        # Use regex to extract the new span from the response
        if '"' in response_rp:
            match = re.search(r'"(.*)"', response_rp)

            if match is None:
                #warnings.warn(f"Match not found in LLM response for abbreviation: {response_ab} for span {new_span} in sentence {sentence_text}")
                return span
            
            return match.group(1)
        
        return response_rp

    def _disambiguate_concepts_names(self, concept_ids : list[int], concept_names : list[str]) -> list[str]:
        """Auxiliary method to help disambiguate the name of concepts that share a name. For concepts with the same name, the semantic
        type is added between parenthesis to differentiate them.
        
        Parameters:
            concept_ids (list[int]):
                List of SNOMED CT concept IDs. It must share the order of the list of names.
            concept_names (list[str]):
                List of SNOMED CT concept names. It must share the order of the list of IDs.
                
        Returns:
            A copy of concept names with disambiguated names.
        """
        name_counts = {}
        for name in concept_names:
            if name not in name_counts:
                name_counts[name] = 0
            
            name_counts[name] += 1
        
        new_concept_names = []
        for cid, name in zip(concept_ids, concept_names):
            if name_counts[name] > 1:
                new_concept_names.append(f"{name} ({self.snomed.get_semantic_type(cid)})") 
            else:
                new_concept_names.append(name)
        
        return new_concept_names
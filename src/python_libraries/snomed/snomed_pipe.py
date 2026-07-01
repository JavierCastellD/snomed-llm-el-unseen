import warnings

from ..entity_linker.entity_linker import EntityLinker
from ..annotated_datasets.annotated_dataset import remap_annotations
from ..sentence_ner import Sentence

def precise_position(sentence : Sentence, entity : dict) -> tuple[int, int] | None:
    """Function that returns the most precise known true position of an entity in the original
    document. Prefers true_start/true_end, the annotation's own true position in the original
    document (set directly from the gold annotation - this is given-span entity linking, so this
    is exact, not searched for). Falls back to a flat per-sentence offset only for entities that
    don't originate from a gold annotation with a known true position.

    Parameters:
        sentence (Sentence):
            Sentence from which the entity was extracted.
        entity (dict):
            Dictionary representation of the entity, as returned by Sentence.entities_to_dicts().
    Returns:
        out: A tuple with the true (start, end) position of the entity, or None if unknown.
    """
    true_start = entity['other'].get('true_start')
    true_end = entity['other'].get('true_end')
    if true_start is not None and true_end is not None:
        return true_start, true_end
    if sentence.original_start is not None:
        return entity['start'] + sentence.original_start, entity['end'] + sentence.original_start
    return None

def postprocess_sentences(original_text : str, sentences : list[Sentence], remap_annotations_to_original : bool = True) -> list[dict]:
    """Function used to extract the entities from the Sentence list and fix the start and end
    indexes so that it can be compared to annotations in the original text.

    Parameters:
        original_text (str):
            Original text from which the sentences were extracted.
        sentences (list[Sentence]):
            List of Sentences with the entities found in each of them.
    Returns:
        out: A list of dictionary with the keys: start, end, and concept_id.
    """
    if remap_annotations_to_original:
        all_known = all(
            precise_position(sentence, entity) is not None
            for sentence in sentences
            for entity in sentence.entities_to_dicts()
        )

        if all_known:
            # Every entity's true position in the original text is known (computed while segmenting
            # the note), so positions can be used directly without the fuzzy whole-document
            # character remap, which is unsafe when the note has repeated phrasing.
            entities = []
            for sentence in sentences:
                for entity in sentence.entities_to_dicts():
                    true_start, true_end = precise_position(sentence, entity)
                    entities.append({'start' : true_start,
                                    'end' : true_end,
                                    'concept_id' : entity['label'],
                                    'options' : entity['options'],
                                    'confidence' : entity['confidence'],
                                    'other' : entity['other']
                                    })
            mapped_entities = entities
        else:
            # Fall back to the legacy whole-document character remap when some sentence's true
            # offset could not be determined (e.g. datasets that don't compute it).
            modified_text = ""
            offset = 0
            entities = []
            for sentence in sentences:
                # Add and offset the entities
                for entity in sentence.entities_to_dicts():
                    entities.append({'start' : entity['start'] + offset,
                                    'end' : entity['end'] + offset,
                                    'concept_id' : entity['label'],
                                    'options' : entity['options'],
                                    'confidence' : entity['confidence'],
                                    'other' : entity['other']
                                    })

                # Add the new sentence to the modified text and update the offset
                modified_text = modified_text + sentence.text
                offset = len(modified_text)

            # Remap the entities to the original text
            mapped_entities = remap_annotations(original_text, modified_text, entities)

            if len(entities) != len(mapped_entities):
                warnings.warn(f'There was some error in the remapping process and some entities were lost. Remapped entities ({len(mapped_entities)}) are less than the entities found ({len(entities)})', RuntimeWarning)
    else:
        entities = []
        for sentence in sentences:
            # Add and offset the entities        
            for entity in sentence.entities_to_dicts():
                entities.append({'start' : entity['start'],
                                'end' : entity['end'],
                                'concept_id' : entity['label'],
                                'options' : entity['options'],
                                'confidence' : entity['confidence'],
                                'other' : entity['other']
                                })
                
        mapped_entities = entities

    return mapped_entities

class SnomedPipe():
    """Class that represents our pipeline to extract entities and link them to concepts
    from a Knowledge Base.
    
    Attributes:
        entity_linker (EntityLinker):
            Object that allows to perform the entity linking step.
    """
    def __init__(self, entity_linker : EntityLinker):
        """Method that initializes the SNOMED pipeline for NER and EL.
        
        Parameters:
            entity_linker (EntityLinker):
                Object that allows to perform the entity linking step.
        """
        self.entity_linker = entity_linker
    
    def link_entities_given_spans(self, text : str, sentences : list[Sentence], remap_annotations_to_original : bool = True) -> list[dict]:
        """Method that performs the entity link for the sentences, given that those sentences
        have already the spans identified.

        Parameters:
            text (str):
                Text from which to link the entities.
            sentences (list[Sentence]):
                List of Sentence objects with Entities already identified, i.e, start and end of each entity
                has been assigned.
        
        Returns:
            out: A list of dictionaries that contain the following keys: start, end, concept_id.
        """
        entities = []
        for sentence in sentences:
            self.entity_linker.link_entities_from_sentence(sentence)

        entities = postprocess_sentences(text, sentences, remap_annotations_to_original)
      
        return entities

                

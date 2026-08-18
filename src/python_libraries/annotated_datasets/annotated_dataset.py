from abc import ABC, abstractmethod

from ..sentence_ner import Sentence

NOT_RELEVANT_CHARACTERS = [' ', '\n']

def remap_annotations(original_text : str, transformed_text : str, annotations : list[dict]) -> list[dict]:
    """Function that maps a list of annotations from the preprocessed text into the original text. The list of
    annotations it receives should have the following keys: start, end, concept_id.
    
    Parameters:
        original_text (str):
            Text from which the annotations were extracted.
        transformed_text (str):
            Processed text.
        annotations (list[dict]):
            List of dictionaries, where each dictionary has the keys: start, end, and concept_id.

    Returns:
        A list of dictionaries with the keys: start, end, and concept_id, but mapped to the original text
        indexes.
    """
    # To map the annotations to the original indexes
    transformed_to_original = {}

    # Auxiliary variables to help with some things from the preprocessing
    looking_for_dot = False
    aux_i = 0

    # Indexes for the original and transformed texts
    i = 0  # original index
    j = 0  # transformed index
    
    # Since the original text has characters that were removed we want to be able to map
    # the annotations in the preprocessed text to the original indexes
    while (i < len(original_text)):   
        char = original_text[i]

        if j < len(transformed_text):
            # Some breaks are transformed into spaces during the preprocessing
            if char == transformed_text[j] or (transformed_text[j] == ' ' and char == '\n'):
                looking_for_dot = False
                transformed_to_original[j] = i
                j += 1
            # The segmentation process might add dots that are not found in the original text
            # If we find one dot and we need to skip a relevant character in the original text,
            # this means that said dot is an artifical one and can be skipped
            elif transformed_text[j] == '.':
                looking_for_dot = True
                aux_i = i 
        
        # If we are going to skip a relevant character because of an artificial dot
        # Instead we skip the dot and return to the previous position in the original text
        if looking_for_dot and char not in NOT_RELEVANT_CHARACTERS:
            j += 1
            looking_for_dot = False
            i = aux_i
        else:
            i += 1

    # Map the annotations back to the original "space"
    mapped_annotations = []
    for annotation in annotations:
        start = annotation['start']
        end = annotation['end']

        original_start = transformed_to_original.get(start, None)
        original_end = transformed_to_original.get(end - 1, None)  # Adjust for inclusivity

        if original_start is not None and original_end is not None:
            mapped_annotations.append({'start' : original_start, 
                                       'end' : original_end + 1, 
                                       'concept_id' : annotation['concept_id'],
                                       'options' : annotation['options'],
                                       'confidence' : annotation['confidence'],
                                       'other' : annotation['other']
                                       })

    return mapped_annotations

class AnnotatedDataset(ABC):
    """Abstract class that represents the entity linking pipeline. Each subclass is expected to provide functionality to link the entities found in a Sentence to their corresponding concept."""

    @abstractmethod
    def get_note_text(self, note_id : str) -> str:
        """Method that returns the text of a note by their note_id.
        
        Parameters:
            note_id (str):
                String that represents the identifier of the note.
        
        Returns:
            A string that contains the text of the note.
        """
        pass

    @abstractmethod
    def get_note_annotations(self, note_id : str) -> list[dict]:
        """Method that returns the annotations of a note by their note_id. Each annotation is a dictionary with the keys: start, end, label, and span.
        
        Parameters:
            note_id (str):
                String that represents the identifier of the note.
        
        Returns:
            A list that contains the annotations for the note. If the note
            can not be found, an empty list is returned instead.
        """
        pass

    @abstractmethod
    def get_note_ids(self) -> list[str]:
        """Method that returns the available note_ids.
        
        Returns:
            A list that contains the available note_ids in the class.
        """
        pass
    
    @abstractmethod
    def get_annotated_sentences_from_note(self, note_id : str, transform : bool = True) -> list[dict]|list[Sentence]:
        """Method that returns the sentences and annotations for a given text from MIMIC.
        
        Parameters:
            note_id (str):
                String that represents the identifier of the note.
            transform (bool):
                Whether to transform the output into a list of Sentence.
        
        Returns:
            A list of dictionaries with keys: ['sentence', 'annotations']. If transform is set to True, 
            a list of Sentence objects is returned instead.
        """
        pass
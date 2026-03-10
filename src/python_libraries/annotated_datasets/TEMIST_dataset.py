import pandas as pd
import re
import warnings

from .annotated_dataset import AnnotatedDataset
from ..ner_model.sentence_ner import Sentence, Entity

# distemist  = filename mark    label	off0	    off1	    span	code	semantic_rel
# symptemist = filename	        label	span_ini	span_end	text	code	sem_rel	        is_abbrev	is_composite	need_context
# medprocner = filename	        label	start_span	end_span	text	code	sem_rel	        is_abbrev	is_composite	need_context

def extract_sentences_index(text : str) -> list[tuple[int, int, str]]:
    """Method that returns a list of tuples (start, end, sentence_text)."""
    # Pattern to segment sentences from TEMIST
    pattern = r'(?<!\b\w\.\w)(?<!\b\w\.)((?<=[.!?])\s+)(?=[A-ZÁÉÍÓÚÑ])'
    
    # Extract sentences
    sentences = []
    last_end = 0
    for match in re.finditer(pattern, text):
        s_end = match.start()
        sentence = text[last_end:s_end].strip()
        if sentence:
            sentences.append((last_end, s_end, sentence))
        last_end = match.end()

    if last_end < len(text):
        sentences.append((last_end, len(text), text[last_end:].strip()))

    return sentences

def find_sentence_for_annotation(sentences : list[dict], start : int) -> int:
    """Method that returns the index from the list that corresponds
    to the annotation which starts at start."""
    for i, sentence in enumerate(sentences):
            if sentence['start'] <= start < sentence['end']:
                return i
    warnings.warn(message=f"Could not find a sentence for annotation starting at {start}. Returning None")
    return None

class TEMIST_dataset(AnnotatedDataset):
    """Class that represents the annotated notes from DisTEMIST, MedProcNER, or SympTEMIST.
    
    Attributes:
        annotations (dict):
            Dictionary that stores, for each note_id, its corresponding annotations and text.
    """
    def __init__(self, notes_folder_path : str, annotations_tsv_path : str, dataset_type : str = "distemist", ignore_combined : bool = False, ignore_no_code : bool = True) -> None:
        """Loads the notes and annotations into the class.
        
        Parameters:
            notes_folder_path (str):
                Path to the folder where the notes are found.
            annotation_csv_path (str):
                Path to the annotation CSV file. Only the notes whose id is found
                in the CSV will be loaded into the class.
            dataset_type (str):
                Name of the dataset to be loaded: 'distemist', 'medprocner', or 'symptemist'.
            ignore_combined (bool):
                Whether to ignore combined annotations. Defaults to False.
            ignore_no_code (bool):
                Whether to ignore NO_CODE annotations. Defaults to True.
        """
        # Read the DF with the annotations
        annotations_df = pd.read_csv(annotations_tsv_path, sep="\t")

        if dataset_type == 'distemist':
            tsv_map = {'off0' : 'start',
                       'off1' : 'end',
                       'span' : 'text'}
        elif dataset_type == 'symptemist':
            tsv_map = {'span_ini' : 'start',
                       'span_end' : 'end'}
        elif dataset_type == 'medprocner':
            tsv_map = {'start_span' : 'start',
                       'end_span' : 'end'}
        else:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. "
                f"Expected one of ['distemist', 'symptemist', 'medprocner']."
            ) 
        annotations_df = annotations_df.rename(columns=tsv_map)

        # Create the dictionary that contains the text and annotations for each note_id
        self.annotations = {}

        # Extract the annotations
        current_note_id = None
        for _, row in annotations_df.iterrows():
            # Read current note_id
            note_id = row['filename']

            # If we change the note_id, we need to load the text
            # and create a new entry for it in the dictionary
            if current_note_id is None or note_id != current_note_id:
                # Change current note
                current_note_id = note_id

                # Load the note
                note_path = notes_folder_path + current_note_id + '.txt'
                with open(note_path, 'r', encoding='utf-8') as text_file:
                    text = text_file.read()

                    if current_note_id not in self.annotations:
                        self.annotations[current_note_id] = {'text' : text, 'annotations' : []}

            # Crete the annotation entry and add it to the dictionary
            codes = [row['code']]

            if ignore_no_code and row['code'] == 'NO_CODE':
                continue
            
            if '+' in row['code']:
                if ignore_combined:
                    continue
                else:
                    codes = row['code'].split('+')
            for code in codes:
                self.annotations[current_note_id]['annotations'].append({'start' : row['start'],
                                                                        'end' : row['end'],
                                                                        'label' : code,
                                                                        'span' : row['text']})

    def get_note_text(self, note_id : str) -> str:
        """Method that returns the text of a note by their note_id.
        
        Parameters:
            note_id (str):
                String that represents the identifier of the note.
        
        Returns:
            A string that contains the text of the note. If the note
            can not be found, an empty string is returned instead.
        """
        if note_id in self.annotations:
            return self.annotations[note_id]['text']
        raise ValueError(f"Invalid note_id '{note_id}'.")

    def get_note_annotations(self, note_id : str) -> list[dict]:
        """Method that returns the annotations of a note by their note_id. Each annotation is a dictionary with the keys: start, end, label, and span.
        
        Parameters:
            note_id (str):
                String that represents the identifier of the note.
        
        Returns:
            A list that contains the annotations for the note. If the note
            can not be found, an empty list is returned instead.
        """
        if note_id in self.annotations:
            return self.annotations[note_id]['annotations']
        raise ValueError(f"Invalid note_id '{note_id}'.")

    def get_note_ids(self) -> list[str]:
        """Method that returns the available note_ids.
        
        Returns:
            A list that contains the available note_ids in the class.
        """
        return list(self.annotations.keys())
    
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
        # Obtain the text
        text = self.get_note_text(note_id=note_id)

        # Segment into sentences
        sentences_raw = [{'start' : start, 'end' : end, 'sentence' : sentence, 'annotations' : []} 
                         for start, end, sentence in extract_sentences_index(text=text)]

        # Obtain the annotations
        annotations = self.get_note_annotations(note_id=note_id)

        # Add annotations to sentences
        for ann in annotations:
            sentence_i = find_sentence_for_annotation(sentences=sentences_raw, start=ann['start'])

            if sentence_i is not None:
                sentences_raw[sentence_i]['annotations'].append(ann)

        # Transform into a list of Sentence objects if flag set to True
        if transform:
            map_dictionary = {'text' : 'span', 'start' : 'start',
                              'end' : 'end',   'label' : 'label'}
        
            sentences_transformed = []
            for sentence in sentences_raw:
                entities = [Entity.from_dictionary(annotation, map_dictionary) for annotation in sentence['annotations']]
                sentence = Sentence(text=sentence['sentence'], section="", entities=entities)
                sentences_transformed.append(sentence)
            return sentences_transformed
        return sentences_raw
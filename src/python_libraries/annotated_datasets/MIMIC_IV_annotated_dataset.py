import pandas as pd
import re

from .annotated_dataset import AnnotatedDataset
from ..sentence_ner import Sentence, Entity

SECTIONS = ["Name", "Unit No", "Admission Date", "Discharge Date", "Date of Birth", "Sex", "Service", 
            "Allergies", "Attending", "Chief Complaint", "Major Surgical or Invasive Procedure", "History of Present Illness",
            "Past Medical History", "Social History", "Family History", "Physical Exam", "Pertinent Results",
            "Brief Hospital Course", "Medications on Admission", "Discharge Medications", "Discharge Disposition",
            "Discharge Diagnosis", "Discharge Condition", "Discharge Instructions", "Followup Instructions"]

NOT_RELEVANT_CHARACTERS = [' ', '\n']

# Functions for extracting the sections
def find_annotations_per_section(text_sections : dict[str], annotations : list[dict]) -> dict[str]:
    """Function that given a dictionary composed of text sections and a list of annotations,
    returns to which sections of the text corresponds each annotation. This method assumes that
    the annotations are ordered by order of aparition in the text.
    
    Parameters:
        text_sections (dict):
            Dictionary that stores text sections. Each key corresponds to a section's name. Each value
            is a dictionary with at least the keys: start, end, span_start, and text; where start and end 
            denote the limits of the section in text, span_start takes into account the heading; and text
            is the corresponding text for the section.

        annotations (list):
            List of annotations. Each annotation is a dictionary with the keys: start, end, label, and span.

    Returns:
        A modified version of text_sections, where a new key has been assigned called 'annotations', which
        stores the list of annotations that appear in each section.
    """
    annotated_text_sections = {}

    current_section_index = 0
    current_section = SECTIONS[current_section_index]

    annotated_text_sections[current_section] = text_sections[current_section]
    annotated_text_sections[current_section]['annotations'] = []

    for annotation in annotations:
        while(text_sections[current_section]['end'] < annotation['end']):
            # Advance to the next section
            current_section_index += 1
            current_section = SECTIONS[current_section_index]

            while(current_section not in text_sections):
                current_section_index += 1
                current_section = SECTIONS[current_section_index]

            annotated_text_sections[current_section] = text_sections[current_section]
            annotated_text_sections[current_section]['annotations'] = []
        
        if annotation['start'] >= text_sections[current_section]['span_start'] and annotation['end'] <= text_sections[current_section]['end']:
            annotated_text_sections[current_section]['annotations'].append(annotation)
        else:
            print('Error aligning annotations')

    return annotated_text_sections

def find_anonymized_heading(text : str, section : str) -> str|None:
    """Function that returns the anonymized version of certain heading's names.
    
    Parameters:
        text (str):
            Text of the clinical note.
        section (str):
            Name of the section whose heading we are interested in.
    
    Returns:
        A string with the corresponding anonymized version of the heading found
        in the text. Otherwise returns None.
    """
    # We only call this method if there has been any type of deanonymization of
    # the heading, and we might be interested in getting back the anonymized version
    if section == "Chief Complaint":
        match_obj = re.search('___ Complaint:', text)
        if match_obj is None:
            return 'Chief ___'
        return '___ Complaint'
    elif section == "Physical Exam":
        match_obj = re.search('___ Exam:', text)
        if match_obj is None:
            return 'Physical ___'
        return '___ Exam'
    elif section == "Medications on Admission":
        return '___ on Admission'
    return None

def get_deanonymized_section_name(section : str):
    """Method to get the deanonymized version of the section's name.
    
    Parameters:
        section (str):
            Name of the section.
    
    Returns:
        The deanonymized version of the section's name or the section's name if it is not anonymized.
    """
    if section in ['___ Complaint', 'Chief ___']:
        return "Chief Complaint"
    elif section in ['___ Exam', 'Physical ___']:
        return "Physical Exam"
    elif section in ['___ on Admission']:
        return "Medications on Admission"
    else:
        return section

def extract_sections(text : str) -> dict[str]:
    """Function that extracts the sections from the text of a MIMIC note. It uses as
    reference the sections defined in SECTIONS. It assumes that those sections are
    ordered by order of apparition in the text.

    Parameters:
        text (str):
            Text of a clinical note from MIMIC.
    
    Returns:
        dict: A dictionary that, for each section, stores the following keys: start, end, span_start,
        and text; where start and end represent the limits of the section without the heading, 
        span_start takes into account the heading, and text is the content of the section.
    """
    last_section = 0
    section_dict = {}
    for i, section in enumerate(SECTIONS):
        match_obj = re.search(section+':', text)
        anonymized_heading = False

        # Certain notes' headings are anonymized for whatever reason
        if match_obj is None:
            if section == "Chief Complaint":
                match_obj = re.search('___ Complaint:|Chief ___:', text)
                anonymized_heading = True
            elif section == "Physical Exam":
                match_obj = re.search('___ Exam:|Physical ___:', text)
                anonymized_heading = True
            elif section == "Medications on Admission":
                match_obj = re.search('___ on Admission:', text)
                anonymized_heading = True
            

        # We might find some texts where one of the sections is missing
        if match_obj is not None:
            start, end = match_obj.span()

            # If it is not the first section, we update last's section end
            # Last section will either be the directly previous one or the last
            # one we could find
            if i > 0:
                section_dict[SECTIONS[last_section]]['end'] = start
                section_dict[SECTIONS[last_section]]['text'] = text[section_dict[SECTIONS[last_section]]['start']:section_dict[SECTIONS[last_section]]['end']]
                last_section = i

            # Span start takes into account the section name
            if not anonymized_heading: 
                section_dict[section] = {'span_start' : end - len(section) - 1, 
                                          'start' : end, 'end' : -1, 
                                          'text' : '',
                                          'anonymized' : False}
            else:
                anonymized_heading_text = find_anonymized_heading(text, section)
                section_dict[section] = {'span_start' : end - len(anonymized_heading_text) - 1, 
                                          'start' : end, 'end' : -1, 
                                          'text' : '',
                                          'anonymized' : True}

            if i == len(SECTIONS) - 1:
                section_dict[section]['end'] = len(text)
                section_dict[section]['text'] = text[section_dict[section]['start']:section_dict[section]['end']]
                
        # If the section we couldn't find is the last one, we will have to update
        # the last section's end
        elif i == len(SECTIONS) - 1:
            section_dict[SECTIONS[last_section]]['end'] = len(text)
            section_dict[SECTIONS[last_section]]['text'] = text[section_dict[SECTIONS[last_section]]['start']:section_dict[SECTIONS[last_section]]['end']]
    
    return section_dict

# Functions for segmenting the text into sentences
def segment_text_regex_dot(text : str) -> list[str]:
    """Function that splits the text into sentences by considering a dot
    followed by a white space as a separator. Exceptions to these are Dr.,
    Mr., or numbered sections.
    
    Parameters:
        text (str):
            Text to be segmented into sentences.
    
    Returns:
        A list of strings.
    """
    regex_sep_dot = r'(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\d)\.\s'
    
    return re.split(regex_sep_dot, text)

def segment_text_n_line(text : str) -> list[str]:
    """Function that splits the text into sentences by considering a numbered
    line as a separator. A numbered line is defined as a break followed by a
    digit and a dot.
    
    Parameters:
        text (str):
            Text to be segmented into sentences.
    
    Returns:
        A list of strings.
    """
    regex_sep_n_line = r'\n\d+\.'
    
    return re.split(regex_sep_n_line, text)

def segment_text_header(text : str, old_version : bool = False) -> list[str]:
    """Function that splits the text into sentences by headings as a separator. A
    heading is one or more words followed by :. Headings are also returned along with the rest
    of the sentences.
    
    Parameters:
        text (str):
            Text to be segmented into sentences.
    
    Returns:
        A list of strings.
    """
    if old_version:
        regex_sep_header = r'(\S+:)'
    else:
        regex_sep_header = r'([^\n\t]+:)' #r'(\S+:)'

    return re.split(regex_sep_header, text)

def segment_text_break(text : str) -> list[str]:
    """Function that splits the text into sentences depending on the type
    of breaks found. If the text contains a list denoted by break and hyphen,
    it uses that as a separator. If the list is denoted by a break and a mayus,
    it uses that instead. Otherwise it uses just a break as a separator. 
    
    Parameters:
        text (str):
            Text to be segmented into sentences.
    
    Returns:
        A list of strings.
    """
    regex_sep_break = ''
    if re.search(r'\n-', text) is not None:
        regex_sep_break = r'\n(?=-)'
    elif re.search(r'\n[A-Z]', text) is not None:
        regex_sep_break = r'\n(?=[A-Z])'
    else:
        regex_sep_break = r'\n'
    return re.split(regex_sep_break, text) 

def segment_text_simple_break(text : str) -> list[str]:
    """Function that splits the text into sentences depending on the type
    of breaks found. Similar to segment_text_break, but it does not take
    into account lists denoted by mayus.
    
    Parameters:
        text (str):
            Text to be segmented into sentences.
    
    Returns:
        A list of strings.
    """
    regex_sep_break = ''
    if re.search('\n-', text) is not None:
        regex_sep_break = r'\n(?=-)'
    else:
        regex_sep_break = r'\n'
    return re.split(regex_sep_break, text)

def segment_pertinent_results(text : str) -> list[str]:
    """Function that splits the text for the section Pertinent Results. Initially it
    segments it by considering double breaks. Then, each subsegment is split either
    by using segment_text_regex_dot or by looking at anonymized time denoted by ___
    and a time. 
    
    Parameters:
        text (str):
            Text to be segmented into sentences.
    
    Returns:
        A list of strings.
    """
    initial_segment = text.split('\n\n')

    results = []
    for segment in initial_segment:
        if re.search(r'\n___ \d+:', segment) is not None:
            results += re.split(r'\n___ (?=\d+:)', segment)
        else:
            segments = segment_text_regex_dot(segment)    
            for seg in segments:
                seg_strip = seg.strip()
                if len(seg_strip) > 0:
                    if len(segments) > 1:
                        results.append(seg_strip + '.')
                    else:
                        results.append(seg_strip)

    return results

def segment_text_into_sentences(section : str, text : str, all_text : str, old_version : bool = False, transform : bool = False, anonymized : bool = False) -> list[str]|list[Sentence]:
    """Function that splits the text into sentences depending on the section.
    It uses the previously defined segment methods. It also preprocesses the text
    by removing breaks. 
    
    Parameters:
        section (str):
            Section the text belonged to.
        text (str):
            Text to be split into sentences.
        transform (bool):
            Whether to return the sentences as a Sentence class rather than str.
    Returns:
        A list of strings or Sentence objects that contains the sentences.
    """
    sentences = []

    # Segment by regex expression with dot
    if section in ['History of Present Illness', 'Family History', 'Brief Hospital Course', 'Discharge Instructions']:
        sentences = segment_text_regex_dot(text)

        # Add the dot to each sentence
        sentences_aux = []
        for sentence in sentences:
            sentence_strip = sentence.strip()
            if len(sentence_strip) > 0:
                sentences_aux.append(sentence_strip + '.')
        sentences = sentences_aux
    # Segment by numbered lines
    elif section in ['Medications on Admission', 'Discharge Medications']:
        sentences = segment_text_n_line(text)
    # Segment by header and :
    elif section in ['Physical Exam']:
        sentences = segment_text_header(text, old_version)

        # Merge header with the rest of the text
        if len(sentences) > 1:
            sentences_aux = []
            start_index = 0

            # If odd number of sentences, we ignore the first element
            if len(sentences) % 2 == 1:
                sentences_aux.append(sentences[0])
                start_index += 1

            for s1, s2 in zip(sentences[start_index::2], sentences[start_index+1::2]):
                sentences_aux.append(s1 + s2)
            
            sentences = sentences_aux
    # Segment by line break without spaces
    elif section in ['Discharge Diagnosis', 'Discharge Condition']:
        sentences = segment_text_break(text)
    # Segment by simple line break
    elif section in ['Past Medical History']:
        sentences = segment_text_simple_break(text)
    # Special multiple step segmentation
    elif section in ['Pertinent Results']:
        sentences = segment_pertinent_results(text)
    # No segmentation needed
    else:
        sentences = [text]

    # Postprocess each sentence
    sentences_processed = []

    # Add the section name
    if anonymized:
        anonymized_heading = find_anonymized_heading(all_text, section)
        section = anonymized_heading if anonymized_heading is not None else section

    if transform:
        sentences_processed.append(Sentence(text=section + ':', section=section))
    else:
        sentences_processed.append(section + ':')

    for sentence in sentences:
        # Preprocess sentence
        sentence = preprocess_sentence(sentence)

        if len(sentence) > 0:
            if transform:
                sentences_processed.append(Sentence(text=sentence, section=section))
            else:
                sentences_processed.append(sentence)

    return sentences_processed

def preprocess_sentence(sentence : str) -> str:
    """Simple preprocessing for a sentence. It transforms breaks
    to spaces, removes multiple spaces, as well as initial or ending ones.
    
    Parameters:
        sentence (str):
            String that contains a sentence.

    Returns:
        A string that contains the preprocessed sentence.
    """
    # Remove breaks
    sentence = sentence.replace('\n', ' ')

    # Remove multiple spaces
    sentence = re.sub(' +', ' ', sentence)

    # Remove initial and ending spaces
    sentence = sentence.strip()

    return sentence

# Functions for fixing annotation indexes
def fix_annotations_per_sentence(sentences : list[str], annotations : list[dict], transform : bool = False, section : str = None, adapt_indexes : bool = True) -> list[dict]|list[Sentence]:
    """Function that links to which sentence corresponds each annotation. It assumes
    that annotations are ordered by order of apparition in text. 
    
    Parameters:
        sentences (list):
            List of sentences represented by strings.
        annotations (list):
            List of annotations, where each annotation is represented by a dictionary with
            the keys: start, end, span, and label.
        transform (bool):
            Whether to transform the output into a list of Sentence.
        section (str):
            String that denotes the section of the sentences. 
        adapt_indexes (bool):
            Whether to adapt the start and end index of each annotation to each sentence or maintain the original ones, which are relative to the whole text.
            
    Returns:
        list: A list of dictionaries, where each dictionary has two keys: sentence, and annotations. If transform is set to True, a list of Sentence
        objects is returned instead.
    """
    sentences_annotated = []

    current = {'sentence' : sentences[0], 'annotations' : []}
    sent_index = 0
    char_index = 0

    for annotation in annotations:
        annotation_fix = preprocess_sentence(annotation['span'])

        # Iterate until we find the sentence for the current annotation
        start = -1
        while(start == -1):
            # Try to find if the annotation is in current sentence
            start = current['sentence'][char_index:].find(annotation_fix)

            # If it is not in the sentence it might be because:
            # a) It is actually not in the sentence
            # b) The annotation is split between multiple sentences
            if start == -1:
                # This is the other case
                if ' ' in annotation_fix:
                    # If the annotation might be split, we check if at least
                    # the initial part is in current sentence
                    annotation_split = annotation_fix.split(' ')
                    
                    # Particular case of a dot as part of the first split
                    if '.' in annotation_split[0]:
                        #print('Done for', annotation_fix)
                        aux = []
                        aux.append(annotation_split[0][:annotation_split[0].find('.')])
                        aux.append('.')
                        aux += annotation_split[1:]

                        annotation_split = aux
                    
                    if current['sentence'][char_index:].find(annotation_split[0]) != -1:

                        # If the initial part of the annotation is in the split
                        # we know that the actual sentence should be split at most in
                        # a number of splits equal to the pieces of the annotation
                        # i.e, if an annotation is split into two parts, it can be at
                        # most divided between two sentences
                        i = 0
                        search_sentence = current['sentence']
                        while(start == -1 and i < len(annotation_split) - 1 and (sent_index + i + 1) <= len(sentences)):
                            # Increment the sentence number
                            i += 1

                            # Add the text of the next sentence
                            search_sentence += ' ' + sentences[sent_index + i]

                            start = search_sentence[char_index:].find(annotation_fix)

                        # If we found the group of sentences we need to update the 
                        # text of the current sentence and increment the index of the sentences
                        if start != -1:
                            #print('MERGE!')
                            current['sentence'] = search_sentence
                            sent_index += i
                
                # At this point, if we have not yet found it
                # we need to change the sentence
                if start == -1:
                    sentences_annotated.append(current)

                    # As long as there are annotations, there should be 
                    # sentences
                    sent_index += 1
                    char_index = 0
                    current = {'sentence' : sentences[sent_index], 'annotations' : []}
        

        # We have found the correct sentence
        end = start + char_index + len(annotation_fix)
        current['annotations'].append({'start' : start + char_index if adapt_indexes else annotation['start'],
                                       'end' : end if adapt_indexes else annotation['end'],
                                       'span' : annotation_fix,
                                       'label' : annotation['label']})
        char_index = end

    # Add the current sentence
    sentences_annotated.append(current)
    sent_index += 1

    # Add the remaining sentences, that do not have annotations
    # (if there are any)
    while(sent_index < len(sentences)):
        sentences_annotated.append({'sentence' : sentences[sent_index], 
                                    'annotations' : []})
        sent_index += 1

    # If transform is set to True, we transform the sentences to the Sentence class
    if transform:
        map_dictionary = {'text' : 'span',
                          'start' : 'start',
                          'end' : 'end',
                          'label' : 'label'}
        
        sentences_transformed = []
        for sentence in sentences_annotated:
            entities = [Entity.from_dictionary(annotation, map_dictionary) for annotation in sentence['annotations']]
            sentence = Sentence(sentence['sentence'], section, entities)
            sentences_transformed.append(sentence)
        
        return sentences_transformed
    return sentences_annotated

class MIMIC_IV_dataset(AnnotatedDataset):
    """Class that represents the annotated notes from MIMIC IV. More specifically, this class has
    been prepared for the notes annotated for the SNOMED CT Entity Linking Challenge 
    (https://www.drivendata.org/competitions/258/competition-snomed-ct/).
    
    Attributes:
        annotations (dict):
            Dictionary that stores, for each note_id, its corresponding annotations and text.
    """
    def __init__(self, notes_folder_path : str, notes_csv_path : str, annotation_csv_path : str) -> None:
        """Loads the notes and annotations into the class. It needs the path to the CSV with the annotations,
        and either the path to the folder with the notes or the path to a CSV that contains the notes.
        
        Parameters:
            notes_folder_path (str):
                Path to the folder where the notes are found.
            notes_csv_path (str):
                Path to the CSV file containing the notes.
            annotation_csv_path (str):
                Path to the annotation CSV file. Only the notes whose ids are found
                in the CSV will be loaded into the class.
        """
        # Read the DF with the annotations
        annotations_df = pd.read_csv(annotation_csv_path)

        # Create the dictionary that contains the text and annotations for each note_id
        self.annotations = {}

        # Extract the annotations
        current_note_id = None
        for _, row in annotations_df.iterrows():
            # Read current note_id
            note_id = row['note_id']

            # If we change the note_id, we need to load the new note
            # and create a new entry for it in the dictionary
            if current_note_id is None or note_id != current_note_id:
                # Change current note
                current_note_id = note_id

                # Load the note
                if notes_csv_path is not None:
                    notes_df = pd.read_csv(notes_csv_path)
                    note_row = notes_df[notes_df['note_id'] == current_note_id]
                    text = note_row['text'].iloc[0]
                else:
                    note_path = notes_folder_path + current_note_id + '.txt'
                    with open(note_path, 'r', encoding='utf-8') as text_file:
                        text = text_file.read()

                self.annotations[current_note_id] = {'text' : text, 'annotations' : []}

            # Crete the annotation entry and add it to the dictionary
            self.annotations[current_note_id]['annotations'].append({'start' : row['start'],
                                                                     'end' : row['end'],
                                                                     'label' : str(row['concept_id']),
                                                                     'span' : text[row['start']:row['end']]})
            
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
        return ''

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
        return []

    def get_note_ids(self) -> list[str]:
        """Method that returns the available note_ids.
        
        Returns:
            A list that contains the available note_ids in the class.
        """
        return list(self.annotations.keys())
    
    def get_annotated_sentences_from_note(self, note_id : str, transform : bool = False, adapt_annotation_index : bool = True) -> list[dict]|list[Sentence]:
        """Method that returns the sentences and annotations for a given text from MIMIC.
        
        Parameters:
            note_id (str):
                String that represents the identifier of the note.
            transform (bool):
                Whether to transform the output into a list of Sentence.
            adapt_annotation_index (bool):
                Whether to adapt the start and end index of each annotation to each sentence or maintain the original ones, which are relative to the whole text.
        
        Returns:
            list: A list of dictionaries, where each dictionary has the following keys: sentence,
            and annotations. If transform is set to True, a list of Sentence objects is returned instead.
        """
        # Obtain current text
        text = self.get_note_text(note_id)

        # Obtain the annotations
        annotations = self.get_note_annotations(note_id)

        # Segment the text into sections
        text_sections = extract_sections(text)

        # Find the annotations for each text_section
        annotated_text_sections = find_annotations_per_section(text_sections, annotations)

        # Obtain the sentences per section
        sentences_annotations = []
        for section in annotated_text_sections.keys():
            anonymized = text_sections[section]['anonymized']

            sentences = segment_text_into_sentences(section, annotated_text_sections[section]['text'], all_text=text, anonymized=anonymized)
            
            sentences_annotations += fix_annotations_per_sentence(sentences, annotated_text_sections[section]['annotations'], transform=transform, section=section, adapt_indexes=adapt_annotation_index)

        return sentences_annotations
            
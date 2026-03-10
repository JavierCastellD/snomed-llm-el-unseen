import configparser
import json
import os
import sys
from tqdm import tqdm

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder

from python_libraries.embedding_models.embedding_model import load_embeddings
from python_libraries.embedding_models.sentencetransformer_EM import SentenceTransformerEM
from python_libraries.entity_linker.entity_linker_llm_dict import EntityLinkerLLMDictionary
from python_libraries.annotated_datasets.MIMIC_IV_annotated_dataset import MIMIC_IV_dataset
from python_libraries.llm_queries.LLM_query_helper_openai import LLMQueryHelperOpenAI
from python_libraries.reranker import Reranker
from python_libraries.snomed.snomed import Snomed
from python_libraries.snomed.snomed_embedder import SnomedEmbedder
from python_libraries.snomed.snomed_pipe import SnomedPipe
from python_libraries.utils import load_config, annotations_to_df, concatenate_annotations

# RUN CONFIGURATION
config_run_file = sys.argv[1]
model_name = sys.argv[2]
triplet_type = sys.argv[3]

config_dic = load_config(config_run_file)

EXECUTION_NAME = model_name + '_' + triplet_type + '_' + config_dic['execution_name']
span_dictionary_path = config_dic['span_dictionary_path']

disambiguate_abbreviations = config_dic['disambiguate_abbreviations']
llm_for_el = config_dic['llm_for_el']
rephrase = config_dic['rephrase']
replace_span = config_dic['replace_span']
use_fsn = config_dic['use_fsn']
number_of_options = config_dic['number_of_options']
rerank_top_n = config_dic['rerank_top_n']
trust_training = config_dic['trust_training']
use_reranker = config_dic['use_reranker']

threshold_for_dictionary = config_dic['threshold_for_dictionary']
threshold = config_dic['threshold']

# CONFIGURATION FOR DICT OPTIONS
dictionary_options = config_dic['dictionary_options']

# Files for the checkpoints
DIRECTORY_PATH = 'el_checkpoints/' + EXECUTION_NAME + '_checkpoints'

if not os.path.exists(DIRECTORY_PATH):
    os.makedirs(DIRECTORY_PATH)

# LLM files
CONFIG_FILE = "config.cfg"

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

#ENDPOINT = config['AZURE']['endpoint']
ENDPOINT = config['AZURE']['endpoint_openai']
API_KEY = config['AZURE']['apikey']

# SNOMED CT files
SNOMED_VERSION = "20230531"
CONCEPTS_PATH = f"snomed_data/conceptInternational_{SNOMED_VERSION}.txt" 
RELATIONS_PATH = f"snomed_data/relationshipInternational_{SNOMED_VERSION}.txt"
DESCRIPTIONS_PATH = f"snomed_data/descriptionInternational_{SNOMED_VERSION}.txt" 
 
# Embedding files
EMBEDDING_MODEL_PATH = f'sentence_bert_models/{model_name}_umls_{triplet_type}_en/'
EMBEDDING_DICTIONARY_PATH = f'snomed_dictionaries/desc_ent_all_{model_name}_umls_{triplet_type}_en_sct_dict.npz' 
#EMBEDDING_MODEL_PATH = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token" 
#EMBEDDING_DICTIONARY_PATH = "snomed_dictionaries/desc_ent_all_sapbert_mean_base_sct_dict.npz" #"snomed_dictionaries/desc_ent_all_sapbert_mean_base_sct_dict.json" #"snomed_dictionaries/desc_ent_all_sapbert_mean_base_rephrasings_sct_dict.json" 
DICTIONARY_DESCRIPTIONS = True

# Load SNOMED
snomed = Snomed(CONCEPTS_PATH, RELATIONS_PATH, DESCRIPTIONS_PATH)

# Load the embedding model
embedding_model = SentenceTransformerEM(EMBEDDING_MODEL_PATH)

# Load the dictionary
embedding_dictionary = load_embeddings(EMBEDDING_DICTIONARY_PATH)

# Create dictionary for SESemantic
types_hierarchy = {}
types_hierarchy['Clinical finding'] = {}
types_hierarchy['Procedure'] = {}
types_hierarchy['Body structure'] = {}

for cid_key in embedding_dictionary.keys():
    cid = int(cid_key.split('_')[0])
    top_level = snomed.get_top_level_concept(cid)
    top_level_fsn = snomed.get_fsn(top_level)

    if top_level_fsn in types_hierarchy:
        types_hierarchy[top_level_fsn][cid_key] = embedding_dictionary[cid_key]

# Load the id2name
with open(f"snomed_dictionaries/id2name_desc_ent_{model_name}_umls_{triplet_type}_en_sct_dict.json", "r") as id2name_file:
    id2name = json.load(id2name_file)
    
# Load SnomedEmbedder
snomed_embedder = SnomedEmbedder(snomed=snomed, embedding_model=embedding_model, id2name=id2name, embedding_types_dictionary=types_hierarchy, 
                                 dictionary_descriptions=DICTIONARY_DESCRIPTIONS)

# Load the LLM query helper
llm_query_helper = LLMQueryHelperOpenAI(API_KEY, ENDPOINT, model_name='gpt-5-mini', temperature=1)

# Load the CrossEncoder
#cross_encoder = CrossEncoder("cross-encoder/ce_abv_50_all_train")
cross_encoder = CrossEncoder(f"cross-encoder/cef_{model_name}_umls_{triplet_type}_en_snomed_sim_cand_200_epoch_1_bs_128")

# Create the Reranker
reranker = Reranker(cross_encoder)

# Load the span dictionary
with open(span_dictionary_path, "r") as dict_file:
    span_dictionary = json.load(dict_file)

ner_type2hierarchy = {'Body structure' : 'Body structure', 
                      'Clinical finding' : 'Clinical finding', 
                      'Procedure' : 'Procedure'}

# Load the text files
NOTES_FOLDER_PATH = 'mimic_data/mimic_notes_test/'
TRAIN_NOTES_PATH = 'mimic_notes_split/train_note_ids.txt'
TEST_NOTES_PATH = 'mimic_notes_split/test_note_ids.txt'
ANNOTATIONS_CSV_PATH = 'mimic_data/test_annotations.csv'
ANNOTATIONS_TRAIN_CSV_PATH = 'mimic_data/train_annotations.csv'

# Load the notes and annotations
mimic = MIMIC_IV_dataset(NOTES_FOLDER_PATH, ANNOTATIONS_CSV_PATH)

# Load the IDs of train notes
with open(TRAIN_NOTES_PATH, 'r') as train_file:
    train_notes_ids = [note.strip('\n') for note in train_file.readlines()]

# Load training concepts
anns_train = pd.read_csv(ANNOTATIONS_TRAIN_CSV_PATH)

training_concepts = list(anns_train['concept_id'].unique())

# Load the Entity Linker
entity_linker = EntityLinkerLLMDictionary(snomed=snomed, snomed_embedder=snomed_embedder, llm_query=llm_query_helper, reranker=reranker, span_dictionary=span_dictionary,
                                          dictionary_options=dictionary_options, disambiguate_abbreviations=disambiguate_abbreviations, llm_for_el=llm_for_el, 
                                          rephrase=rephrase, replace_span=replace_span, use_fsn=use_fsn, number_of_options = number_of_options, rerank_top_n = rerank_top_n, 
                                          trust_training=trust_training, use_reranker=use_reranker, threshold_for_dictionary=threshold_for_dictionary, threshold=threshold,
                                          ner_type2hierarchy=ner_type2hierarchy, training_concepts=training_concepts)

# Load the Snomed Pipe
snomed_pipe = SnomedPipe(entity_linker)

# Load the IDs of train notes
test_df = pd.read_csv('mimic_data/mimic-iv_notes_test_set.csv')

# Iterate through texts
for note_id in tqdm(test_df['note_id']):
    print(f'Current note_id: {note_id}')

    # Obtain the text
    text = mimic.get_note_text(note_id)
    
    # Obtain the annotated entities
    annotations = mimic.get_note_annotations(note_id)
    
    sentences = mimic.get_annotated_sentences_from_note(note_id=note_id, transform=True, adapt_annotation_index=True)

    # Obtain the predicted entities
    if os.path.exists(DIRECTORY_PATH+'/'+EXECUTION_NAME + '_' + note_id + '.csv'):
        predicted_entities = pd.read_csv(DIRECTORY_PATH+'/'+EXECUTION_NAME + '_' + note_id + '.csv')
        
    else:
        for sentence in sentences:
            for entity in sentence.entities:
                entity.ner_type = snomed.get_fsn(snomed.get_top_level_concept(int(entity.label)))
                entity.label = None
                    
        predicted_entities = snomed_pipe.link_entities_given_spans(text, sentences)
        df = annotations_to_df(note_id, predicted_entities, {'label' : 'concept_id', 'start' : 'start', 'end' : 'end'},  add_options=True, add_confidence=True, add_other=True)
        df.to_csv(DIRECTORY_PATH+'/'+EXECUTION_NAME + '_' + note_id + '.csv')

# Save the predictions to a single csv
concatenated_df = concatenate_annotations(folder_path=DIRECTORY_PATH, file_base_name=EXECUTION_NAME)

# Fix wrong start, end - do not know why
diff_mask_start = (concatenated_df[['start']] != test_df[['start']]).any(axis=1)
concatenated_df.loc[diff_mask_start, 'start'] = test_df[diff_mask_start]['start']
diff_mask_end = (concatenated_df[['end']] != test_df[['end']]).any(axis=1)
concatenated_df.loc[diff_mask_end, 'end'] = test_df[diff_mask_end]['end']

concatenated_df.to_csv(EXECUTION_NAME + '_predictions.csv', index=False)
    
llm_query_helper.save_cache()
import json
import os
import sys
from tqdm import tqdm

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder

from python_libraries.embedding_models.embedding_model import load_embeddings
from python_libraries.embedding_models.sentencetransformer_EM import SentenceTransformerEM
from python_libraries.entity_linker import EntityLinkerAdaptiveLLM
from python_libraries.annotated_datasets.MIMIC_IV_annotated_dataset import MIMIC_IV_dataset
from python_libraries.llm_queries import LLMQueryHelperOpenAI, OllamaQueryHelper
from python_libraries.reranker import Reranker
from python_libraries.snomed import Snomed, SnomedEmbedder, SnomedPipe
from python_libraries.utils import load_config, annotations_to_df, concatenate_annotations

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# RUN CONFIGURATION
config_run_file = sys.argv[1]
model_name = sys.argv[2]
triplet_type = sys.argv[3]

config_dic, config = load_config(config_run_file)

disambiguate_abbreviations = config_dic['disambiguate_abbreviations']
rephrase = config_dic['rephrase']
replace_span = config_dic['replace_span']
use_fsn = config_dic['use_fsn']
number_of_options = config_dic['number_of_options']
rerank_top_n = config_dic['rerank_top_n']
use_reranker = config_dic['use_reranker']

# CONFIGURATION FOR DICT OPTIONS
dictionary_options = config_dic['dictionary_options']

# LLM settings (read from the same ConfigParser object returned by load_config)

LLM_BACKEND    = os.environ.get('LLM_BACKEND',     config['LLM']['backend'])
LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', config['LLM']['model_name'])
LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', config['LLM']['temperature']))

_llm_name = LLM_MODEL_NAME.replace(':', '-')
_parts = ['snomed', model_name, triplet_type]
if use_reranker:
    _parts.append('rer')
if disambiguate_abbreviations:
    _parts.append('abv')
_parts += [str(rerank_top_n), str(number_of_options), _llm_name]
EXECUTION_NAME = '_'.join(_parts)

# Files for the checkpoints
CHECKPOINTS_FOLDER = config_dic['checkpoints_folder']
DIRECTORY_PATH = os.path.join(BASE_DIR, CHECKPOINTS_FOLDER, EXECUTION_NAME + '_checkpoints')

if not os.path.exists(DIRECTORY_PATH):
    os.makedirs(DIRECTORY_PATH)

# SNOMED CT files
SNOMED_VERSION = "20230531"
CONCEPTS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'conceptInternational_{SNOMED_VERSION}.txt')
RELATIONS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'relationshipInternational_{SNOMED_VERSION}.txt')
DESCRIPTIONS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'descriptionInternational_{SNOMED_VERSION}.txt')

# Embedding files
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, 'sentence_bert_models', f'{model_name}_{triplet_type}_en/')
EMBEDDING_DICTIONARY_PATH = os.path.join(BASE_DIR, 'snomed_dictionaries', f'desc_ent_all_{model_name}_{triplet_type}_en_sct_dict.npz')
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
with open(os.path.join(BASE_DIR, 'snomed_dictionaries', f'id2name_desc_ent_{model_name}_{triplet_type}_en_sct_dict.json'), "r") as id2name_file:
    id2name = json.load(id2name_file)
    
# Load SnomedEmbedder
snomed_embedder = SnomedEmbedder(snomed=snomed, embedding_model=embedding_model, id2name=id2name, embedding_types_dictionary=types_hierarchy, 
                                 dictionary_descriptions=DICTIONARY_DESCRIPTIONS)

# Load the LLM query helper
if LLM_BACKEND == 'ollama':
    llm_query_helper = OllamaQueryHelper(model_name=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE,
                                         host=config['OLLAMA']['host'])
else:
    llm_query_helper = LLMQueryHelperOpenAI(api_key=config['AZURE']['apikey'],
                                            endpoint=config['AZURE']['endpoint_openai'],
                                            model_name=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

# Load the CrossEncoder
cross_encoder = CrossEncoder(os.path.join(BASE_DIR, 'cross-encoder', f'cef_{model_name}_{triplet_type}_en_snomed_sim_cand_200_epoch_1_bs_128'))

# Create the Reranker
reranker = Reranker(cross_encoder)

ner_type2hierarchy = {'Body structure' : 'Body structure',
                      'Clinical finding' : 'Clinical finding', 
                      'Procedure' : 'Procedure'}

# Load the text files
#NOTES_FOLDER_PATH = os.path.join(BASE_DIR, 'mimic_data', 'mimic_notes_test/')
NOTES_CSV_PATH = os.path.join(BASE_DIR, 'mimic_data', 'mimic-iv_notes_test_set.csv')
ANNOTATIONS_CSV_PATH = os.path.join(BASE_DIR, 'data', 'df_snomed_ct_el_challenge_UM_UC_combined.csv')

# Load the notes and annotations
mimic = MIMIC_IV_dataset(annotation_csv_path=ANNOTATIONS_CSV_PATH, notes_csv_path=NOTES_CSV_PATH)

# Load the Entity Linker
entity_linker = EntityLinkerAdaptiveLLM(snomed=snomed, snomed_embedder=snomed_embedder, llm_query=llm_query_helper, reranker=reranker,
                                        dictionary_options=dictionary_options, disambiguate_abbreviations=disambiguate_abbreviations,
                                        rephrase=rephrase, replace_span=replace_span, use_fsn=use_fsn, number_of_options=number_of_options, rerank_top_n=rerank_top_n,
                                        ner_type2hierarchy=ner_type2hierarchy)

# Load the Snomed Pipe
snomed_pipe = SnomedPipe(entity_linker)

# Load the IDs of train notes
test_df = pd.read_csv(os.path.join(BASE_DIR, 'mimic_data', 'mimic-iv_notes_test_set.csv'))

# Iterate through texts
for note_id in tqdm(test_df['note_id']):
    print(f'Current note_id: {note_id}', flush=True)

    # Obtain the text
    text = mimic.get_note_text(note_id)
    
    # Obtain the annotated entities
    annotations = mimic.get_note_annotations(note_id)
    
    sentences = mimic.get_annotated_sentences_from_note(note_id=note_id, transform=True, adapt_annotation_index=True)

    # Obtain the predicted entities
    if os.path.exists(os.path.join(DIRECTORY_PATH, f'{EXECUTION_NAME}_{note_id}.csv')):
        predicted_entities = pd.read_csv(os.path.join(DIRECTORY_PATH, f'{EXECUTION_NAME}_{note_id}.csv'))
        
    else:
        for sentence in sentences:
            for entity in sentence.entities:
                entity.ner_type = snomed.get_fsn(snomed.get_top_level_concept(int(entity.label)))
                entity.label = None
                    
        predicted_entities = snomed_pipe.link_entities_given_spans(text, sentences)
        df = annotations_to_df(note_id, predicted_entities, {'label' : 'concept_id', 'start' : 'start', 'end' : 'end'},  add_options=True, add_confidence=True, add_other=True)
        df.to_csv(os.path.join(DIRECTORY_PATH, f'{EXECUTION_NAME}_{note_id}.csv'), index=False)
        llm_query_helper.save_cache()

llm_query_helper.save_cache()

# Save the predictions to a single csv
concatenated_df = concatenate_annotations(folder_path=DIRECTORY_PATH)
concatenated_df.to_csv(os.path.join(DIRECTORY_PATH, f'{EXECUTION_NAME}_predictions.csv'), index=False)
    
llm_query_helper.save_cache()
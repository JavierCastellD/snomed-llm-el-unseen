import json
import os
import sys
import configparser
from tqdm import tqdm

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder

from python_libraries.embedding_models.embedding_model import load_embeddings
from python_libraries.embedding_models.sentencetransformer_EM import SentenceTransformerEM
from python_libraries.entity_linker import EntityLinkerLLMDictionary
from python_libraries.annotated_datasets.TEMIST_dataset import TEMIST_dataset
from python_libraries.llm_queries.LLM_query_helper_openai import LLMQueryHelperOpenAI
from python_libraries.reranker import Reranker
from python_libraries.snomed import Snomed, SnomedEmbedder, SnomedPipe
from python_libraries.utils import load_config, annotations_to_df, concatenate_annotations

# RUN CONFIGURATION
config_run_file = sys.argv[1]
dataset = sys.argv[2]
embedding_type = sys.argv[3]

if len(sys.argv) >= 5:
    triplet_type = sys.argv[4]
else:
    triplet_type = None

config_dic = load_config(config_run_file)

EXECUTION_NAME = config_dic['execution_name']
PREFIX_EXECUTION_NAME = dataset + '_' + embedding_type
if triplet_type is not None:
    PREFIX_EXECUTION_NAME += '_' + triplet_type
EXECUTION_NAME = PREFIX_EXECUTION_NAME + '_' + EXECUTION_NAME #'_CE_mine_' + EXECUTION_NAME
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

ENDPOINT = config['AZURE']['endpoint_openai']
API_KEY = config['AZURE']['apikey']

# SNOMED CT files
SNOMED_VERSION = "20221031" #"20230531"
CONCEPTS_PATH = f"snomed_data/conceptInternational_{SNOMED_VERSION}.txt" 
RELATIONS_PATH = f"snomed_data/relationshipInternational_{SNOMED_VERSION}.txt"
DESCRIPTIONS_PATH = f"snomed_data/descriptionSpanish_{SNOMED_VERSION}.txt" 

# Embedding files
if embedding_type == "sapbert":
    EMBEDDING_MODEL_PATH = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token" 
    EMBEDDING_DICTIONARY_PATH = f"snomed_dictionaries/desc_ent_all_sapbert_mean_base_sct_dict_{dataset}.npz" #"snomed_dictionaries/desc_ent_all_sapbert_mean_base_sct_dict.json" #"snomed_dictionaries/desc_ent_all_sapbert_mean_base_rephrasings_sct_dict.json" 

    CROSS_ENCODER = f"cross-encoder/ce_50_{dataset}_sapbert"
    ID2NAME = f"snomed_dictionaries/id2name_desc_ent_sct_dict_{dataset}.json"
elif embedding_type == "sapbert_esp":
    EMBEDDING_MODEL_PATH = "BSC-NLP4BIA/SapBERT-from-roberta-base-biomedical-clinical-es"
    EMBEDDING_DICTIONARY_PATH = f"snomed_dictionaries/desc_ent_all_sapbert_roberta_es_sct_dict_{dataset}.npz"
    
    CROSS_ENCODER = f"cross-encoder/ce_50_{dataset}_sapbert_spanish"
    ID2NAME = f"snomed_dictionaries/id2name_desc_ent_sapbert_roberta_es_sct_dict_{dataset}.json"
elif embedding_type == "roberta":
    EMBEDDING_MODEL_PATH = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
    EMBEDDING_DICTIONARY_PATH = f"snomed_dictionaries/desc_ent_all_roberta_base_es_sct_dict_{dataset}.npz"
    
    CROSS_ENCODER = f"cross-encoder/ce_50_{dataset}_roberta"
    ID2NAME = f"snomed_dictionaries/id2name_desc_ent_roberta_base_es_sct_dict_{dataset}.json"
elif embedding_type == "sapbert_pubmed_es":
    EMBEDDING_MODEL_PATH = f"sentence_bert_models/sapbert_pubmed_{triplet_type}_es"
    EMBEDDING_DICTIONARY_PATH = f"snomed_dictionaries/desc_ent_all_sapbert_pubmed_{triplet_type}_es_sct_dict_{dataset}.npz"
    
    CROSS_ENCODER = f"cross-encoder/cef_sapbert_pubmed_{triplet_type}_es_{dataset}_sim_cand_200_epoch_1_bs_128"
    ID2NAME = f"snomed_dictionaries/id2name_desc_ent_sapbert_pubmed_{triplet_type}_es_sct_dict_{dataset}.json"
elif embedding_type == "sapbert_roberta_es":
    EMBEDDING_MODEL_PATH = f"sentence_bert_models/sapbert_roberta_{triplet_type}_es"
    EMBEDDING_DICTIONARY_PATH = f"snomed_dictionaries/desc_ent_all_sapbert_roberta_{triplet_type}_es_sct_dict_{dataset}.npz"
    
    #CROSS_ENCODER = f"cross-encoder/ce_50_{dataset}_sapbert_roberta_{triplet_type}_es" -> This is to use the version of CE mine
    CROSS_ENCODER = f"cross-encoder/cef_sapbert_roberta_{triplet_type}_es_{dataset}_sim_cand_200_epoch_1_bs_128"
    ID2NAME = f"snomed_dictionaries/id2name_desc_ent_sapbert_roberta_{triplet_type}_es_sct_dict_{dataset}.json"

DICTIONARY_DESCRIPTIONS = True

# Load SNOMED
snomed = Snomed(CONCEPTS_PATH, RELATIONS_PATH, DESCRIPTIONS_PATH, add_inactive=True)

# Load the embedding model
embedding_model = SentenceTransformerEM(EMBEDDING_MODEL_PATH)

# Load the dictionary
embedding_dictionary = load_embeddings(EMBEDDING_DICTIONARY_PATH)

embedding_dictionary = {k : v for k,v in embedding_dictionary.items() if k not in ['NO_CODE', 'NOMAP', 'NO_CODE_0']}

# Load the id2name
with open(ID2NAME, "r") as id2name_file:
    id2name = json.load(id2name_file)
    
# Load SnomedEmbedder
snomed_embedder = SnomedEmbedder(snomed=snomed, embedding_model=embedding_model, id2name=id2name, embedding_dictionary=embedding_dictionary, dictionary_descriptions=DICTIONARY_DESCRIPTIONS)

# Load the LLM query helper
llm_query_helper = LLMQueryHelperOpenAI(API_KEY, ENDPOINT, model_name='gpt-5-mini', temperature=1)

# Load the CrossEncoder
cross_encoder = CrossEncoder(CROSS_ENCODER)

# Create the Reranker
reranker = Reranker(cross_encoder)

# Load the span dictionary
with open(span_dictionary_path, "r") as dict_file:
    span_dictionary = json.load(dict_file)

ner_type2hierarchy = {'Body structure' : 'Body structure', 
                      'Clinical finding' : 'Clinical finding', 
                      'Procedure' : 'Procedure'}

# Load the text files
if dataset == "distemist":
    NOTES_FOLDER_PATH = "el_datasets/distemist/test_annotated/text_files/"
    ANNOTATIONS_TSV_PATH = "el_datasets/distemist/test_annotated/subtrack2_linking/distemist_subtrack2_test_linking.tsv"
elif dataset == "medprocner":
    NOTES_FOLDER_PATH = "el_datasets/medprocner/medprocner_test/txt/"
    ANNOTATIONS_TSV_PATH = "el_datasets/medprocner/medprocner_test/tsv/medprocner_tsv_test_subtask2.tsv"
elif dataset == "symptemist":
    NOTES_FOLDER_PATH = "el_datasets/distemist/test_annotated/text_files/"
    ANNOTATIONS_TSV_PATH = "el_datasets/symptemist/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2.tsv"

temist = TEMIST_dataset(notes_folder_path=NOTES_FOLDER_PATH, annotations_tsv_path=ANNOTATIONS_TSV_PATH, dataset_type=dataset,
                        ignore_combined=True, ignore_no_code=True)

# Load the Entity Linker
entity_linker = EntityLinkerLLMDictionary(snomed=snomed, snomed_embedder=snomed_embedder, llm_query=llm_query_helper, reranker=reranker, span_dictionary=span_dictionary,
                                          dictionary_options=dictionary_options, disambiguate_abbreviations=disambiguate_abbreviations, llm_for_el=llm_for_el, 
                                          rephrase=rephrase, replace_span=replace_span, use_fsn=use_fsn, number_of_options = number_of_options, rerank_top_n = rerank_top_n, 
                                          trust_training=trust_training, use_reranker=use_reranker, threshold_for_dictionary=threshold_for_dictionary, threshold=threshold,
                                          ner_type2hierarchy=ner_type2hierarchy, spanish_version=True)

# Load the Snomed Pipe
snomed_pipe = SnomedPipe(entity_linker)

saved_notes = 0
# Iterate through texts
for note_id in temist.get_note_ids():#tqdm(temist.get_note_ids()):
    print(f'Current note_id: {note_id}')

    # Obtain the text
    text = temist.get_note_text(note_id)
    
    # Obtain the annotated entities
    annotations = temist.get_note_annotations(note_id)
    
    sentences = temist.get_annotated_sentences_from_note(note_id=note_id, transform=True)

    # Obtain the predicted entities
    if os.path.exists(DIRECTORY_PATH+'/'+EXECUTION_NAME + '_' + note_id + '.csv'):
        predicted_entities = pd.read_csv(DIRECTORY_PATH+'/'+EXECUTION_NAME + '_' + note_id + '.csv')
        
    else:
        n_entities = 0
        for sentence in sentences:
            for entity in sentence.entities:
                entity.ner_type = None #snomed.get_fsn(snomed.get_top_level_concept(int(entity.label)))
                entity.label = None
                n_entities += 1
        
        print(f'Number of entities in sentence: {n_entities}')
        predicted_entities = snomed_pipe.link_entities_given_spans(text, sentences, remap_annotations_to_original=False)
        if len(predicted_entities) != n_entities:
            print(f'Unmatching number of entities between predicted {len(predicted_entities)} and gold {n_entities}')
        df = annotations_to_df(note_id, predicted_entities, {'label' : 'concept_id', 'start' : 'start', 'end' : 'end'},  add_options=True, add_confidence=True, add_other=True)
        print(f'Saving results for {note_id} {saved_notes}/{len(temist.get_note_ids())}')
        df.to_csv(DIRECTORY_PATH+'/'+EXECUTION_NAME + '_' + note_id + '.csv', index=False)
        saved_notes += 1

# Save the predictions to a single csv
concatenated_df = concatenate_annotations(folder_path=DIRECTORY_PATH)
concatenated_df.to_csv(EXECUTION_NAME + '_predictions.csv', index=False)

llm_query_helper.save_cache()

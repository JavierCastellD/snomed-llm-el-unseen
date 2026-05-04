import os
import sys

import json
import pandas as pd
from tqdm import tqdm

from python_libraries.embedding_models.embedding_model import save_embeddings
from python_libraries.embedding_models.sentencetransformer_EM import SentenceTransformerEM
from python_libraries.snomed import Snomed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load model info
model_name = sys.argv[1]
triplet_type = sys.argv[2]
dataset = sys.argv[3]

if dataset == 'snomed':
    DICTIONARY_PATH = os.path.join(BASE_DIR, "snomed_dictionaries", f"desc_ent_all_{model_name}_{triplet_type}_en_sct_dict.npz")
    ID2NAME_PATH = os.path.join(BASE_DIR, "snomed_dictionaries", f"id2name_desc_ent_{model_name}_{triplet_type}_en_sct_dict.json")
    EMBEDDING_PATH = os.path.join(BASE_DIR, "sentence_bert_models", f"{model_name}_{triplet_type}_en")
else:
    DICTIONARY_PATH = os.path.join(BASE_DIR, "snomed_dictionaries", f"desc_ent_all_{model_name}_{triplet_type}_es_sct_dict_{dataset}.npz")
    ID2NAME_PATH = os.path.join(BASE_DIR, "snomed_dictionaries", f"id2name_desc_ent_{model_name}_{triplet_type}_es_sct_dict_{dataset}.json")
    EMBEDDING_PATH = os.path.join(BASE_DIR, "sentence_bert_models", f"{model_name}_{triplet_type}_es")

embedding_model = SentenceTransformerEM(EMBEDDING_PATH)

if dataset == 'snomed':
    BODY_STRUCTURE_ID = 123037004
    CLINICAL_FINDING_ID = 404684003
    PROCEDURES_ID = 71388002

    SNOMED_VERSION = "20230531"
    CONCEPTS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'conceptInternational_{SNOMED_VERSION}.txt')
    RELATIONS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'relationshipInternational_{SNOMED_VERSION}.txt')
    DESCRIPTIONS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'descriptionInternational_{SNOMED_VERSION}.txt')

    # Load SNOMED
    snomed = Snomed(CONCEPTS_PATH, RELATIONS_PATH, DESCRIPTIONS_PATH)

    # Find the relevant hierarchies
    body_structures = snomed.get_children_of(BODY_STRUCTURE_ID)
    clinical_findings = snomed.get_children_of(CLINICAL_FINDING_ID)
    procedures = snomed.get_children_of(PROCEDURES_ID)

    concepts_for_embeddings = body_structures + clinical_findings + procedures
    sct_embedding_dict = {}
    id2name = {}

    # Create auxiliar dictionary
    snomed_keys = []
    snomed_names = []

    for concept_id in tqdm(concepts_for_embeddings):
        descriptions = snomed.get_descriptions(concept_id)
        for i, description in enumerate(descriptions):
            key = str(concept_id) + '_' + str(i)
            snomed_keys.append(key)
            snomed_names.append(description)
else:
    if dataset == 'distemist':
        TSV_PATH = os.path.join(BASE_DIR, 'temist', 'distemist', 'dictionary_distemist.tsv')
    elif dataset == 'medprocner':
        TSV_PATH = os.path.join(BASE_DIR, 'temist', 'medprocner', 'medprocner_gazetteer', 'gazzeteer_medprocner_v1_noambiguity.tsv')
    elif dataset == 'symptemist':
        TSV_PATH = os.path.join(BASE_DIR, 'temist', 'symptemist', 'symptemist_gazetteer', 'symptemist_gazetter_snomed_ES_v2.tsv')
    else:
        raise ValueError("Invalid dataset. Choose from: snomed, distemist, medprocner, symptemist.")

    df = pd.read_csv(TSV_PATH, sep='\t')

    # Create auxiliar dictionary
    snomed_keys = []
    snomed_names = []
    aux_codes = {}
    for i, row in tqdm(df.iterrows()):
        sct_id = row['code']
        name = row['term']
        
        if sct_id not in aux_codes:
            aux_codes[sct_id] = 0
        
        key = str(sct_id) + '_' + str(aux_codes[sct_id])
        aux_codes[sct_id] += 1
        
        snomed_keys.append(key)
        snomed_names.append(name)

# Obtain the embeddings for all the concepts
all_embeddings = embedding_model.get_embeddings(snomed_names)

sct_embedding_dict = {}
id2name = {}
for key, name, embedding in zip(snomed_keys, snomed_names, all_embeddings):
    id2name[key] = name
    sct_embedding_dict[key] = embedding.tolist()

# Save ID2Name and dictionary
with open(ID2NAME_PATH, "w") as id2name_file:
    json.dump(id2name, id2name_file, indent=4)
    
save_embeddings(embedding_dictionary=sct_embedding_dict, filename=DICTIONARY_PATH)
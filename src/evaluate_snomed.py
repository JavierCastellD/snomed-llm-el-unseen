import os
import sys

import json
import pandas as pd
import re

from python_libraries.snomed import Snomed
from python_libraries.utils import load_mimic, get_prediction_results

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def static_preprocess(text):
    t = text.lower()
    t = t.replace("\n", "")
    t = re.sub("[^a-z]", "", t)
    t = re.sub("\s+", " ", t)
    t = t.strip()
    return t

def preprocess(span_text : str):
    span_text = span_text.lower()
    span_text = re.sub('\n', '', span_text)
    return span_text

# Load names
predictions_path = sys.argv[1]

# Load the TEMIST predictions
mimic_train, mimic_test = load_mimic()

# Load annotations
df_train = pd.read_csv(os.path.join(BASE_DIR, 'mimic_data', 'train_annotations.csv'))
df_test = pd.read_csv(os.path.join(BASE_DIR, 'mimic_data', 'test_annotations.csv'))

# Add span column
for df, mimic in [(df_train, mimic_train), (df_test, mimic_test)]:
    spans = []
    prep_spans = []
    for i, row in df.iterrows():
        text = mimic.get_note_text(row['note_id'])
        original_text = text[row['start']:row['end']]
        span_text = preprocess(original_text)
        prep_text = static_preprocess(original_text)
        spans.append(span_text)
        prep_spans.append(prep_text)

    df['span'] = spans
    df['span_prep'] = prep_spans

# Obtain seen codes and mentions
codes_seen = set(df_train['concept_id'])
mentions_seen = set(df_train['span'])

# Load SNOMED names (gazetteer)
with open(os.path.join(BASE_DIR, 'snomed_dictionaries', 'id2name_desc_ent_sct_dict.json')) as f:
    sct_names = json.load(f)
    sct_name_values = set(sct_names.values())
    

# Load the predictions
predictions = pd.read_csv(predictions_path)

# Merge correct with predictions
merged_df = predictions.merge(
    df_test[['note_id', 'start', 'end', 'concept_id', 'span', 'span_prep']],
    on=['note_id', 'start', 'end'],
    how='outer',
    suffixes=('', '_correct')
)

# Rename the concept_id from DF_A to correct_id
merged_df = merged_df.rename(columns={'concept_id_correct': 'correct_id', 'span' : 'text', 'span_prep' : 'text_prep'})

# Fill missing correct_id values with -1
merged_df['correct_id'] = merged_df['correct_id'].fillna(-1).astype('Int64')
merged_df['concept_id'] = merged_df['concept_id'].fillna(-1).astype('Int64')

# Obtain unseen mentions and codes
unseen_mentions = [mention for mention in df_test['span'] if mention not in mentions_seen and mention not in sct_name_values]
unseen_codes = [code for code in df_test['concept_id'] if code not in codes_seen]

# Load SNOMED
SNOMED_VERSION = "20230531"
CONCEPTS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'conceptInternational_{SNOMED_VERSION}.txt')
RELATIONS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'relationshipInternational_{SNOMED_VERSION}.txt')
DESCRIPTIONS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'descriptionInternational_{SNOMED_VERSION}.txt')

snomed = Snomed(CONCEPTS_PATH, RELATIONS_PATH, DESCRIPTIONS_PATH, add_inactive=False)

# Add column with sem_types
sem_types = []
hierarchies = []

for _, row in merged_df.iterrows():
    sem_types.append(snomed.get_semantic_type(int(row['correct_id'])))
    hierarchies.append(snomed.get_fsn(snomed.get_top_level_concept(int(row['correct_id']))))

merged_df['sem_types'] = sem_types
merged_df['hierarchies'] = hierarchies

# Obtain the prediction_results
prediction_results = get_prediction_results(merged_df=merged_df, unseen_mentions=unseen_mentions, unseen_codes=unseen_codes)

print(f"SNOMED CT Entity Linking Challenge  -\t{prediction_results['correct_code']}\t{prediction_results['correct_mentions']}")
print("Type\t\tEmb\tRer\tHits@1\tHits@5\tHits@10\tHits@20")
for type in ['general', 'code', 'mentions']:#, 'mentions_seen_code']:

    rer = round(prediction_results[f'hits_reranker_{type}'][1]/prediction_results[f'len_{type}'] *100, 2)
    emb = round(prediction_results[f'hits_embedding_{type}'][1]/prediction_results[f'len_{type}'] *100, 2)
    h1 = round(prediction_results[f'correct_{type}']/prediction_results[f'len_{type}'] *100, 2)
    h5 = round(prediction_results[f'hits_reranker_{type}'][5]/prediction_results[f'len_{type}'] *100, 2)
    h10 = round(prediction_results[f'hits_reranker_{type}'][10]/prediction_results[f'len_{type}'] *100, 2)
    h20 = round(prediction_results[f'hits_reranker_{type}'][20]/prediction_results[f'len_{type}'] *100, 2)
    
    if type == 'mentions':
        print(f"{type}\t{emb}%\t{rer}%\t{h1}%\t{h5}%\t{h10}%\t{h20}%")
    else:
        print(f"{type}\t\t{emb}%\t{rer}%\t{h1}%\t{h5}%\t{h10}%\t{h20}%")
print()
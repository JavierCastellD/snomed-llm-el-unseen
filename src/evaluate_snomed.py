import os
import sys

import pandas as pd

from python_libraries.snomed import Snomed
from python_libraries.utils import load_mimic, get_prediction_results

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load names
predictions_path = sys.argv[1]

# Load the TEMIST predictions
mimic_train, mimic_test = load_mimic()

# Load annotations
df_test = pd.read_csv(os.path.join(BASE_DIR, 'mimic_data', 'test_annotations.csv'))

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

# Load SNOMED's UC and UM
df_um = pd.read_csv(os.path.join(BASE_DIR, 'data', 'df_snomed_ct_el_challenge_UM.tsv'), sep='\t')
df_uc = pd.read_csv(os.path.join(BASE_DIR, 'data', 'df_snomed_ct_el_challenge_UC.tsv'), sep='\t')

unseen_mentions = [mention for mention in df_um['span']]
unseen_codes = [int(code) for code in df_uc['concept_id']]

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
for type in ['code', 'mentions']:#, 'mentions_seen_code']:

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
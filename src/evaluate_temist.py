import os
import sys

import pandas as pd

from python_libraries.snomed import Snomed
from python_libraries.utils import load_temist_files, get_prediction_results

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load names
dataset= sys.argv[1]
predictions_path = sys.argv[2]

# Load the TEMIST predictions
train_set, test_set, gaz = load_temist_files(dataset=dataset, base_path="..", use_temist_naming=False)

# Load Fernando's UC and UM
df_um = pd.read_csv(os.path.join(BASE_DIR, 'temist', f'{dataset}', 'df_um.tsv'), sep='\t')
df_uc = pd.read_csv(os.path.join(BASE_DIR, 'temist', f'{dataset}', 'df_uc.tsv'), sep='\t')

unseen_mentions = [mention for mention in df_um['term']]
unseen_codes = [int(code) for code in df_uc['code']]

# Remove from test set COMPOSITE and NO_CODE
df_correct = test_set[test_set['sem_rel'] != 'COMPOSITE']
boolean_filter = []
for i, row in df_correct.iterrows():
    boolean_filter.append('+' not in row['code'] and row['code'] not in ['NO_CODE', 'NOMAP'])

df_correct = df_correct[boolean_filter]

# Load the predictions
predictions = pd.read_csv(predictions_path)

# Merge correct with predictions
merged_df = predictions.merge(
    df_correct[['filename', 'start', 'end', 'code', 'sem_rel', 'text']],
    on=['filename', 'start', 'end'],
    how='outer',
    suffixes=('', '_correct')
)

# Rename the concept_id from DF_A to correct_id
merged_df = merged_df.rename(columns={'code': 'correct_id'})

# Fill missing correct_id values with -1
merged_df['correct_id'] = merged_df['correct_id'].fillna(-1).astype('Int64')
merged_df['concept_id'] = merged_df['concept_id'].fillna(-1).astype('Int64')

# Load SNOMED
SNOMED_VERSION = "20221031"
CONCEPTS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'conceptInternational_{SNOMED_VERSION}.txt')
RELATIONS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'relationshipInternational_{SNOMED_VERSION}.txt')
DESCRIPTIONS_PATH = os.path.join(BASE_DIR, 'snomed_data', f'descriptionSpanish_{SNOMED_VERSION}.txt')

snomed = Snomed(CONCEPTS_PATH, RELATIONS_PATH, DESCRIPTIONS_PATH, add_inactive=True)

# Add column with sem_types
sem_types = [snomed.get_semantic_type(int(row['correct_id'])) for _, row in merged_df.iterrows()]
merged_df['sem_types'] = sem_types

# Obtain the prediction_results
prediction_results = get_prediction_results(merged_df=merged_df, unseen_mentions=unseen_mentions, unseen_codes=unseen_codes)

print(f"{dataset}  -\t{prediction_results['correct_code']}\t{prediction_results['correct_mentions']}")
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
"""Evaluation of the control condition without a candidate list (no_candidates = True in the config file).

For the unseen codes and unseen mentions subsets, it reports:
  - Validity: percentage of generated names that exactly match a synonym in the search space
    (matched similarity >= threshold, stored as 'valid_concept' by EntityLinkerNoCandidatesLLM).
  - Strict accuracy: the generated name is valid and corresponds to the correct concept.
  - Relaxed accuracy: the generated name, mapped to its most similar synonym with the bi-encoder,
    corresponds to the correct concept.

Usage (from the repository root):
    python src/evaluate_no_candidates.py snomed <predictions.csv>
    python src/evaluate_no_candidates.py distemist|symptemist|medprocner <predictions.csv>
"""
import os
import re
import sys

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TEMIST_DATASETS = ['distemist', 'symptemist', 'medprocner']


def load_predictions(predictions_path : str, id_column : str) -> pd.DataFrame:
    """Loads the predictions keeping the SNOMED CT IDs as integers (large IDs lose precision as floats)
    and removing duplicated annotations (e.g., a checkpoint saved twice)."""
    preds = pd.read_csv(predictions_path, dtype={'concept_id': 'Int64'})
    if 'filename' in preds.columns and 'note_id' in preds.columns:
        preds = preds.drop(columns=['filename'])
    preds = preds.rename(columns={'filename': id_column, 'note_id': id_column, 'concept_id': 'pred'})
    preds = preds.drop_duplicates(subset=[id_column, 'start', 'end'])
    preds['valid_concept'] = preds['valid_concept'].astype(str).str.lower() == 'true'
    return preds[[id_column, 'start', 'end', 'pred', 'valid_concept', 'llm_generated_name', 'matched_name', 'matched_similarity']]


def load_snomed(predictions_path : str):
    test = pd.read_csv(os.path.join(BASE_DIR, 'mimic_data', 'test_annotations.csv'), dtype={'concept_id': 'Int64'})
    notes = pd.read_csv(os.path.join(BASE_DIR, 'mimic_data', 'mimic-iv_notes_test_set.csv'))
    note_text = dict(zip(notes.note_id, notes.text))
    # Same span computation as evaluate_snomed.py
    test['text'] = [re.sub(r'\n', '', note_text[n][s:e].lower()) for n, s, e in zip(test.note_id, test.start, test.end)]

    merged = test.merge(load_predictions(predictions_path, 'note_id'), on=['note_id', 'start', 'end'], how='left')
    merged = merged.rename(columns={'concept_id': 'correct_id'})

    df_um = pd.read_csv(os.path.join(BASE_DIR, 'data', 'df_snomed_ct_el_challenge_UM.csv'))
    df_uc = pd.read_csv(os.path.join(BASE_DIR, 'data', 'df_snomed_ct_el_challenge_UC.csv'))
    subsets = {'Unseen codes': merged['correct_id'].isin(set(df_uc['concept_id'].astype('int64'))),
               'Unseen mentions': merged['text'].isin(set(df_um['span']))}
    return merged, subsets


def load_temist(dataset : str, predictions_path : str):
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
    from python_libraries.utils import load_temist_files

    _, test_set, _ = load_temist_files(dataset=dataset, base_path=BASE_DIR, use_temist_naming=False)

    # Same filtering as evaluate_temist.py: remove COMPOSITE, combined codes, NO_CODE and NOMAP
    df_correct = test_set[test_set['sem_rel'] != 'COMPOSITE']
    df_correct = df_correct[[('+' not in str(c)) and (c not in ['NO_CODE', 'NOMAP']) for c in df_correct['code']]].copy()
    df_correct['correct_id'] = df_correct['code'].astype('int64')

    merged = df_correct[['filename', 'start', 'end', 'correct_id', 'text']].merge(
        load_predictions(predictions_path, 'filename'), on=['filename', 'start', 'end'], how='left')

    df_um = pd.read_csv(os.path.join(BASE_DIR, 'temist', dataset, 'df_um.tsv'), sep='\t')
    df_uc = pd.read_csv(os.path.join(BASE_DIR, 'temist', dataset, 'df_uc.tsv'), sep='\t')
    subsets = {'Unseen codes': merged['correct_id'].isin(set(df_uc['code'].astype('int64'))),
               'Unseen mentions': merged['text'].isin(set(df_um['term']))}
    return merged, subsets


def evaluate(merged : pd.DataFrame, subsets : dict) -> pd.DataFrame:
    rows = []
    for name, mask in subsets.items():
        sub = merged[mask]
        has_pred = sub['pred'].notna()
        correct = has_pred & (sub['pred'].fillna(-1).astype('int64') == sub['correct_id'].astype('int64'))
        valid = sub['valid_concept'].eq(True)
        rows.append({'Subset': name,
                     'N': len(sub),
                     'Missing predictions': int((~has_pred).sum()),
                     'No response': int(sub['llm_generated_name'].isna().sum() - (~has_pred).sum()),
                     'Validity (%)': round(100 * valid.mean(), 2),
                     'Strict accuracy (%)': round(100 * (correct & valid).mean(), 2),
                     'Relaxed accuracy (%)': round(100 * correct.mean(), 2)})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    dataset = sys.argv[1]
    predictions_path = sys.argv[2]

    if dataset == 'snomed':
        merged, subsets = load_snomed(predictions_path)
    elif dataset in TEMIST_DATASETS:
        merged, subsets = load_temist(dataset, predictions_path)
    else:
        raise ValueError(f'Unknown dataset {dataset}. Use snomed, {", ".join(TEMIST_DATASETS)}.')

    print(f'{dataset} - {os.path.basename(predictions_path)}')
    print(evaluate(merged, subsets).to_string(index=False))

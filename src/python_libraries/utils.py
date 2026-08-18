import ast
import configparser
import numpy as np
import os
import pandas as pd

from python_libraries.annotated_datasets.MIMIC_IV_annotated_dataset import MIMIC_IV_dataset

DF_HEADERS = ['note_id', 'start', 'end', 'concept_id']

def load_model_paths_es(embedding_type : str, triplet_type : str, dataset : str, base_path : str = "..") -> dict:
    """Loads the paths for the embedding model, the embedding dictionary, the cross-encoder, and the id2name dictionary, 
    according to the given embedding type, triplet configuration, and dataset."""
    embedding_info = {}
    if embedding_type == "sapbert_esp":
        embedding_info['emb_model_path'] = "BSC-NLP4BIA/SapBERT-from-roberta-base-biomedical-clinical-es"
        embedding_info['emb_dic_path'] = f"{base_path}/snomed_dictionaries/desc_ent_all_sapbert_roberta_es_sct_dict_{dataset}.npz"
        
        embedding_info['cross_encoder_path'] = f"{base_path}/cross-encoder/ce_50_{dataset}_sapbert_spanish"
        embedding_info['id2name_path'] = f"{base_path}/snomed_dictionaries/id2name_desc_ent_sapbert_roberta_es_sct_dict_{dataset}.json"
    elif embedding_type == "roberta":
        embedding_info['emb_model_path'] = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
        embedding_info['emb_dic_path'] = f"{base_path}/snomed_dictionaries/desc_ent_all_roberta_base_es_sct_dict_{dataset}.npz"
        
        embedding_info['cross_encoder_path'] = f"{base_path}/cross-encoder/ce_50_{dataset}_roberta"
        embedding_info['id2name_path'] = f"{base_path}/snomed_dictionaries/id2name_desc_ent_roberta_base_es_sct_dict_{dataset}.json"
    elif embedding_type in ['sapbert', 'pubmed', 'sapbert_pubmed', 'sapbert_roberta']:
        embedding_info['emb_model_path'] = f"{base_path}/sentence_bert_models/{embedding_type}_{triplet_type}_es"
        embedding_info['emb_dic_path'] = f"{base_path}/snomed_dictionaries/desc_ent_all_{embedding_type}_{triplet_type}_es_sct_dict_{dataset}.npz"
        
        embedding_info['cross_encoder_path'] = f"{base_path}/cross-encoder/cef_{embedding_type}_{triplet_type}_es_{dataset}_sim_cand_200_epoch_1_bs_128"
        embedding_info['id2name_path'] = f"{base_path}/snomed_dictionaries/id2name_desc_ent_{embedding_type}_{triplet_type}_es_sct_dict_{dataset}.json"
    else:
        raise ValueError(f"Embedding type {embedding_type} not recognized.")
    
    return embedding_info

def load_mimic(path_to_mimic_data : str = ".."):
    NOTES_TRAIN_CSV_PATH = os.path.join(path_to_mimic_data, 'mimic_data', 'mimic-iv_notes_training_set.csv')
    NOTES_TEST_CSV_PATH = os.path.join(path_to_mimic_data, 'mimic_data', 'mimic-iv_notes_test_set.csv')
    ANNOTATIONS_TRAIN_CSV_PATH = os.path.join(path_to_mimic_data, 'mimic_data', 'train_annotations.csv')
    ANNOTATIONS_TEST_CSV_PATH = os.path.join(path_to_mimic_data, 'mimic_data', 'test_annotations.csv')

    # Load the notes and annotations
    mimic_train = MIMIC_IV_dataset(annotation_csv_path=ANNOTATIONS_TRAIN_CSV_PATH, notes_csv_path=NOTES_TRAIN_CSV_PATH)
    mimic_test = MIMIC_IV_dataset(annotation_csv_path=ANNOTATIONS_TEST_CSV_PATH, notes_csv_path=NOTES_TEST_CSV_PATH)

    return mimic_train, mimic_test

def load_temist_files(dataset : str = "distemist", base_path : str = "..", use_temist_naming : bool = False):
    """Loads and returns the corresponding train, test, and gazetteer from DisTEMIST, MedProcNER, or SympTEMIST.
    
    Parameters:
        dataset (str):
            Name of the dataset to be loaded: 'distemist', 'symptemist', or 'medprocner'.
        base_path (str):
            Path to the 'temist' folder.
        use_temist_naming (bool):
            Whether the TEMIST datasets folder names remained unchange, i.e, DisTEMIST rather than distemist, etc.
    """
    if dataset == "distemist" or dataset == "DisTEMIST":
        if use_temist_naming:
            folder_path = f"{base_path}/temist/DisTEMIST"
        else:
            folder_path = f"{base_path}/temist/distemist"
        TSV_TRAINING_FILE_1 = f"{folder_path}/training/subtrack2_linking/distemist_subtrack2_training1_linking.tsv"
        TSV_TRAINING_FILE_2 = f"{folder_path}/training/subtrack2_linking/distemist_subtrack2_training2_linking.tsv"
        TSV_TEST_FILE = f"{folder_path}/test_annotated/subtrack2_linking/distemist_subtrack2_test_linking.tsv"
        GAZ_FILE = f"{folder_path}/dictionary_distemist.tsv"

        df1 = pd.read_csv(TSV_TRAINING_FILE_1, sep='\t')
        df2 = pd.read_csv(TSV_TRAINING_FILE_2, sep='\t')

        train_set = pd.concat([df1, df2], ignore_index=True)
        std_map = {'off0' : 'start',
                    'off1' : 'end',
                    'span' : 'text',
                    'semantic_rel' : 'sem_rel'}            
    elif dataset == "medprocner" or dataset == "MedProcNER":
        if use_temist_naming:
            folder_path = f"{base_path}/temist/MedProcNER"
        else:
            folder_path = f"{base_path}/temist/medprocner"
        TSV_TRAINING_FILE = f"{folder_path}/medprocner_train/tsv/medprocner_tsv_train_subtask2.tsv"
        TSV_TEST_FILE = f"{folder_path}/medprocner_test/tsv/medprocner_tsv_test_subtask2.tsv"
        GAZ_FILE = f"{folder_path}/medprocner_gazetteer/gazzeteer_medprocner_v1_noambiguity.tsv"

        train_set = pd.read_csv(TSV_TRAINING_FILE, sep='\t')
        std_map = {'start_span' : 'start',
                    'end_span' : 'end'}
    elif dataset == "symptemist" or dataset == "SympTEMIST":
        if use_temist_naming:
            folder_path = f"{base_path}/temist/SympTEMIST"
        else:
            folder_path = f"{base_path}/temist/symptemist"
        TSV_TRAINING_FILE = f"{folder_path}/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2.tsv"
        TSV_TEST_FILE = f"{folder_path}/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2.tsv"
        GAZ_FILE = f"{folder_path}/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv"

        train_set = pd.read_csv(TSV_TRAINING_FILE, sep='\t')
        std_map = {'span_ini' : 'start',
                    'span_end' : 'end'}

    gaz = pd.read_csv(GAZ_FILE, sep="\t", dtype={'code' : str})
    train_set = train_set.rename(columns=std_map)
    test_set = pd.read_csv(TSV_TEST_FILE, sep="\t")   
    test_set = test_set.rename(columns=std_map)

    return train_set, test_set, gaz

def concatenate_annotations(folder_path : str):
    """Loads all the checkpoints and concatenates them into a single DataFrame."""
    dfs = []
    for file_path in os.listdir(f"{folder_path}"):
        df_new = pd.read_csv(f"{folder_path}/{file_path}")
        dfs.append(df_new)

    concatenated = pd.concat(dfs, ignore_index=True)

    concatenated = concatenated.rename(columns={'note_id' : 'filename'})
    concatenated = concatenated.drop_duplicates()

    return concatenated

def annotations_to_df(note_id : str, annotations : list[dict], key_maps : dict[str] = None, add_options : bool = False, add_confidence : bool = False, add_other : bool = False) -> pd.DataFrame:
    """Function that transform the annotations from a note into a DataFrame with the structure needed for the Evaluator.
    
    Parameters:
        note_id (str):
            Identifier of the note.
        annotations (list[dict]):
            List that contains the annotations. Each annotation is a dict that should at least have the keys: label, start, and end. If those keys are not found in the dictionary, a mapping dictionary should be given.
        key_maps (dict[str]):
            Dictionary to map the 'label', 'start', and 'end' keys to the respective ones used by the user.
        add_options (bool):
            Whether to add options to the dataframe.
        add_confidence (bool):
            Whether to add the confidence value to the dataframe.

    Returns:
        A pandas DataFrame with the columns: note_id, start, end, concept_id, and ner_id.
    """
    values = []
    other_headers = []

    for ann in annotations:
        values_annotations = []
        concept_id = ann['label'] if key_maps is None else ann[key_maps['label']]
        start = ann['start'] if key_maps is None else ann[key_maps['start']]
        end = ann['end'] if key_maps is None else ann[key_maps['end']]
        
        values_annotations = [note_id, start, end, concept_id]
        
        if add_options:
            options = ann['options'] if key_maps is None or 'options' not in key_maps else ann[key_maps['options']]
            values_annotations.append(options)
        
        if add_confidence:
            confidence = ann['confidence'] if key_maps is None or 'confidence' not in key_maps else ann[key_maps['confidence']]
            values_annotations.append(confidence)
        
        if add_other:
            other = ann['other'] if key_maps is None or 'other' not in key_maps else ann[key_maps['other']]
            for k, v in other.items():
                if k not in other_headers:
                    other_headers.append(k)
                
                values_annotations.append(v)

        values.append(values_annotations)        
    
    headers = DF_HEADERS
    if add_options:
        headers = headers + ["options"]
    
    if add_confidence:
        headers = headers + ["confidence"]

    if add_other:
        headers = headers + other_headers
    
    df = pd.DataFrame(values, columns=headers)

    if df.concept_id.dtype == 'O':
        df.concept_id = df.concept_id.astype(np.int64)

    return df

def load_config(config_path : str):
    config_run = configparser.ConfigParser()
    config_run.read(config_path)

    config_dic = {}

    config_dic['execution_name'] = config_run.get('CONF', 'execution_name', fallback=None)
    config_dic['span_dictionary_path'] = config_run.get('CONF', 'span_dictionary_path', fallback=None)
    config_dic['checkpoints_folder'] = config_run.get('CONF', 'checkpoints_folder', fallback='el_checkpoints')

    config_dic['disambiguate_abbreviations'] = config_run.getboolean('CONF', 'disambiguate_abbreviations')
    config_dic['llm_for_el'] = config_run.getboolean('CONF', 'llm_for_el')
    config_dic['rephrase'] = config_run.getboolean('CONF', 'rephrase')
    config_dic['replace_span'] = config_run.getboolean('CONF', 'replace_span')
    config_dic['use_fsn'] = config_run.getboolean('CONF', 'use_fsn')
    config_dic['number_of_options'] = config_run.getint('CONF', 'number_of_options')
    config_dic['rerank_top_n'] = config_run.getint('CONF', 'rerank_top_n')
    config_dic['trust_training'] = config_run.getboolean('CONF', 'trust_training', fallback=False)
    config_dic['use_reranker'] = config_run.getboolean('CONF', 'use_reranker')

    config_dic['threshold_for_dictionary'] = None if not config_run.has_option('CONF', 'threshold_for_dictionary') or config_run['CONF']['threshold_for_dictionary'] == 'None' else config_run.getfloat('CONF', 'threshold_for_dictionary')
    config_dic['threshold'] = None if config_run['CONF']['threshold'] == 'None' else config_run.getfloat('CONF', 'threshold')

    # CONFIGURATION FOR DICT OPTIONS
    config_dic['dictionary_options'] = {}

    if config_run.has_section('OPTS'):
        config_dic['dictionary_options']['method'] = config_run['OPTS']['method']
        config_dic['dictionary_options']['use_new_span'] = config_run.getboolean('OPTS', 'use_new_span')
        config_dic['dictionary_options']['use_both_spans_for_dict'] = config_run.getboolean('OPTS', 'use_both_spans_for_dict')
        config_dic['dictionary_options']['use_synonyms'] = config_run.getboolean('OPTS', 'use_synonyms')
        config_dic['dictionary_options']['use_new_span_for_llm'] = config_run.getboolean('OPTS', 'use_new_span_for_llm')
        config_dic['dictionary_options']['use_new_span_for_validation_and_llm'] = config_run.getboolean('OPTS', 'use_new_span_for_validation_and_llm')
        config_dic['dictionary_options']['trust_validation'] = config_run.getboolean('OPTS', 'trust_validation')
        config_dic['dictionary_options']['rerank_validation'] = config_run.getboolean('OPTS', 'rerank_validation')
        config_dic['dictionary_options']['top_concepts_embeddings'] = config_run.getint('OPTS', 'top_concepts_embeddings')
        config_dic['dictionary_options']['top_validation_reranked'] = config_run.getint('OPTS', 'top_validation_reranked')
        config_dic['dictionary_options']['preprocess_span'] = config_run.getboolean('OPTS', 'preprocess_span', fallback=False)
        config_dic['dictionary_options']['preprocess_new_span'] = config_run.getboolean('OPTS', 'preprocess_new_span', fallback=False)

    if config_run.has_option('CONF', 'threshold_for_reranker'):
        config_dic['dictionary_options']['threshold_reranker'] = None if config_run['CONF']['threshold_for_reranker'] == 'None' else config_run.getfloat('CONF', 'threshold_for_reranker')
    else:
        config_dic['dictionary_options']['threshold_reranker'] = None

    if config_run.has_option('CONF', 'threshold_for_llm'):
        config_dic['dictionary_options']['threshold_llm'] = None if config_run['CONF']['threshold_for_llm'] == 'None' else config_run.getfloat('CONF', 'threshold_for_llm')
    else:
        config_dic['dictionary_options']['threshold_llm'] = None

    return config_dic, config_run

def get_prediction_results(merged_df, unseen_mentions = None, unseen_codes = None) -> dict:
    """Returns a dictionary with predictions, which contains the following keys:
        -correct_general
        -len_general
        -hits_embedding_general (dictionary with keys 1, 5, 10, 20, 25)
        -hits_reranker_general (dictionary with keys 1, 5, 10, 20, 25)

        If unseen_mentions is not None, it also contains the same keys with _mentions rather than _general
        If unseen_codes is not None, it also contains the same keys with _codes rather than _general
    """
    data_dic = {}

    subsets_to_identify = ['general']
    if unseen_codes is not None:
        subsets_to_identify.append('code')
    if unseen_mentions is not None:
        subsets_to_identify.append('mentions')
        # subsets_to_identify.append('mentions_seen_code')

    for unseen_type in subsets_to_identify:
        if unseen_type == 'general':
            f_merged_df = merged_df
        elif unseen_type == 'code':
            f_merged_df = merged_df[merged_df['correct_id'].isin(unseen_codes)]
        elif unseen_type == 'mentions':
            f_merged_df = merged_df[merged_df['text'].isin(unseen_mentions)]
        # elif unseen_type == 'mentions_seen_code':
        #     f_merged_df = merged_df[merged_df['text'].isin(unseen_mentions)]
        #     f_merged_df = f_merged_df[~f_merged_df['correct_id'].isin(unseen_codes)]

        correct_df = f_merged_df[f_merged_df['concept_id'] == f_merged_df['correct_id']]

        data_dic[f'correct_{unseen_type}'] = len(correct_df)
        data_dic[f'len_{unseen_type}'] = len(f_merged_df)

        # Pipeline stats: hits@1, 3, 5, 10, 20, 25, 50 per original options and hits@1, 3, 5, 10 per reranker
        hits_threshold = [1, 5, 10, 20, 25]
        hits = {h : 0 for h in hits_threshold}

        hits_rerank = {h : 0 for h in hits_threshold}

        for i, row in f_merged_df.iterrows():
            if pd.isna(row['original_concepts']):
                continue
            options = ast.literal_eval(row['original_concepts'])
            correct_id = int(row['correct_id'])

            for hit in hits_threshold:
                if correct_id in options[:hit]:
                    hits[hit] += 1

            if 'reranker' in f_merged_df and not pd.isna(row['reranker']):
                rerank_options = ast.literal_eval(row['reranker'])
                for hit in hits_threshold:
                    if correct_id in rerank_options[:hit]:
                        hits_rerank[hit] += 1

        data_dic[f'hits_embedding_{unseen_type}'] = hits
        data_dic[f'hits_reranker_{unseen_type}'] = hits_rerank
    
        # To get accuracy per sem_type/hierarchy
        column_types = []
        if 'sem_types' in f_merged_df:
            column_types.append('sem_types')
        if 'hierarchies' in f_merged_df:
            column_types.append('hierarchies')
        
        for column_type in column_types:
            data_dic[f'correct_{unseen_type}_{column_type}'] = {}
            data_dic[f'len_{unseen_type}_{column_type}'] = {}
            for f_type in f_merged_df[column_type].unique():
                # Filter per type
                f_merged_df_type = f_merged_df[f_merged_df[column_type] == f_type]
                
                # Find the correct ones
                correct_df = f_merged_df_type[f_merged_df_type['concept_id'] == f_merged_df_type['correct_id']]

                data_dic[f'correct_{unseen_type}_{column_type}'][f_type] = len(correct_df)
                data_dic[f'len_{unseen_type}_{column_type}'][f_type] = len(f_merged_df_type)
    return data_dic
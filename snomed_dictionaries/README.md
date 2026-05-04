# Folder for embedding dictionaries/database

Folder that contains the embedding dictionaries generated through *create_sct_dictionary.py*. Each embedding database is made up of two files: a npz file which contains the embedding for each description/synonym of a concept, and a json dictionary file (id2name), which maps the corresponding ID of a synonym to its textual form.

It is not the most elegant solution, but for now the embedding dictionaries' names are hardcoded using a naming convention. 
For the SNOMED CT Entity Linking Challenge it is the following:

- desc_ent_all_{model_name}_{triplet_type}_en_sct_dict.npz
- id2name_desc_ent_{model_name}_{triplet_type}_en_sct_dict.json

For TEMIST is the following naming convention:

- desc_ent_all_{model_name}_{triplet_type}_es_sct_dict_{dataset}.npz
- id2name_desc_ent_{model_name}_{triplet_type}_es_sct_dict_{dataset}.json

If you want to use your own, you should change the following files:

- In *create_sct_dictionary.py*, as this is where the dictionary of embeddings is created.
- In _utils.py_, the function _load_model_paths_es_ loads the embedding dictionary for the Spanish datasets (TEMIST) for the prediction script.
- In *predict_snomed.py*, as this is where the embedding dictionary is loaded for the SNOMED CT Entity Linking Challenge.
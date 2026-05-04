# Folder for trained bi-encoders 

The purpose of this folder is to store the trained bi-encoders.

It is not the most elegant solution, but for now the embedding models' names are hardcoded using the following naming convention: {model_name}_{triplet_type}_{language}

If you want to include your own models, you should change the following files:

- In *create_sct_dictionary.py*, as this is where the dictionary of embeddings is created.
- In _utils.py_, the function *load_model_paths_es* loads the models for the Spanish datasets (TEMIST) for the prediction script.
- In *predict_snomed.py*, as this is where the embedding model is loaded for the SNOMED CT Entity Linking Challenge.

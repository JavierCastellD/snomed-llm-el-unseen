# Folder for trained cross-encoders

The purpose of this folder is to store the trained cross-encoders.

It is not the most elegant solution, but for now the cross-enocders' names are hardcoded using the following naming convention: 'cef_{model_name}_{triplet_type}_{language}_{dataset}_sim_cand_200_epoch_1_bs_128'

If you want to include your cross-enocders with your own naming, you should change the following files:

- In _utils.py_, the function _load_model_paths_es_ loads the cross-encoders for the Spanish datasets (TEMIST) for the prediction script.
- In *predict_snomed.py*, as this is where the cross-enocder model is loaded for the SNOMED CT Entity Linking Challenge.
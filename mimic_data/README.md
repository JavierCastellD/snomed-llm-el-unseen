# Folder for MIMIC data from SNOMED CT Entity Linking Challenge

The SNOMED CT Entity Linking Challenge can be downloaded from their [PhysioNet page](https://physionet.org/content/snomed-ct-entity-challenge/1.2.1/).

Four files should be stored here:
- *train_annotations.csv* and *test_annotations.csv*, which contains the spans and codes for the annotations.
- *mimic-iv_notes_training_set.csv* and *mimic-iv_notes_test_set.csv*, which contains the clinical notes where the annotations were made.

If you want to use your own naming convention, you should made the appropiate path changes in the following files:
- *predict_snomed.py*, which loads the annotations to generate the predictions. 
- *evaluate_snomed.py*, which loads the test annotations to evaluate the predictions.
- In _utils.py_, the function *load_mimic* loads the texts and annotations.

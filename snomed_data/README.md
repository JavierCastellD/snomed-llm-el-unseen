# Folder for SNOMED CT files

Download the RF2 files of the international edition of SNOMED CT from [NIH](https://www.nlm.nih.gov/healthit/snomedct/international.html).

The SNOMED CT files can be prepared in the *snomed_data* folder by using the following script:
```
python create_snomed_data.py <path_to_snomed_ct_international_rf2_folder>
```

If you want to include Spanish description's for SNOMED CT concepts, the Spanish edition can be downloaded from the web of the [SNOMED CT Spanish's Ministry of Health](https://snomed-ct.sanidad.gob.es/snomed-ct/solicitudLicencia.do). To use the Spanish version, the international edition is also required. The files can be prepared by using:
```{python}
python create_snomed_data.py <path_to_snomed_ct_international_rf2_folder> <path_to_snomed_ct_spanish_rf2_folder>
```

If you want to use your own naming convention, you should made the appropiate path changes in the following files:
- *create_sct_dictionary.py*, which loads SNOMED CT data to obtain the descriptions of each concept.
- *predict_snomed.py* and *predict_temist.py*, which loads SNOMED CT for the Entity Linking Pipeline. 
- *evaluate_snomed.py* and *evaluate_temist.py*, which loads SNOMED CT to add hierarchical information for the evaluation.
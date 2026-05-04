# Folder for data for DisTEMIST, SympTEMIST, and MedProcNER

For each dataset, there should be a folder with the corresponding files extracted from their shared task homepages: [DisTEMIST](https://temu.bsc.es/distemist/), [SympTEMIST](https://temu.bsc.es/symptemist/), and [MedProcNER](https://temu.bsc.es/medprocner/). The main folder for each dataset should use lowercase for the naming (*distemist* for *DisTEMIST*, etc.).

Additionally, you should include the files for the *unseen mentions* and *unseen codes* subsets in each folder, extracted from [ClinLinker-KG's GitHub page](https://github.com/ICB-UMA/KnowledgeGraph/tree/fernando/data).

If you want to use your own naming convention, you should made the appropiate path changes in the following files:
- *create_sct_dictionary.py*, which loads the gazetteers.
- *predict_temist.py*, which loads the annotations and text files to generate the predictions. 
- *evaluate_temist.py*, which loads the *unseen codes* and *unseen mentions* subsets.
- In _utils.py_, the function *load_temist_files* loads the train, test, and the gazetteer.

# LLMs for the Unknown: Generalizing Clinical Entity Linking to Unseen Mentions and Concepts
This repository contains our approach to perform medical entity linking, focusing on unseen mentions and unseen concepts. Our proposed method follows a three step pipeline:

1. Bi-encoder candidate retrieval.
2. Cross-encoder candidate reranking.
3. LLM-based candidate selection.

![ADD IMAGE HERE]()

## Train bi-encoder and cross-encoder from the article

Our approach, as described in the paper, uses a trained bi-encoder and trained cross-encoder following the methodology described by [ClinLinker-KB](https://github.com/ICB-UMA/ClinLinker-KB). 

### Triplet generation and cross-encoder training
Follow the steps described in their [repository](https://github.com/ICB-UMA/ClinLinker-KB), as there is one script for [triplet definition](https://github.com/ICB-UMA/ClinLinker-KB/blob/master/notebooks/triplets_definition.ipynb) and another one for [training the cross-encoder](https://github.com/ICB-UMA/ClinLinker-KB/blob/master/scripts/cross_encoder_training.py).

### Bi-encoder training
To train your bi-encoder using the new triplets generated, follow the steps described in [SapBERT's repository](https://github.com/cambridgeltl/sapbert).

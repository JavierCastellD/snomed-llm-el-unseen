from abc import ABC, abstractmethod

from ..ner_model.sentence_ner import Sentence

class EntityLinker(ABC):
    """Abstract class that represents the entity linking pipeline. Each subclass is expected to provide functionality to link the entities found in a Sentence to their corresponding concept."""

    @abstractmethod
    def link_entities_from_sentence(self, sentence : Sentence):
        """Method that finds the relevant concept in the KG, ontology or vocabulary for the Entities
        in the Sentence. The concepts are assigned to the Entities.

        Parameters:
            sentence (Sentence):
                Sentence object that contains the entities to be linked, as well as additional information about the sentence.
        """
        pass
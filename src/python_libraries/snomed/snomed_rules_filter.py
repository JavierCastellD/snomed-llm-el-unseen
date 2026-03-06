import warnings

import pandas as pd
import re

from .snomed import Snomed, IS_A_ID

class SnomedRulesFilter:
    """Class that represents the domain and range constraints for SNOMED CT and offers filtering functions based
    on that. It is created using the attributeRange file, usually named der2_ssccRefset_MRCMAttributeRangeSnapshot_INT_[SCT-version].txt.
    
    Attributes:
        snomed (Snomed):
            Object that represents SNOMED CT.
        dom_range_rel (dict[int][int]):
            Dictionary that contains the possible relationships for a given pair of domain and range.
        dom_rel_range (dict[int][int]):
            Dictionary that contains the possible ranges for a given pair of domain and relationship.
        possible_domains (set[int]):
            Contains the possible domains across SNOMED CT.
        possible_ranges (set[int]):
            Contains the possible ranges across SNOMED CT.
    
    """
    def __init__(self, snomed : Snomed, attribute_range_file_path : str):
        """Method that initializes the SnomedRulesFilter object given the attributeRange file and SNOMED CT.
        This method builds the corresponding dictionaries and domain/range sets.
        
        Parameters:
            snomed (Snomed):
                Snomed object that contains the information about SNOMED CT.
            attribute_range_file_path (str):
                String that contains the path to the attributeRange file.
        """
        range_df = pd.read_csv(attribute_range_file_path, sep = '\t')

        # The possible domains and ranges sets are used for inherited rules
        self.possible_domains = set()
        self.possible_ranges = set()

        # We store SNOMED CT to offer some functionality that might be useful
        self.snomed = snomed

        # Domain and ranges here are pseudo top level concepts
        # For a given domain and range, it has the possible relationships
        self.dom_range_rel = {}
        
        # For a given domain and relationship, it has the possible ranges
        self.dom_rel_range = {}
        

        # Each attribute rule defines some domain and range constraints for a givel relationship in SNOMED CT
        for attribute_rule, rel_id, text_range in zip(range_df['attributeRule'], 
                                                      range_df['referencedComponentId'], 
                                                      range_df['rangeConstraint']):
            # To obtain the domains, we first need to extract it from the attribute rules
            text_possible_domains = attribute_rule.split('OR (<<')
            text_domain = ''

            for text_possible_domain in text_possible_domains:
                text_domain += text_possible_domain.split(':')[0]
            
            # We obtain the domain and range for the relation rel_id
            domains = [int(domain_id) for domain_id in re.findall('\d\d+', text_domain)]
            ranges = [int(range_id) for range_id in re.findall('\d\d+', text_range)]

            # Then we need to add them to the dictionaries for further easy access
            for domain_id in domains:
                self.possible_domains.add(domain_id)

                for range_id in ranges:
                    self.possible_ranges.add(range_id)

                    # This dictionary stores the set of relationships available 
                    # given the domain and range
                    if domain_id in self.dom_range_rel:
                        if range_id in self.dom_range_rel[domain_id]:
                            self.dom_range_rel[domain_id][range_id].add(rel_id)
                        else:
                            self.dom_range_rel[domain_id][range_id] = set([rel_id])
                    else:
                        self.dom_range_rel[domain_id] = {range_id : set([rel_id])}
                
                # This dictionary stores the list of ranges available given a domain
                # and relationship
                if domain_id in self.dom_rel_range:
                    if rel_id in self.dom_rel_range[domain_id]:
                        current_ranges = self.dom_rel_range[domain_id][rel_id] + ranges
                        self.dom_rel_range[domain_id][rel_id] = list(set(current_ranges))
                    else:
                        self.dom_rel_range[domain_id][rel_id] = ranges
                else:
                    self.dom_rel_range[domain_id] = {rel_id : ranges}

        # For each pair of different domain
        for domain_A in self.possible_domains:
            for domain_B in self.possible_domains:
                if domain_A != domain_B:
                    # We check if one is the child of the other
                    if self.snomed.is_child_of(domain_A, domain_B):
                        # In which case we should add their (domain, rel, range) triplets
                        # to the dictionary
                        for range_id, rels in self.dom_range_rel[domain_B].items():
                            if range_id in self.dom_range_rel[domain_A]:
                                self.dom_range_rel[domain_A][range_id].union(rels)
                            else:
                                self.dom_range_rel[domain_A][range_id] = rels

                        for rel_id, ranges in self.dom_rel_range[domain_B].items():
                            if rel_id in self.dom_rel_range[domain_A]:
                                self.dom_rel_range[domain_A][rel_id].union(ranges)
                            else:
                                self.dom_rel_range[domain_A][rel_id] = ranges
                           
    def get_relations(self, subject_id : int, object_id : int) -> list[int]:
        """Method that returns the possible relationships that can exist between two given concepts in SNOMED CT.

        Parameters:
            subject_id (int):
                ID of the SNOMED CT concept that acts as subject of the relationship and defines the domain.
            object_id (int):
                ID of the SNOMED CT concept that acts as object of the relationship and defines the range.

        Returns:
            A list that contains the IDs of the relationships that can exist between subject_id and object_id 
            according to the SNOMED CT logical model.
        """
        # We first obtain the pseudo top level domain and range concepts
        top_domains = self.snomed.get_top_concept_list(subject_id, self.possible_domains)
        top_ranges = self.snomed.get_top_concept_list(object_id, self.possible_ranges)

        if len(top_domains) == 0:
            warnings.warn('Empty domains. This might be caused because the subject_id does not belong to SNOMED,\
                           or because it is not related with any of the possible domains.')
            return []

        if len(top_ranges) == 0:
            warnings.warn('Empty ranges. This might be caused because the subject_id does not belong to SNOMED,\
                           or because it is not related with any of the possible domains.')
            return []

        possible_relations = []
        
        # We iterate through the possible domains related to subject_id and the ranges related to object_id
        for domain_id in top_domains:
            for range_id in top_ranges:
                # and extract the relationships that exist between each pair (domain_id, range_id)
                if domain_id in self.dom_range_rel and range_id in self.dom_range_rel[domain_id]:
                    possible_relations += self.dom_range_rel[domain_id][range_id]

        return list(set(possible_relations))

    def get_ranges(self, subject_id : int, rel_id : int) -> list[int]:
        """Method that returns the possible ranges that can exist for a given subject concept and relationship
        in SNOMED CT.

        Parameters:
            subject_id (int):
                ID of the SNOMED CT concept that acts as subject of the relationship and defines the domain.
            rel_id (int):
                ID of the SNOMED CT relationship.

        Returns:
            A list that contains the IDs of the valid ranges for the subject_id and relationship according
            to the SNOMED CT logical model.
        """
        # We first obtain the pseudo top level domain concepts
        top_domains = self.snomed.get_top_concept_list(subject_id, self.possible_domains)

        if len(top_domains) == 0:
            warnings.warn('Empty domains. This might be caused because the subject_id does not belong to SNOMED,\
                           or because it is not related with any of the possible domains.')
            return []
        
        possible_ranges = []

        for domain_id in top_domains:
            if domain_id in self.dom_rel_range and rel_id in self.dom_rel_range[domain_id]:
                possible_ranges += self.dom_rel_range[domain_id][rel_id]
        
        return list(set(possible_ranges))
    
    def is_in_valid_range(self, subject_id : int, rel_id : int, concept_id : int) -> bool:
        """Method that returns whether concept_id is a valid range for the triplet (subject_id, rel_id, concept_id).
        
        Parameters:
            subject_id (int):
                ID of the SNOMED CT concept that acts as subject of the relationship and defines the domain.
            rel_id (int):
                ID of the SNOMED CT relationship.
            concept_id (int):
                ID of the SNOMED CT concept that would act as object of the relationship.

        Returns:
            True if the concept is in the valid range. Otherwise returns False.
        """
        # Obtain the valid ranges
        valid_ranges = self.get_ranges(subject_id, rel_id)

        # Obtain the top ranges of the concept
        top_ranges = self.snomed.get_top_concept_list(concept_id, self.possible_ranges)

        # If any of the top ranges is in valid ranges, then return True. Returns False otherwise
        return any(range_id in valid_ranges for range_id in top_ranges)      
        
    def get_possible_relations(self, subject_id : int) -> list[int]:
        """Method that returns the possible relationships that can exist for a given SNOMED CT concept.

        Parameters:
            subject_id (int):
                ID of the SNOMED CT concept that acts as subject of the relationship and defines the domain.

        Returns:
            A list that contains the IDs of the relationships that can exist using subject_id as the domain
            according to the SNOMED CT logical model.
        """
        # We first obtain the pseudo top level domain and range concepts
        top_domains = self.snomed.get_top_concept_list(subject_id, self.possible_domains)
        
        if len(top_domains) == 0:
            warnings.warn('Empty domains. This might be caused because the subject_id does not belong to SNOMED,\
                           or because it is not related with any of the possible domains.')
            return []
        
        
        possible_relations = []
        
        # We iterate through the possible domains related to subject_id and the ranges related to object_id
        for domain_id in top_domains:
            # and extract the relationships that exist between each pair (domain_id, range_id)
            if domain_id in self.dom_rel_range:
                possible_relations += list(self.dom_rel_range[domain_id].keys())

        return list(set(possible_relations))
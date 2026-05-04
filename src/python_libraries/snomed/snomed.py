from collections.abc import Iterable
from collections import deque
import re
import warnings

import pandas as pd


# SCT-Codes
FULLY_SPECIFIED_NAME_ID = 900000000000003001
IS_A_ID = 116680003
ROOT_CONCEPT = 138875005
METADATA_ROOT = 900000000000441003
TOP_CONCEPTS = [123037004, 404684003, 308916002, 272379006, 363787002,
                410607006, 373873005, 78621006, 260787004, 71388002,
                362981000, 419891008, 243796009, 48176007, 370115009,
                123038009, 254291000, 105590001, 900000000000441003]

class Snomed:
    """Represents the SNOMED CT ontology-based terminology.

    Attributes:
        concepts (dict):
            A dictionary that maps each SCT-ID to a dictionary with the following keys: FSN, description, relations,
            relationsAux, definition, and semantic_type. Relations is a list of tuples (tail concept SCT-ID, relationship
            SCT-ID) that represents the relationships of the concept. RelationsAux is similar, but contains the inverse
            of the is-a relationships.

        metadata (dict):
            A dictionary that maps each SCT-ID from a metadata concept to a dictionary with the same keys as the
            concepts' attribute.
    """
    def __init__(self, con_path: str, rel_path: str, desc_path: str, def_path: str = None, add_inactive : bool = False):
        """Loads SNOMED-CT from certain files.

        Parameters:
            con_path (str):
               Path to the concepts file from the international release.
            rel_path (str):
                Path to the relationship file from the international release.
            desc_path (str):
                Path to the descriptions file from a national or the international release.
            def_path (str):
                Path to the definitions file from a national or the international release. This is optional.
        """

        concepts_pd = pd.read_csv(con_path, delimiter='\t')

        # We check the status of the concepts in the international release to avoid differences
        # with the possible national releases
        concept_status = dict()
        for scid, active in zip(concepts_pd['id'], concepts_pd['active']):
            concept_status[scid] = active

        descriptions_pd = pd.read_csv(desc_path, delimiter='\t')

        self.concepts = dict()
        for scid, active, typeID, description in zip(descriptions_pd['conceptId'],
                                                    descriptions_pd['active'],
                                                    descriptions_pd['typeId'],
                                                    descriptions_pd['term']):

            # If the concept is active
            if add_inactive or (active == 1 and (scid in concept_status and concept_status[scid] == 1)):
                if scid not in self.concepts:
                    self.concepts[scid] = {'FSN': '', 'description': [], 'relations': [],
                                           'relationsAux': [], 'definition': '', 'semantic_type': ''}

                if typeID == FULLY_SPECIFIED_NAME_ID:
                    match = re.search('(.+)(\(.+\))', description)
                    if match is not None:
                        self.concepts[scid]['FSN'] = match.group(1).strip() 
                        self.concepts[scid]['semantic_type'] = match.group(2).strip()[1:-1]
                        if match.group(1).strip() not in self.concepts[scid]['description']:
                            self.concepts[scid]['description'].append(match.group(1).strip())
                    else:
                        self.concepts[scid]['FSN'] = description
                        if description not in self.concepts[scid]['description']:
                            self.concepts[scid]['description'].append(description)
                elif description not in self.concepts[scid]['description']:# and description == description:
                    self.concepts[scid]['description'].append(description)



        # The file of definitions is optional
        if def_path is not None:
            definition_pd = pd.read_csv(def_path, delimiter='\t')
            for active, scid, definition in zip(definition_pd['active'],
                                               definition_pd['conceptId'],
                                               definition_pd['term']):
                if (add_inactive or active == 1) and scid in self.concepts:
                    self.concepts[scid]['definition'] = definition

        relations_pd = pd.read_csv(rel_path, delimiter='\t')
        for active, sourceID, destID, typeID in zip(relations_pd['active'],
                                                    relations_pd['sourceId'],
                                                    relations_pd['destinationId'],
                                                    relations_pd['typeId']):
            if (add_inactive or active == 1) and sourceID in self.concepts and destID in self.concepts:
                self.concepts[sourceID]['relations'].append([destID, typeID])
                if typeID == IS_A_ID:
                    self.concepts[destID]['relationsAux'].append(sourceID)

        # This is to extract the metadata
        unexplored_metadata = [METADATA_ROOT]
        self.metadata = dict()

        while len(unexplored_metadata) > 0:
            sourceID = unexplored_metadata.pop(0)

            if sourceID not in self.metadata:
                for destID in self.concepts[sourceID]['relationsAux']:
                    unexplored_metadata.append(destID)

                self.metadata[sourceID] = self.concepts.pop(sourceID)

    def get_fsn(self, sct_id : int):
        """Method that returns the full specified name (FSN) of a concept given its ID.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        Returns:
            A string that represents the FSN. It returns an empty string if the concept is not in SNOMED.
        """
        if sct_id in self.concepts:
            return self.concepts[sct_id]['FSN']
        elif sct_id in self.metadata:
            return self.metadata[sct_id]['FSN']
        warnings.warn("Concept " + str(sct_id) + " was not found in this version of SNOMED CT.")
        return ''
    
    def get_descriptions(self, sct_id : int):
        """Method that returns the descriptions of a concept given its ID.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        Returns:
            A list of string that contains the description of the concept. It returns an empty list
            if the concept is not in SNOMED.
        """
        if sct_id in self.concepts:
            return self.concepts[sct_id]['description']
        elif sct_id in self.metadata:
            return self.metadata[sct_id]['description']
        warnings.warn("Concept " + str(sct_id) + " was not found in this version of SNOMED CT.")
        return []

    def get_semantic_type(self, sct_id : int):
        """Method that returns the semantic type, that is what appears between parenthesis in the concept's FSN.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        Returns:
            A string that represents the semantic type. It returns an empty string if the concept is not in SNOMED.
        """
        if sct_id in self.concepts:
            return self.concepts[sct_id]['semantic_type']
        elif sct_id in self.metadata:
            return self.metadata[sct_id]['semantic_type']
        warnings.warn("Concept " + str(sct_id) + " was not found in this version of SNOMED CT.")
        return ''
    
    def get_related_concepts(self, sct_id : int, filter_rels : list[int] = None):
        """Method that returns which SNOMED CT concepts are related to the concept given as a parameter. Concepts
        that are the object part of a relationship with the concept parameter are considered related concepts. If
        filter_rels is set to a list of integers containing valid relationships, only concepts that are part
        of those types of relationships will be returned.
        
        Parameters:
            sct_id (int):
                Integer that represents an ID of SNOMED CT.
            filter_rels (list):
                List that contains which relationships from SNOMED CT we are interested in.        
        
        Returns:
            A list containing tuples (relationship_ID, sct_ID) for each related concept. If there are no related concepts,
            an empty list is returned instead.
        """
        related_concepts = []
        if sct_id in self.concepts:
            related_concepts = [(rel_id, object_id) for object_id, rel_id in self.concepts[sct_id]['relations'] 
                                                    if filter_rels is None or rel_id in filter_rels]
        elif sct_id in self.metadata:
            related_concepts = [(rel_id, object_id) for object_id, rel_id in self.metadata[sct_id]['relations'] 
                                                    if filter_rels is None or rel_id in filter_rels]
        else:
            warnings.warn("Concept " + str(sct_id) + " was not found in this version of SNOMED CT.")

        return related_concepts

    def get_top_level_concept(self, sct_id : int):
        """Method that returns to which of the 19 top level hierarchies the concept belongs to.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        Returns:
            The SCT-ID of the top level concept.
        """
        if sct_id in TOP_CONCEPTS:
            return sct_id
        
        if sct_id == ROOT_CONCEPT or sct_id == METADATA_ROOT:
            return sct_id

        if sct_id in self.concepts:
            parent_id = [destID for destID, typeID in self.concepts[sct_id]['relations'] if typeID == IS_A_ID][0]
        elif sct_id in self.metadata:
            parent_id = [destID for destID, typeID in self.metadata[sct_id]['relations'] if typeID == IS_A_ID][0]
        else:
            warnings.warn("Concept " + str(sct_id) + " was not found in this version of SNOMED CT.")
            return None
        return self.get_top_level_concept(parent_id)

    def get_top_concept_list(self, sct_id : int, top_list : Iterable):
        """Method that returns to which of the concepts in the top_list the sct_id is related to in the is-a hierarchy.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
            top_list (iterable):
                Collection of SCT-IDs.
        Returns:
            A list with the SCT-ID of the concepts from the top_list to which the concept is related to. 
            If no concept in top_list is related with sct_id, an empty list is returned instead.
        """
        if sct_id == ROOT_CONCEPT or sct_id == METADATA_ROOT:
            return []

        if sct_id in top_list:
            elements_from_top_list = [sct_id]
        else:
            elements_from_top_list = []
        
        if sct_id in self.concepts:
            parent_ids = [destID for destID, typeID in self.concepts[sct_id]['relations'] if typeID == IS_A_ID]
        elif sct_id in self.metadata:
            parent_ids = [destID for destID, typeID in self.metadata[sct_id]['relations'] if typeID == IS_A_ID]
        else:
            warnings.warn("Concept " + str(sct_id) + " was not found in this version of SNOMED CT.")
            return []
            
        for parent_id in parent_ids:
            elements_from_top_list += self.get_top_concept_list(parent_id, top_list)
            
        return list(set(elements_from_top_list))

    def get_depth(self, sct_id : int):
        """Method that returns how deep a concept is in the is_a hierarchy, being the the root concept 
        138875005 | SNOMED CT Concept (SNOMED RT+CTV3) of depth 1.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        Returns:
            An integer representing the depth of the concept. If the concept is not present in SNOMED, it will
            return -1.
        """
        if sct_id == ROOT_CONCEPT:
            return 1
        
        # Identify if the concept is in the concepts or in the metadata, as we treat them separately
        hierarchy_to_search = self.concepts if sct_id in self.concepts else self.metadata if sct_id in self.metadata else None

        # If the concept is not part of SNOMED CT, we return -1
        if hierarchy_to_search is None:
            return -1
        
        # Identify its parents
        parents = [destID for destID, typeID in hierarchy_to_search[sct_id]['relations'] if typeID == IS_A_ID]
        
        # If there are no parents, it means that this concept is excluded from the hierarchy, so we return a -1
        if len(parents) == 0:
            return -1
        
        # Otherwise, we return the maximum depth of one of its parents plus one 
        return self.get_depth(parents[0]) + 1

    def get_sct_concepts(self, concepts : bool = True, metadata : bool = True):
        """Method that returns the concepts in SNOMED CT. If concepts is set to true, non-metadata concepts
        will be returned. If metadata is set to true, metadata concepts will be returned as well.
        
        Paramters:
            concepts (bool):
                Whether to return non-metadata concepts or not.
            metadata (bool):
                Whether to return metadata concepts or not.
        
        Returns:
            A list of integers representing SCT_IDs. If both concepts and metadata are set to false,
            an empty list is returned instead.
        """
        concepts_list = []

        if concepts:
            concepts_list += list(self.concepts.keys())
        
        if metadata:
            concepts_list += list(self.metadata.keys())

        return concepts_list
    
    def get_children_of(self, sct_id : int):
        """Method that returns the concepts in SNOMED CT that are children of sct_id. This method travels through the 
        whole hiearchy, so not only direct children are returned.
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        
        Returns:
            A list of integers representing SCT_IDs.
        """
        if sct_id in self.concepts:
            children_ids = self.concepts[sct_id]['relationsAux']
        elif sct_id in self.metadata:
            children_ids = self.metadata[sct_id]['relationsAux']
        else:
            warnings.warn("Concept " + str(sct_id) + " was not found in this version of SNOMED CT.")
            return []
        
        children = [sct_id]
        for children_id in children_ids:
            children += self.get_children_of(children_id)

        return list(set(children))
    
    def is_leaf_concept(self, sct_id : int):
        """Method that returns if a concept in SNOMED CT is a leaf concept or not. A leaf concept
        is one which has no child with an Is-a relationship (116680003).
        
        Parameters:
            sct_id (int):
                ID of a SNOMED CT concept.
        
        Returns:
            True if the concept is a leaf concept. Otherwise, False is returned.
        """
        if sct_id in self.concepts:
            return len(self.concepts[sct_id]['relationsAux']) == 0
        
        if sct_id in self.metadata:
            return len(self.metadata[sct_id]['relationsAux']) == 0
        
        warnings.warn("Concept " + str(sct_id) + " was not found in this version of SNOMED CT.")
        return False

    def is_child_of(self, sct_id_A : int, sct_id_B : int):
        """Method that returns if concept denoted by sct_id_A is a child of the concept denoted by sct_id_B. This relation does not have to be direct, as the
        method only checks if sct_id_B is part of any of the concepts in the is_a hierarchy of sct_id_A created by going towards the root concept.
        
        Parameters:
            sct_id_A (int):
                ID of the SNOMED CT concept that might be the child.
            sct_id_B (int):
                ID of the SNOMED CT concept that might be the parent.
        
        Returns:
            True if sct_id_A is child of sct_id_B. False otherwise.
        """
        if sct_id_A in self.concepts:
            parent_ids = [destID for destID, typeID in self.concepts[sct_id_A]['relations'] if typeID == IS_A_ID]
        elif sct_id_A in self.metadata:
            parent_ids = [destID for destID, typeID in self.metadata[sct_id_A]['relations'] if typeID == IS_A_ID]
        else:
            return False
        
        if sct_id_B in parent_ids:
            return True
        
        for parent_id in parent_ids:
            if self.is_child_of(parent_id, sct_id_B):
                return True
        
        return False

    def is_parent_of(self, sct_id_A : int, sct_id_B : int):
        """Method that returns if the concept sct_id_A is a parent of concept sct_id_B. This relation does not have to be 
        direct, as it checks for each ascendant of sct_id_B.

        Parameters:
            sct_id_A (int):
                ID of the SNOMED CT concept that might be the parent.
            sct_id_B (int):
                ID of the SNOMED CT concept that might be the child.

        Returns:
            True if sct_id_A is a parent of sct_id_B. Otherwise returns False.
        """
        # A is a parent of B if B is a child of A
        return self.is_child_of(sct_id_B, sct_id_A)    

    def path_between_concepts(self, sct_id_A : int, sct_id_B : int):
        """Method that returns the shortest path between concepts sct_id_A and sct_id_B. If any of them is not present in SNOMED CT,
        None is returned instead.
        
        Parameters:
            sct_id_A (int):
                ID of a SNOMED CT concept.
            sct_id_B (int):
                ID of the target SNOMED CT concept.
        
        Returns:
            A list with the concepts that conform the path from sct_id_A to sct_id_B, both included. If any is not present in
            SNOMED CT, None is returned instead.
        """
        if sct_id_A not in self.concepts:
            warnings.warn('Concept ' + str(sct_id_A) + ' was not found in SNOMED CT.')
            return None
        
        if sct_id_B not in self.concepts:
            warnings.warn('Concept ' + str(sct_id_B) + ' was not found in SNOMED CT.')
            return None

        if sct_id_A == sct_id_B:
            return [sct_id_A]
        
        # Visited is used to prevent loops
        visited = set()
        # To visit contains tuples of (sct_id, path_to_sct_id)
        to_visit = deque([(sct_id_A, [sct_id_A])])

        while len(to_visit) > 0:
            current_sct_id, path = to_visit.popleft()
            visited.add(current_sct_id)

            children = [sct_id for sct_id in self.concepts[current_sct_id]['relationsAux'] if sct_id in self.concepts] # This is to prevent getting into the metadata part of SNOMED CT if we get to the root
            parents = [sct_id for _, sct_id in self.get_related_concepts(current_sct_id, filter_rels=[IS_A_ID]) if sct_id in self.concepts] 
            neighbours = children + parents

            for neighbour_id in neighbours:
                if neighbour_id == sct_id_B:
                    return path + [neighbour_id]

                if neighbour_id not in visited:
                    to_visit.append((neighbour_id, path + [neighbour_id]))
                    visited.add(neighbour_id)

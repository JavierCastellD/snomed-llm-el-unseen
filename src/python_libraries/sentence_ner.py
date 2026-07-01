class Entity:
    """Class wrapper for an entity represented by the start and end index, as well as the corresponding text from that span.
    
    Attributes:
        text (str):
            Text that the entity represents.

        start (int):
            Start index position for the entity.

        end (int):
            End index position for the entity.

        ner_type (str):
            String that denotes the type of entity.

        label (str):
            Label associated with the entity.

        options (list):
            List of possible labels.

        confidence (float):
            Float number between 0 and 1 that denotes how confident is the prediction.

        other (dictionary):
            Used to store other information that can be used for debug purposes.
    """
    def __init__(self, text : str, start : int, end : int, ner_type : str = None, label : str = None, options : list = [], confidence : float = None, other : dict = {}):
        self.text = text
        self.start = start
        self.end = end
        self.ner_type = ner_type
        self.label = label
        self.options = options
        self.confidence = confidence
        self.other = other

    @staticmethod
    def from_dictionary(prediction : dict[str], key_maps : dict[str] = {'start' : 'start', 'end' : 'end', 'text' : 'text'}):
        """Method that transforms a prediction in dictionary form to an Entity object given a map for the keys.
        
        Parameters:
            prediction (dict[str]):
                Dictionary with the predictions.
            key_maps (dict[str]):
                Dictionary that maps the attributes of an Entity to the keys of the dictionary. It should at least have: text, start, and end.
        
        Returns:
            An Entity with the values from the prediction.
        """
        # Extract the relevant parts
        text = prediction[key_maps['text']]
        start = prediction[key_maps['start']]
        end = prediction[key_maps['end']]
        ner_type = prediction[key_maps['ner_type']] if 'ner_type' in key_maps else None
        label = prediction[key_maps['label']] if 'label' in key_maps else None
        options = prediction[key_maps['options']] if 'options' in key_maps else []
        confidence = prediction[key_maps['confidence']] if 'confidence' in key_maps else None
        other = prediction[key_maps['other']] if 'other' in key_maps else {}

        return Entity(text=text, start=start, end=end, ner_type=ner_type, label=label, options=options, confidence=confidence, other=other)

    @staticmethod
    def from_transformers_pipeline(prediction : dict[str]):
        """Method that transforms a prediction from the transformers NER pipeline into an Entity.
        
        Parameters:
            prediction (dict[str]):
                Dictionary with keys: entity_group, score, start, end, word.
        
        Returns:
            An Entity with the values from the prediction.
        """
        # Create the mapping for the prediction dictionary from the transformers pipeline
        map_dictionary = {'text' : 'word',
                          'start' : 'start',
                          'end' : 'end',
                          'ner_type' : 'entity_group'}

        return Entity.from_dictionary(prediction, map_dictionary)
    
    def set_label(self, label : str):
        """Method that sets the label of the entity.
        
        Parameters:
            label (str):
                Label to be assigned.
        """
        self.label = label

    def set_options(self, options : list):
        """Method that sets the options of the entity.
        
        Parameters:
            options (str):
                Options to be assigned.
        """
        self.options = options

    def set_confidence(self, confidence : float):
        """Method that sets the confidence of the entity.
        
        Parameters:
            confidence (float):
                Confidence to be assigned.
        """
        self.confidence = confidence

    def offset_entity(self, offset : int):
        """Method that moves the Entity's start and end an amount of characters
        denoted by offset.
        
        Parameters:
            offset (int):
                Number of characters to move the Entity.
        """
        self.start += offset
        self.end += offset

    def __str__(self):
        """Method to print out Entity objects."""
        return f"{{Text: '{self.text}', Index: {self.start}:{self.end}, Label: {self.label}, NER type: {self.ner_type}}}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.text == other.text and self.start == other.start and self.label == other.label and self.ner_type == other.ner_type
    
    def __hash__(self):
        return hash((self.text, self.start, self.end, self.label, self.ner_type))

class Sentence:
    """Class wrapper for a sentence and information about it like the section or the entities found in it.

    Attributes:
        text (str):
            Text of the sentence.

        section (str):
            Section to which the sentence corresponds to.

        entities (list[Entity]):
            List of entities found in the sentence.

        original_start (int):
            True offset of this sentence's text within the original, unmodified document text.
            None if unknown (e.g. not computed by the dataset loader).
    """
    def __init__(self, text : str, section : str = "", entities : list[Entity] = [], original_start : int = None):
        self.text = text
        self.section = section
        self.entities = entities
        self.original_start = original_start

    def add_entities(self, entities : list[Entity]):
        """Method to add entities to the list of entities found in the sentence.
        
        Parameters:
            entities (list[Entity]):
                List of entities to add.
        """
        for entity in entities:
            self.entities.append(entity)

    def length(self):
        """Method that returns the length of the sentence's text.
        
        Returns:
            An integer representing the length of the sentence.
        """
        return len(self.text)
    
    def offset_entities(self, offset : int):
        """Method that moves the entities' start and end of the Sentence an amount of characters
        denoted by offset.
        
        Parameters:
            offset (int):
                Number of characters to move each Entity in the Sentence.
        """
        for entity in self.entities:
            entity.offset_entity(offset)
    
    def entities_to_dicts(self, key_list : list[str] = None) -> list[dict]:
        """Method that returns the entities as a list of dictionaries. The keys of each dictionary
        are all the attributes of the Entity class unless key_list is defined.
        
        Parameters:
            key_list (list[str]):
                List with the relevant attributes from Entity to return.
        
        Returns:
            A list with the entities transformed to dictionaries.
        """
        # If no list of keys was given, we return a list of dictionaries for
        # all attributes from the entity
        if key_list is None:
            return [entity.__dict__ for entity in self.entities]
        # Otherwise, only those keys listed are returned
        else:
            entities = []
            for entity in self.entities:
                entity_d = entity.__dict__
                entities.append({key : entity_d[key] for key in key_list if key in entity_d})
            
            return entities
        
    def __str__(self):
        """Method to print out Sentence objects."""
        return f"{{Sentence: '{self.text}', Section: {self.section}, Entities: {self.entities}}}"
    
    def __repr__(self):
        return self.__str__()
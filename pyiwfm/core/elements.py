import os
import numpy as np
import pandas as pd

class Element:
    ''' Defines a Model Element Object. This is a base class
    and has no other knowledge of other Element objects defined
    in a model application

    Attributes
    ----------
    element_id : int
        unique identifier for element

    node_ids : list, tuple, np.array
        iterable of 4 node_ids denoting the vertices of an element.
        for triangular elements, node_ids[3] = 0

    subregion : int
        identifier for subregion the element is assigned

    Methods
    -------
    from_string : classmethod
        creates an Element object from a string containing 6 values
    '''
    def __init__(self, element_id, node_ids, subregion):

        # check that element_id is an integer
        if not isinstance(element_id, int):
            raise TypeError("element_id must be an integer")

        # check that node_ids are a list, tuple, or array of integers
        if not isinstance(node_ids, (list, tuple, np.array)):
            raise TypeError("node_ids must be provided as a list, tuple, or np.array")
        else:
            # length must be equal to 4
            assert len(node_ids) == 4

            # check that all values in node_ids are integers
            if not all([isinstance(val, int) for val in node_ids]):
                raise TypeError("each node id in node_ids must be an integer")

        # check that subregions is an integer
        if not isinstance(subregion, int):
            raise TypeError("subregion must be an integer")

        self.element_id = element_id
        
        if isinstance(node_ids, np.array):
            self.node_ids = node_ids
        else:
            self.node_ids = np.array(node_ids)

        self.subregion = subregion

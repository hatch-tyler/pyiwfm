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

            # check if a 0 is provided as one of the ids in node_ids, 
            # there must only be 1 it must be the last value node_id
            if isinstance(node_ids, np.array):
                zero_index = np.where(node_ids, 0)[0]
            else:
                zero_index = [i for i in range(len(node_ids)) if node_ids[i] == 0]
                
            if len(zero_index) > 1:
                raise ValueError("{} node ids were provided equal to 0. There can only be one".format(len(zero_index)))
            elif len(zero_index) == 1 and zero_index[0] != 3:
                raise IndexError("A 0 can only be provided as the last value of node_ids")

        # check that subregions is an integer
        if not isinstance(subregion, int):
            raise TypeError("subregion must be an integer")

        self.element_id = element_id
        
        if isinstance(node_ids, np.array):
            self.node_ids = node_ids
        else:
            self.node_ids = np.array(node_ids)

        self.subregion = subregion

    def __repr__(self):
        return 'Element(element_id={}, node_ids={}, subregion={})'.format(self.element_id, self.node_ids, self.subregion)

    @classmethod
    def from_string(cls, string):
        ''' alternate class constructor designed to be used to read
        from a text file 
        '''
        if not isinstance(string, str):
            raise TypeError("value provided must be a string type. type provided: {}".format(type(string)))

        string_list = string.split()

        # check list has 6 items i.e. element_id, node1, node2, node3, node4, subregion
        if len(string_list) != 6:
            raise ValueError("string must include exactly 6 values for a groundwater node")

        element_id = int(string_list[0])
        node_ids = np.array([int(val) for val in string_list[1:6]])
        subregion = int(string_list[6])

        return cls(element_id, node_ids, subregion)


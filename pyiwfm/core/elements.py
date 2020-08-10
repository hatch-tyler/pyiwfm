import os
import numpy as np
import pandas as pd

class IWFMElements:
    ''' Defines IWFM Elements object. This class is composed
    of many Elements that exist within a model application

    Attributes
    ----------
    ne : int
        number of Element objects in the model application

    nregn : int
        number of subregions in the model application

    rnames : list of str
        names defined for each of the nregn subregions

    elements : list
        list of Element Object instances

    Methods
    -------
    get_subregion_from_id : instance method
        returns the subregion integer id for a given element_id

    get_nodes_from_id : instance method
        returns the array of node_ids for a given element_id

    from_file : class method
        creates an IWFMElements object from the IWFM element configuration file

    '''
    def __init__(self, ne, nregn, rnames, elements):
        # check that the number of elements is an integer
        if not isinstance(ne, int):
            raise TypeError("the number of elements, ne, must be an integer")

        self.ne = ne

        # check that the number of subregions is an integer
        if not isinstance(nregn, int):
            raise TypeError("the number of subregions, nregn, must be an integer")

        self.nregn = nregn

        # check that the rnames is a list of strings
        if not isinstance(rnames, list) and all([isinstance(val, str) for val in rnames]):
            raise TypeError("rnames must be a list of strings")

        # check that length of rnames is equal to nregn
        if len(rnames) != nregn:
            raise ValueError("There must be {} subregion names in rnames. {} provided".format(nregn, len(rnames)))

        self.rnames = rnames
                
        # check that elements is a list of Element objects
        if not isinstance(elements, (list, tuple)) and all([isinstance(element, Element) for element in elements]):
            raise TypeError("elements must be a list or tuple of Element objects")

        # check that length of elements is equal to ne
        if len(elements) != ne:
            raise ValueError("There must be {} elements. {} provided.".format(ne, len(elements)))

        self.elements = elements

    def get_subregion_from_id(self, element_id):
        ''' returns the subregion id for a given element_id

        Parameters
        ----------
        element_id : int
            element_id with a corresponding subregion id

        Returns
        -------
        int
            subregion id for Element object
        '''
        if not isinstance(element_id, int):
            raise TypeError("id must be an integer. value provided is a {}".format(type(element_id)))

        element_ids = [element.element_id for element in self.elements]
        
        if element_id not in element_ids:
            raise ValueError("element_id provided is not a valid element_id")

        return [element.subregion for element in self.elements if element_id == element.element_id][0]

    def get_subregion_name_from_id(self, element_id):
        ''' returns the subregion name for the subregion associated 
        with a given element_id '''
        if not isinstance(element_id, int):
            raise TypeError("id must be an integer. value provided is a {}".format(type(element_id)))

        element_ids = [element.element_id for element in self.elements]
        
        if element_id not in element_ids:
            raise ValueError("element_id provided is not a valid element_id")

        subregion_id = [element.subregion for element in self.elements if element_id == element.element_id][0]

        return self.rnames[subregion_id - 1]

    def get_nodes_from_id(self, element_id):
        ''' returns the array of node_ids for a given element_id

        Parameters
        ----------
        element_id : int
            element_id with a corresponding set of node_ids

        Returns
        -------
        np.array
            integer array of node_ids for given element_id
        '''
        if not isinstance(element_id, int):
            raise TypeError("id must be an integer. value provided is a {}".format(type(element_id)))

        element_ids = [element.element_id for element in self.elements]
        
        if element_id not in element_ids:
            raise ValueError("element_id provided is not a valid element_id")

        return [element.node_ids for element in self.elements if element_id == element.element_id][0]

    def to_dict(self):
        ''' converts the list of Element objects to a python dictionary '''
        element_ids = np.array([element.element_id for element in self.elements])
        node_array = np.array([element.node_ids for element in self.elements])
        node1 = node_array[:,0]
        node2 = node_array[:,1]
        node3 = node_array[:,2]
        node4 = node_array[:,3]
        subregion = np.array([element.subregion for element in self.elements])

        return dict(element_id=element_ids,
                    node1=node1,
                    node2=node2,
                    node3=node3,
                    node4=node4,
                    subregion=subregion)

    def to_dataframe(self, filter_zeros=True):
        '''converts the list of Element objects to a pandas DataFrame object 
        Parameters
        ----------
        filter_zeros : bool, default=True
            flag to remove node_ids equal to 0
            
        Returns
        -------
        pd.DataFrame
            pandas DataFrame object of Element data 
        '''
        element_ids = np.array([element.element_id for element in self.elements])
        node_array = np.array([element.node_ids for element in self.elements])
        subregions = np.array([element.subregion for element in self.elements])

        num_elem, num_nodes = node_array.shape

        elements_dict = dict(ElementID=np.repeat(element_ids, num_nodes),
                             NodeNum=np.tile(np.arange(1, num_nodes + 1), num_elem),
                             NodeID=node_array.flatten(),
                             Subregion=np.repeat(subregions, num_nodes))
        
        df = pd.DataFrame(elements_dict)

        if filter_zeros:
            return df[df['NodeID'] != 0]
        else:
            return df

    @classmethod
    def from_file(cls, elements_file):
        ''' alternate class constructor read from a text file 
        
        Parameters
        ----------
        elements_file : str
            file path and name for the IWFM elements file

        Returns
        -------
        instance of IWFMElements class
        '''
        if not isinstance(elements_file, str):
            raise TypeError("elements_file must be a string")

        with open(elements_file, 'r') as f:
            count = 0
            for line in f:
                if line[0] not in ['C', 'c', '*']:
                    if count == 0:
                        ne = int(line.split('/')[0].strip())
                    elif count == 1:
                        nregn = int(line.split('/')[0].strip())
                    count += 1
                    if count == 2:
                        break

            rnames=[]
            count = 0
            for line in f:
                if line[0] not in ['C', 'c', '*']:
                    rnames.append(line.split('/')[0].strip())
                    count += 1
                    if count == nregn:
                        break

            elements = []
            for line in f:
                if line[0] not in ['C', 'c', '*']:
                    elements.append(Element.from_string(line))

        return cls(ne, nregn, rnames, elements)

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
        if not isinstance(node_ids, (list, tuple, np.ndarray)):
            raise TypeError("node_ids must be provided as a list, tuple, or np.array")
        else:
            # length must be equal to 4
            assert len(node_ids) == 4

            # check that all values in node_ids are integers
            if not all([isinstance(val, (int, np.int32, np.int64)) for val in node_ids]):
                raise TypeError("each node id in node_ids must be an integer: {}".format(node_ids))

            # check if a 0 is provided as one of the ids in node_ids, 
            # there must only be 1 it must be the last value node_id
            if isinstance(node_ids, np.ndarray):
                zero_index = np.where(node_ids == 0)[0]
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
        
        if isinstance(node_ids, np.ndarray):
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
        node_ids = np.array([int(val) for val in string_list[1:5]])
        subregion = int(string_list[5])

        return cls(element_id, node_ids, subregion)


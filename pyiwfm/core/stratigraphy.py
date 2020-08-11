import os
import numpy as np
import pandas as pd

class IWFMStratigraphy:
    ''' Defines the IWFM Stratigraphy object. This class contains the 
    layering information 
    
    Attributes
    ----------
    nl : int
        number of layers in the model application

    fact : float
        conversion factor for elevations and thicknesses in the 
        stratigraphic data

    stratigraphy : list
        list of NodeStratigraphy objects
    
    Methods
    -------
    get_ground_surface_elevation : instance method
        returns a numpy array of the ground surface elevations at each GroundwaterNode

    to_dataframe : instance method
        converts the stratigraphy data into a pandas DataFrame
    '''
    def __init__(self, nl, fact, stratigraphy):

        if not isinstance(nl, int):
            raise TypeError("nl must be an integer")

        self.nl = nl

        if not isinstance(fact, (int, float)):
            raise TypeError("fact must be an number")

        self.fact = fact

        # check that stratigraphy is a list and all items in the list are NodeStratigraphy objects
        if not isinstance(stratigraphy, list) and all([isinstance(item, NodeStratigraphy) for item in stratigraphy]):
            raise TypeError("stratigraphy must be a list of NodeStratigraphy objects")

        # check that the layer thickness arrays are all equal to 2 times the number of layers
        layer_check = np.where(np.array([len(ns.layer_thicknesses) for ns in stratigraphy]) != 2 * self.nl)[0]
        
        if len(layer_check) > 0:
            raise ValueError("stratigraphy provided does not match the number of layers provided")

        self.stratigraphy = stratigraphy

    def get_ground_surface_elevations(self):
        ''' returns all ground surface elevations provided in the IWFM Stratigraphy File '''
        return np.array([ns.gse for ns in self.stratigraphy])

    def to_dataframe(self):
        ''' converts the list of NodeStratigraphy objects to a pandas DataFrame '''
        names = IWFMStratigraphy._get_stratigraphy_column_names(self.nl)

        node_ids = np.array([ns.node_id for ns in self.stratigraphy])
        gse = np.array([ns.gse for ns in self.stratigraphy])
        layer_thicknesses = np.array([ns.layer_thicknesses for ns in self.stratigraphy])

        df = pd.concat([pd.DataFrame(node_ids),pd.DataFrame(gse), pd.DataFrame(layer_thicknesses)], axis=1)
        df.columns = names

        return df

    @staticmethod
    def _get_stratigraphy_column_names(num_layers):
        ''' private static method to generate variable names for aquitard
        and aquifer layers 
        
        Parameters
        ----------
        num_layers : int
            number of layers in the model
            
        Returns
        -------
        list
            names for data columns provided in IWFM Stratigraphy File 
        '''
        names = ['NodeID', 'GSE']

        if not isinstance(num_layers, int):
            raise TypeError("num_layers must be an integer")

        for i in range(num_layers):
            names.append('A{}'.format(i+1))
            names.append('L{}'.format(i+1))

        return names

    @classmethod
    def from_file(cls, stratigraphy_file):
        ''' alternate class constructor read from a text file 
        
        Parameters
        ----------
        stratigraphy_file : str
            file path and name for the IWFM Stratigraphy file

        Returns
        -------
        instance of IWFMStratigraphy class
        '''
        if not isinstance(stratigraphy_file, str):
            raise TypeError("stratigraphy_file must be a string")

        with open(stratigraphy_file, 'r') as f:
            count = 0
            for line in f:
                if line[0] not in ['C', 'c', '*']:
                    if count == 0:
                        nl = int(line.split('/')[0].strip())
                    elif count == 1:
                        fact = float(line.split('/')[0].strip())
                    
                    count += 1
                    
                    if count == 2:
                        break
            
            stratigraphy = []
            for line in f:
                if line[0] not in ['C', 'c', '*']:
                    
                    # handle case where a blank line exists
                    if len(line.split()) == 0:
                        continue
                    
                    stratigraphy.append(NodeStratigraphy.from_string(line))

        return cls(nl=nl, fact=fact, stratigraphy=stratigraphy)

class NodeStratigraphy:
    ''' Base class defining the stratigraphy of the model at a single GroundwaterNode

    Attributes
    ----------
    node_id : int
        node_id of a GroundwaterNode

    gse : float
        ground surface elevation at GroundwaterNode

    layer_thicknesses : list, tuple, np.ndarray
        thicknesses of each layer and aquitard

    Methods
    -------

    '''
    def __init__(self, node_id, gse, layer_thicknesses):
        
        if not isinstance(node_id, int):
            raise TypeError("node_id must be an integer")

        self.node_id = node_id

        if not isinstance(gse, float):
            raise TypeError("gse must be a float")

        self.gse = gse

        if not isinstance(layer_thicknesses, (list, tuple, np.ndarray)):
            raise TypeError("layer_thicknesses must be a list, tuple, or np.ndarray")
        
        # assume values provides are numbers and able to be type converted to floats
        # will raise a ValueError if not numeric
        layer_thicknesses = np.array(layer_thicknesses, dtype=np.float)

        self.layer_thicknesses = layer_thicknesses

    def __repr__(self):
        return 'NodeStratigraphy(node_id={}, gse={}, layer_thicknesses={})'.format(self.node_id, self.gse, self.layer_thicknesses)

    @classmethod
    def from_string(cls, string):
        ''' alternate class constructor designed to be used to read
        from a text file 
        '''
        if not isinstance(string, str):
            raise TypeError("value provided must be a string type. type provided: {}".format(type(string)))

        string_list = string.split()

        len_string_list = len(string_list)

        if len_string_list < 4:
            raise IndexError("string provided must have at least 4 values")

        node_id = int(string_list[0])
        gse = float(string_list[1])
        layer_thicknesses = np.array(string_list[2:], dtype=np.float)

        return cls(node_id, gse, layer_thicknesses)

    





        


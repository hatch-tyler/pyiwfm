class IWFMNodes:
    ''' defines the IWFM Nodes object. This class is composed of many
    GroundwaterNodes that exist within a model application. 

    Attributes
    ----------
    nd : int
        number of GroundwaterNode objects in the model application

    fact : float
        length conversion factor for the x- and y- coordinates

    nodes : list
        list of GroundwaterNode objects

    Methods
    -------
    from_file : classmethod
        creates an IWFMNodes object from the IWFM nodal coordinate file
    '''
    def __init__(self, nd, fact, nodes):
        # check that the number of nodes variable nd is an integer
        if not isinstance(nd, int):
            raise TypeError("nd must be an integer")

        self.nd = nd

        # check that the conversion factor variable is a number
        if not isinstance(fact, (int, float)):
            raise TypeError("fact must be a number")

        self.fact = fact

        # check that nodes is a list of GroundwaterNode objects
        if not isinstance(nodes, (list, tuple)) and all([isinstance(node, GroundwaterNode) for node in nodes]):
            raise TypeError("nodes must be a list or tuple of GroundwaterNode objects")

        if len(nodes) != nd:
            raise ValueError("there must be {} nodes. {} were provided".format(nd, len(nodes)))

        self.nodes = nodes
    
    @classmethod
    def from_file(cls, nodes_file):
        ''' alternate class constructor read from a text file '''

        if isinstance(nodes_file, str):
            with open(nodes_file, 'r') as f:
                count = 0
                for line in f:
                    if line[0] not in ['C', 'c', '*']:
                        if count == 0:
                            nd = int(line.split('/')[0].strip())
                        elif count == 1:
                            fact = float(line.split('/'[0].strip()))
                        count += 1
                    if count == 2:
                        break

                nodes = []
                for line in f:
                    if line[0] not in ['C', 'c', '*']:
                        nodes.append(GroundwaterNode.from_string(line))

        return cls(nd, fact, nodes)

class GroundwaterNode:
    ''' defines the groundwater node object. This class is a base
    class and has no knowledge of other node objects defined in a
    model application.
    
    Attributes
    ----------
    node_id : int
        unique identifier for node
        
    x : float
        x-coordinate for node location
        
    y : float
        y-coordinate for node location
        
    Methods
    -------
    from_string : classmethod
        creates a GroundwaterNode object from a string containing 3 values
    '''
    def __init__(self, node_id, x, y):

        # check that node_id is an integer
        if not isinstance(node_id, int):
            raise TypeError("Node ID must be an integer")

        # check that x is an int or float
        if not isinstance(x, (int, float)):
            raise TypeError("x-coordinate must be a number")

        # check that y is an int or float
        if not isinstance(y, (int, float)):
            raise TypeError("y-coordinate must be a number")

        self.node_id = node_id
        self.x = x
        self.y = y

    @classmethod
    def from_string(cls, string):
        ''' alternate class constructor designed to be used to read
        from a text file 
        '''
        # check that string is a string
        if not isinstance(string, str):
            raise TypeError("value provided must be a string type. type provided: {}".format(type(string)))

        string_list = string.split()

        # check list has 3 items i.e. node_id, x, y
        if len(string_list) != 3:
            raise ValueError("string must include exactly 3 values for a groundwater node")

        node_id = int(string_list[0])
        x = float(string_list[1])
        y = float(string_list[2])

        return cls(node_id, x, y)




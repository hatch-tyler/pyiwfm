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




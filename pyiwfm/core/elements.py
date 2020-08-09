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
    def __init__(self, element_id, node_ids, subregions):

        # check that element_id is an integer
        if not isinstance(element_id, int):
            raise TypeError("element_id must be an integer")

        # check that node_ids are a list, tuple, or array
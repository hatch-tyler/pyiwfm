'''
generic_linked_list.py
Author: Tyler Hatch PhD, PE

This is the generic_linked_list module of the python version of IWFM

Notes
-----
GenericLinkedListType in IWFM is an abstract class. However, this is somewhat
different than in python where an abstract class in Fortran has attributes and
methods with functionality, a python abstract class is just a template with 
method names (more like the Type declaration in Fortran) that must be defined 
in the class that inherits from it.

Here it appears as a base class but with methods intact to inherit similarly 
to the Fortran implementation
'''
import numpy as np

# local imports
from message_logger import set_last_message
from message_logger import FATAL
from linked_list_node import LinkedListNode

MODULE_NAME = 'GenericLinkedList::'

class GenericLinkedList:
    ''' This is a base class for a Linked List

    Attributes
    ----------
    __n_nodes : int, private
        number of nodes in linked list

    __head : LinkedListNode
        first node in the linked list

    __tail : LinkedListNode
        last node in the linked list

    __current : LinkedListNode
        current node in the linked list

    Methods
    -------
    get_n_nodes : instance method
        returns the number of nodes in the linked list

    get_current_value : instance method
        returns the current value in the linked list

    convert_to_integer_array : instance method
        converts the linked list to an integer numpy ndarray

    add_node : instance method
        adds a node to the end of the linked list

    reset : instance method
        sets the current value to the head value

    next_node : instance method
        moves the current value to the next node in the linked list

    Usage
    -----
    >>> linked_list = GenericLinkedList()
    >>> for i in range(1,10):
    ...     linked_list.add_node(i)

    >>>print(linked_list.get_n_nodes())
        9

    >>>print(linked_list.convert_to_integer_array())
        [1 2 3 4 5 6 7 8 9]    
    '''
    def __init__(self, n_nodes=None, head=None, tail=None, current=None):
        if n_nodes is None:
            self._n_nodes = 0
        elif not isinstance(n_nodes, int):
            raise TypeError('n_nodes must be an integer')

        if head is None:
            self._head = LinkedListNode(None)
        elif isinstance(head, LinkedListNode):
            self._head = head
        else:
            raise TypeError('head must be of type LinkedListNode')

        if tail is None:
            self._tail = LinkedListNode(None)
        elif isinstance(tail, LinkedListNode):
            self._tail = tail
        else:
            raise TypeError('tail must be of type LinkedListNode')

        if current is None:
            self._current = LinkedListNode(None)
        elif isinstance(current, LinkedListNode):
            self._current = current
        else:
            raise TypeError('current must be of type LinkedListNode')

    def __del__(self):
        # move to the head of the list
        self.reset()

        for _ in range(self._n_nodes):
            current = self._current
            self.next_node()
            del(current)

        self._n_nodes = 0

    def __repr__(self):
        node = self._head
        nodes = []
        while node is not None:
            nodes.append(node.get_value())
            node = node.get_next()
        nodes.append("None")
        
        return " -> ".join([str(i) for i in nodes])

    def get_n_nodes(self):
        return self._n_nodes

    def get_current_value(self):
        return self._current.get_value()

    def convert_to_integer_array(self):
        if self._n_nodes == 0:
            return
        
        this_procedure = MODULE_NAME+"convert_to_integer_array"

        # move to the head of the list
        self.reset()

        out_list = []
        if not out_list:
            set_last_message('Error in allocating memory to convert a linked list to an integer array.', FATAL, this_procedure)
        
        for _ in range(self._n_nodes):
            current = self.get_current_value()
            if isinstance(current, int):
                out_list.append(current)
            self.next_node()

        return np.array(out_list, dtype=np.int)

    def add_node(self, value):
        if self._n_nodes == 0:
           self._head = LinkedListNode(value)
           self._tail = self._head

        else:
            new_node = LinkedListNode(value)
            self._tail.set_next(new_node)
            self._tail = new_node

        self._current = self._tail
        self._n_nodes += 1

    def reset(self):
        self._current = self._head

    def next_node(self):
        self._current = self._current.get_next()

if __name__ == '__main__':
    node_list = GenericLinkedList()
    print(node_list.get_n_nodes())

    for i in range(1, 10):
        node_list.add_node(i)

    print(node_list)

    node_list.reset()
    print(node_list.get_current_value())

    print(node_list.get_n_nodes())
    print(node_list.convert_to_integer_array())
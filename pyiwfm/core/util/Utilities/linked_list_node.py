'''
linked_list_node.py
Author: Tyler Hatch PhD, PE

This is the linked_list_node  module of the python version of IWFM
'''
# local imports
from message_logger import set_last_message
from message_logger import FATAL

MODULE_NAME = "LLNode::"

class LinkedListNode:
    def __init__(self, value):
        self.__value = value
        self.__next_value = None

    def __del__(self):
        pass

    def __repr__(self):
        return "LinkedListNode(value={}, next_value={})".format(self.__value, self.__next_value)

    def get_next(self):
        return self.__next_value

    def get_value(self):
        return self.__value

    def set_next(self, node):
        this_procedure = MODULE_NAME+'set_next'
        
        if not isinstance(node, LinkedListNode):
            raise TypeError("node must be of type LinkedListNode")

        if self.__next_value is not None:
            set_last_message("Can only add a node to the end of the linked list.", FATAL, this_procedure)
        
        self.__next_value = node

if __name__ == '__main__':
    n1 = LinkedListNode("Dec")
    n2 = LinkedListNode("Jan")
    print(n1.get_value())
    n1.set_next(n2)
    print(n1.get_next())
    n2.set_next("Feb")


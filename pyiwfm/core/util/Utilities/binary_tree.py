'''
binary_tree.py
Author: Tyler Hatch PhD, PE

This is the binary_tree module of the python version of IWFM
'''

class BinaryTree:
    def __init__(self, key, data, left, right):
        self.key = key
        self.data = data
        self.left = left
        self.right = right
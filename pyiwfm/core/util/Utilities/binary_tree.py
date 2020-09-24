'''
binary_tree.py
Author: Tyler Hatch PhD, PE

This is the binary_tree module of the python version of IWFM
'''
MODULE_NAME = 'Class_BinaryTree::'

class BinaryTree:
    def __init__(self, key=None, data=None):
        self._key = key
        self._data = data
        self._left = None
        self._right = None

    def __del__(self):
        pass

    def __repr__(self):
        return "BinaryTree(key={}, data={})".format(self._key, self._data)
    
    def add_node(self, key, data):
        if self._data:
            if key > self._key:
                if self._right is None:
                    self._right = BinaryTree(key, data)
                else:
                    self._right.add_node(key, data)
                
            elif key < self._key:
                if self._left is None:
                    self._left = BinaryTree(key, data)
                else:
                    self._left.add_node(key, data)

        else:
            self._key = key
            self._data = data


    def get_n_nodes(self):
        ''' counts the number of nodes in the Binary Tree '''
        count = 0

        if self._key:
            count = self._get_n_nodes_recursive()

        return count


    def get_pointer_to_node(self, key):
        ''' returns the data for a given key '''
        if self._key:
            if key == self._key:
                return self._data

            elif key > self._key:
                return self._right.get_pointer_to_node(key)
            
            else:
                return self._left.get_pointer_to_node(key)


    def get_ordered_key_list(self):
        ''' returns an ordered list of the keys in the Binary Tree '''
        ordered_list = []

        return self._get_ordered_keys_recursive(ordered_list)


    def _get_n_nodes_recursive(self):
        count = 1        
        
        if self._right:
            count += self._right._get_n_nodes_recursive()
                        
        if self._left:
            count += self._left._get_n_nodes_recursive()
        
        return count


    def _get_ordered_keys_recursive(self, ordered_list):
        if self._left:
            self._left._get_ordered_keys_recursive(ordered_list)

        ordered_list.append(self._key)

        if self._right:
            self._right._get_ordered_keys_recursive(ordered_list)

        return ordered_list

if __name__ == '__main__':
    t = BinaryTree()

    t.add_node(5, {'zone': 5, 'elements': [1,2,3,4,5]})
    t.add_node(2, {'zone': 2, 'elements': [6,7,8,9,10]})
    t.add_node(3, {'zone': 3, 'elements': [11,12,13,14,15]})
    t.add_node(10, {'zone': 10, 'elements': [23,24,25,26,27,28,29,30]})

    print(t.get_n_nodes())

    print(t.get_pointer_to_node(3))
    
    print(t.get_ordered_key_list())

    print(t)



    
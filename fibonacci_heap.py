import math
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.parent = None
        self.child = None
        self.left = None
        self.right = None
        self.degree = 0

class FibonacciHeap:
    def __init__(self):
        self.min_pointer = None
        self.root = None
        self.length = 0

    def traverse(self, root):
        temp = []
        node  = root
        end = root
        flag = False
        valid = True
        while valid:
            if node == end and flag is True:
                valid = False
            elif node == end:
                flag = True
            if valid == True:
                temp.append(node)
                node = node.right
        return temp

    def extract_min(self):
        pointer = self.min_pointer
        if pointer is not None:
            if pointer.child is not None:
                child_list = self.traverse(pointer.child)
                # merge the children nodes with the root
                for i in range(len(child_list)):
                    item = child_list[i]
                    if self.root is None:
                        self.root  = item
                    else:
                        item.right = self.root.right
                        item.left = self.root 
                        self.root.right.left = item
                        self.root.right = item
                    child_list[i].parent = None
            if pointer == self.root:
                self.root = pointer.right
            pointer.left.right = pointer.right
            pointer.right.left = pointer.left
            
            # assign new minimum node
            if pointer == pointer.right:
                self.min_pointer = None
                self.root = None
            else:
                self.min_pointer = pointer.right
                self.consolidate()
            self.length -= 1
        return pointer
    
    def insert(self, key, value=None):
        item = Node(key, value)
        item.left = item
        item.right = item
        if self.root is None:
            self.root = item
            self.min_pointer = item
        else:
            item.left = self.root
            item.right = self.root.right 
            self.root.right.left = item
            self.root.right = item
            if self.min_pointer is None or item.key < self.min_pointer.key:
                self.min_pointer = item
        self.length += 1

    def consolidate(self):
        # construct a degree array and perform consolidation
        size = int(math.log(self.length)*2)
        deg_array = [None] * size
        nodes = self.traverse(self.root)
        for j in range(len(nodes)):
            x = nodes[j]
            deg = x.degree
            while deg_array[deg] != None:
                y = deg_array[deg]
                if x.key > y.key:
                    temp = y
                    y = x
                    x = temp
                # linking two nodes together
                if y == self.root:
                    self.root = y.right
                y.right.left = y.left
                y.left.right = y.right
                y.left = y.right = y
                if x.child is None:
                    x.child = y
                else:
                    y.left = x.child
                    y.right = x.child.right 
                    x.child.right.left = y
                    x.child.right = y
                x.degree += 1
                y.parent = x
                deg_array[deg] = None
                deg += 1
            deg_array[deg] = x
        # find minimum node
        m = 0
        while m < len(deg_array):
            if deg_array[m] is not None and deg_array[m].key < self.min_pointer.key:
                self.min_pointer = deg_array[m]
            m += 1





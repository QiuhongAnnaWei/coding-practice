### 4.4
# Check whether or not a binary tree is balanced
# Balanced = the left and right subtrees of every node differ in height by ≤ 1

## Questions
# Null tree is balanced

class Node:
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Height: # necessary because number cannot be modified through argument passing
    def __init__(self):
        self.height = 0 # base case for recursion

# recursive implementation
def isBalanced(node, node_height=Height()):
    """checks whether or not the tree with node as root is balanced"""
    # base case
    if node is None:
        return True # no need to change node_height (default to 0)

    left_height = Height()
    right_height = Height()

    # 1. check left balanced (update left_height as checking)
    left_balanced = isBalanced(node.left, left_height)
    # 2. check right balanced (update right_height as checking)
    right_balanced = isBalanced(node.right, right_height)

    # update node_height
    node_height.height = max(left_height.height, right_height.height) + 1

    # 3. left and right height difference ≤ = 1
    return left_balanced and right_balanced and abs(left_height.height - right_height.height) <= 1

### O(number of nodes): only visits each node once ###

print(isBalanced(Node()))
n1 = Node()
n2 = Node()
n3 = Node()
n4 = Node()
n5 = Node()
n1.left = n2
n2.left = n3
n2.right = n4
n1.right = n5
n6 = Node()
n3.right=n6
print(isBalanced(n1))


#                 1(3/T)
#       2(2/T)                 5(1/T)
#  3(1/T)      4(1/T)        N(0/T)   N(0/T)
# N(0) N(0)   N(0/T) N(0/T)

### 4.5
# Check if a binary tree is a valid BST
# valid = all its left children’s key < node’s key < all its right children’s key

class Node:
    def __init__(self, val=None, left=None, right=None):
        self.nodeVal = val
        self.left = left
        self.right = right

## recursive implementation 2 (improved from 1)
# check each subtree is within min and max (passed down from parent tree)
# Improvement: No need for class Value because no need to pass min and max back up from subtree
    #  Ex: instead of passing max from left subtree up
    #      can check currVal > all values in left subtree in recursive call to left subtree
def isValidBST(node):
    return isValidBST_helper(node)
def isValidBST_helper(node, min=None, max=None): # made helper fun to match signature requirements (if any)
    # base case
    if node is None:
        return True

    if min is not None and node.nodeVal < min:
        return False # no need to check its subtrees
    if max is not None and node.nodeVal > max:
        return False # no need to check its subtrees

    return isValidBST_helper(node.left, min=min, max=node.nodeVal) and isValidBST_helper(node.right, min=node.nodeVal, max=max)

### O(number of nodes): traverses each node once ###

#                  4(N/N/T)
#      2(N/4-T)                5(4/N-T)
# 1(N/2-T)   3(2/4-T)       N(4/5-T) N(5/N-T)
# N N       N  N

print(isValidBST(Node()))

n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
n4 = Node(4)
n5 = Node(5)
n4.left = n2
n2.left = n1
n2.right = n3
n4.right = n5
n6 = Node(3.5)
n3.right=n6
print(isValidBST(n4))





## recursive implementation 1 (didn't check functionality)
# class Value: # necessary because number cannot be modified through argument passing
#     def __init__(self, val):
#         self.value = val

# def isValidBST(node, treeVal, min):
#     # base case
#     if node is None:
#         return True

#     # 1. check left subtree validity
#     left_max = Value(node.left.nodeVal)
#     left_valid = Value(node.left, val=left_max, min=False) if node.left else True
    
#     # 2. check right subtree validity
#     right_min = Value(node.right.nodeVal)
#     right_valid = Value(node.right, val=right_min, min=True) if node.right else True

#     # update val
#     if min: # maintaining mininum value of subtree
#         if node.nodeVal < treeVal:
#             treeVal.value = node.nodeVal
#     else: # maintaining maximum value of subtree
#         if node.nodeVal > treeVal:
#             treeVal.value = node.nodeVal
    
#     # 3. check its validity: left max < currNode < right min
#     return left_valid and right_valid and (left_max < treeVal.value < right_min)
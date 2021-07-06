### 4.2
# Create a minimum binary search tree from a sorted list
# with as little height as possible

class Node:
    def __init__(self, val=None, left=None, right=None):
        self.nodeVal = val
        self.left = left
        self.right = right
    
    def __str__(self):
        return f"""nodeVal={self.nodeVal} | leftVal = {self.left.nodeVal if self.left else None} | rightVal = {self.right.nodeVal if self.right else None}"""

## Recursive implementation: find midpoint and use indices to shrink range of list in recursive calls
    ## odd number of elements: 10 20 30 (0, 3)
    # 1: N(20, left, right)
        # left (0, 1)->0: N(10, None, None)
        # right (2, 3)->2: N(30, None, None)
    ## even number of elements: 10 20 (0, 2)
    # 1: N(20, left, (2, 2)->None)
        # left (0, 1)->0: N(10, None, None)
def create_minBST(sortedList):
    return create_minBST_helper(sortedList, 0, len(sortedList))
def create_minBST_helper(l, startIdx, endIdx):
    """
    parameters:
    endIdx: 0-indexed, not included in range considered
    """
    # base case (0 item):
    if endIdx == startIdx:
        return None
    # find midpoint given l is sorted
    midIdx = int(1/2 * (endIdx+startIdx)) # 1
    # construct Node with left tree formed from list before midIdx and right tree from list after midIdx
    # print(l[midIdx]) # 1 way of checking output
    rootNode = Node(l[midIdx], create_minBST_helper(l, startIdx, midIdx), create_minBST_helper(l, midIdx+1, endIdx))
    return rootNode

### time: number of recursive calls * each time's call = n * O(1) -> O(n) ###

# result1 = create_minBST([])
# print(result1) => None

# result = create_minBST([10, 20, 30])
# print(result.nodeVal)
# print(result.left.nodeVal)
# print(result.right.nodeVal)

# result = create_minBST([10, 20, 30, 40])
# print(result.nodeVal)
# print(result.left.nodeVal)
# print(result.left.left.nodeVal)
# # print(result.left.right.nodeVal) => None
# print(result.right.nodeVal)
# # print(result.right.left.nodeVal) => None
# # print(result.right.right.nodeVal) => None

# result = create_minBST([10, 20, 30, 40])
# print("node:", result)
# print("node.left:", result.left)
# print("node.right:", result.right)

# result = create_minBST([10, 20, 30, 40, 50, 60, 70])
# print("node:", result)
# print("node.left:", result.left)
# print("node.right:", result.right)
# can calculate the expected level: log_2(number of elements) => round up to nearest int
## log_2(7) < 3  => 3

# result = create_minBST([10, 20, 30, 40, 50, 60, 70, 80, 90])
# print(result)
# print(result.left)
# print(result.left.left)
# print(result.left.right)
# print(result.right)
# print(result.right.left)
# print(result.right.right)
# can calculate the expected level: log_2(number of elements) => round up to nearest int
## log_2(9) < 4  => 4



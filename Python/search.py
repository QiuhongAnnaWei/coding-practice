## Searching Algorithms

# Returns index of x in arr if present, else -1
def binarySearch_rec(arr, x):
    return binarySearch_helper(arr, 0, len(arr)-1, x)
def binarySearch_helper (arr, l, r, x):
    # 2 implementations: iterative (while l <= r) or recursive
    """recursiver helper function for binary search
    
    parameters:
    r: right index of the range to search for (inclusive)
    """
	# Base case
    if r < l: # Element is not present in the array
        return -1
    else:
        mid = (l + r) // 2
		# If element is present at the middle itself
        if arr[mid] == x:
            return mid
        elif arr[mid] > x: # left subarray
            return binarySearch_helper(arr, l, mid-1, x)
        else: # right subarray
            return binarySearch_helper(arr, mid + 1, r, x)

def binarySearch_iterative(arr, x):
    """iterative implementation for binary search
    parameters:
    r: right index of the range to search for (inclusive)
    """
    l = 0
    r = len(arr) - 1
    while l <= r:
        mid = l + (l-r)//2 #  to prevent overflow; original: (l + r) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x: # left subarray
            r = mid - 1
        else: # right subarray
            l = mid + 1
    return -1


arr = [ 2, 3, 4, 9, 10, 40, 50, 60, 70, 80, 90 ]
x = 10
result = binarySearch_iterative(arr, x)
print ("Element is present at index % d" % result) if result != -1 else print("Element is not present in array")


def dfs_recursive(graph, source, visited = []):
    """recursive implementation of depth first search returning a list of all nodes
    parameters:
    graph: dictionary"""
    if source not in visited:
        visited.append(source)
        # base case: leaf node -> backtrack
        if source not in graph:
            return visited
        # general case
        for neighbour in graph[source]:
            visited = dfs_recursive(graph, neighbour, visited) # updating visited
            # ^ result from left neighbor passed to right neighbor
    return visited

def dfs_non_recursive(graph, source):
    if source is None or source not in graph:
        return "Invalid input"
    visited = []
    stack = [source]
    while stack:
        s = stack.pop() # (only diff from bfs)
        if s not in visited:
            visited.append(s) # add node to list of visited nodes
        # NOTE: should add else here? if visited, then continue to next loop
        if s in graph: # not leaf node, has children/neighbor
            for neighbor in graph[s]:
                stack.append(neighbor)
    return visited

# graph = {"A":["B","C","D"],
#            "B":["E"],
#            "C":["F","G"],
#            "D":["H"],
#            "E":["I"],
#            "F":["J"]}
# visited_r = dfs_recursive(graph, "A")
# print(" ".join(visited_r)) # expected A B E I C F J G D H (visiting leftmost child first)
# visited_nr = dfs_non_recursive(graph, "A")
# print(" ".join(visited_nr)) # expected A D H C G F J B E I (visiting rightmost child first)


from dataStructures import BinaryNode 
def dfs_binary_tree(root):
    if root is None:
        return
    else:
        print(root.nodeVal, end=" ")
        dfs_binary_tree(root.left) # until hit bottom
        dfs_binary_tree(root.right)

    # n1 = BinaryNode(1)
    # n2 = BinaryNode(2)
    # n3 = BinaryNode(3)
    # n4 = BinaryNode(4)
    # n5 = BinaryNode(5)
    # n6 = BinaryNode(6)
    # n1.left = n2
    # n1.right = n5
    # n2.left = n3
    # n2.right = n4
    # n5.right = n6
    # dfs_binary_tree(n1)


def bfs(graph, start):
    """visits all the nodes of a graph (connected component) using BFS"""
    explored = [] # all visited nodes
    queue = [start] #n nodes to be checked
    while queue:
        s = queue.pop(0) # pop first node from queue (only diff from dfs_non_recursive)
        if s not in explored:
            explored.append(s) # add node to list of visited nodes
        # NOTE: should add else here? if visited, then continue to next loop
        if s in graph: # not leaf node, has children/neighbor
            for neighbor in graph[s]:
                queue.append(neighbor)
    return explored
 
graph = {"A":["B","C","D"],
           "B":["E"],
           "C":["F","G"],
           "D":["H"],
           "E":["I"],
           "F":["J"]}
# visited = bfs(graph,'A') 
# print(" ".join(visited)) # A B C D E F G H I J



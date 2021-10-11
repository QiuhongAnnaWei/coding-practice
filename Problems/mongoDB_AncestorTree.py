# 2021-09-23
'''
Suppose we have some input data describing a graph of relationships between parents and children over multiple generations. The data is formatted as a list of (parent, child) pairs, where each individual is assigned a unique positive integer identifier.

For example, in this diagram, the earliest ancestor of 6 is 14, and the earliest ancestor of 15 is 2. 

         14
         |
  2      4
  |    / | \
  3   5  8  9
 / \ / \     \
15  6   7    11

Write a function that, for a given individual in our dataset, returns their earliest known ancestor -- the one at the farthest distance from the input individual. If there is more than one ancestor tied for "earliest", return any one of them. If the input individual has no parents, the function should return null (or -1).

Sample input and output:

parent_child_pairs_3 = [
    (2, 3), (3, 15), (3, 6), (5, 6), (5, 7),
    (4, 5), (4, 8), (4, 9), (9, 11), (14, 4),
]

find_earliest_ancestor(parent_child_pairs_3, 8) => 14
find_earliest_ancestor(parent_child_pairs_3, 7) => 14
find_earliest_ancestor(parent_child_pairs_3, 6) => 14
find_earliest_ancestor(parent_child_pairs_3, 15) => 2
find_earliest_ancestor(parent_child_pairs_3, 14) => null or -1
find_earliest_ancestor(parent_child_pairs_3, 11) => 14


Additional example:

  14
  |
  2      4    1
  |    / | \ /
  3   5  8  9
 / \ / \     \
15  6   7    11

parent_child_pairs_4 = [
    (2, 3), (3, 15), (3, 6), (5, 6), (5, 7),
    (4, 5), (4, 8), (4, 9), (9, 11), (14, 2), (1, 9)
]

find_earliest_ancestor(parent_child_pairs_4, 8) => 4
find_earliest_ancestor(parent_child_pairs_4, 7) => 4
find_earliest_ancestor(parent_child_pairs_4, 6) => 14
find_earliest_ancestor(parent_child_pairs_4, 15) => 14
find_earliest_ancestor(parent_child_pairs_4, 14) => null or -1
find_earliest_ancestor(parent_child_pairs_4, 11) => 4 or 1

n: number of pairs in the input
'''

def find_earliest_ancestor(parent_child_pairs_2, node): # n pairs
    # construct node2parents
    node2parents = {}
    for p, c in parent_child_pairs_2: # O(n)
        if c in node2parents:
            node2parents[c].append(p)
        else:
            node2parents[c] = [p]
        if p not in node2parents:
            node2parents[p] = []
    anc2height = {}
    nodeAnc = find_earliest_anc_helper(node2parents,node, 0, anc2height) # O(n)
    # find the earliest in anc2height
    earliestAnc, maxHeight = None, 0
    for anc in anc2height: # O(n)
        if anc2height[anc] > maxHeight:
            maxHeight = anc2height[anc]
            earliestAnc = anc
    return earliestAnc
### Time: O(n)
### Space: heap=node2parents+anc2height -> O(n)
### call stack=O(n)


def find_earliest_anc_helper(node2parents, node, heightSoFar, anc2height):
    # anc2height = {anc node value: how far up from node it is}
    # earliestanc + earliestheight
    '''return all the ancestors of node in a set'''
    if len(node2parents[node])==0:
        anc2height[node] = heightSoFar
        return anc2height
    for parent in node2parents[node]:
        anc2height = find_earliest_anc_helper(node2parents, parent, heightSoFar+1, anc2height)
    return anc2height
    ## Recursion: num of calls=O(n) * time per call=O(1) = O(n)
    
parent_child_pairs_3 = [
    (2, 3), (3, 15), (3, 6), (5, 6), (5, 7),
    (4, 5), (4, 8), (4, 9), (9, 11), (14, 4),
]

parent_child_pairs_4 = [
    (2, 3), (3, 15), (3, 6), (5, 6), (5, 7),
    (4, 5), (4, 8), (4, 9), (9, 11), (14, 2), (1, 9)
]


    
find_earliest_ancestor(parent_child_pairs_3, 8) # => 14
find_earliest_ancestor(parent_child_pairs_3, 7) #=> 14
find_earliest_ancestor(parent_child_pairs_3, 6)  #=> 14
find_earliest_ancestor(parent_child_pairs_3, 15) #=> 2
find_earliest_ancestor(parent_child_pairs_3, 14) #=> null or -1
find_earliest_ancestor(parent_child_pairs_3, 11) #=> 14
    

find_earliest_ancestor(parent_child_pairs_4, 8) #=> 4
find_earliest_ancestor(parent_child_pairs_4, 7) #=> 4
find_earliest_ancestor(parent_child_pairs_4, 6) #=> 14
find_earliest_ancestor(parent_child_pairs_4, 15)# => 14
find_earliest_ancestor(parent_child_pairs_4, 14)# => null or -1
find_earliest_ancestor(parent_child_pairs_4, 11)# => 4 or 1




# node2parents = {every node: [immedaite parents]}; ex:{3: [1,2], 11:[]}
# iterative: find all out ancestors of 3

def has_common_ancestor(parent_child_pairs_2, node1, node2): # n pairs
    # construct node2parents
    node2parents = {}
    for p, c in parent_child_pairs_2: # O(n)
        if c in node2parents:
            node2parents[c].append(p)
        else:
            node2parents[c] = [p]
        if p not in node2parents:
            node2parents[p] = []
    # find out all ancestors of node1 and node2
    node1Anc = find_all_ancestors(node2parents,node1,set()) # O(n)
    node2Anc = find_all_ancestors(node2parents,node2,set()) # O(n)
    return len(node1Anc.intersection(node2Anc))>0
### Recursion: number of calls=height=O(n) * time per call=O(1) -> O(n)
### Overall time: O(n)
### Space: heap = node2parents=O(2n) + node1Anc + node1Anc -> O(n)

    
def find_all_ancestors(node2parents, node, ancestorsSoFar):
    '''return all the ancestors of node in a set'''
    if len(node2parents[node])==0:
        return ancestorsSoFar
    for parent in node2parents[node]:
        ancestorsSoFar.add(parent)
        ancestorsSoFar = find_all_ancestors(node2parents, parent, ancestorsSoFar)
    return ancestorsSoFar
    
#              15
#              |
#          14  13
#          |   |
# 1   2    4   12
#  \ /   / | \ /
#   3   5  8  9
#    \ / \     \
#     6   7     11 


parent_child_pairs_1 = [
    (1, 3), (2, 3), (3, 6), (5, 6), (5, 7), (4, 5),
    (4, 8), (4, 9), (9, 11), (14, 4), (13, 12), (12, 9),
    (15, 13)
]

parent_child_pairs_2 = [
    (1, 3), (11, 10), (11, 12), (2, 3), (10, 2), 
    (10, 5), (3, 4), (5, 6), (5, 7), (7, 8)
]


has_common_ancestor(parent_child_pairs_1, 3, 8) # => false
has_common_ancestor(parent_child_pairs_1, 5, 8) # => true
has_common_ancestor(parent_child_pairs_1, 6, 8) # => true
has_common_ancestor(parent_child_pairs_1, 6, 9) # => true
has_common_ancestor(parent_child_pairs_1, 1, 3)# => false
has_common_ancestor(parent_child_pairs_1, 3, 1) #=> false
has_common_ancestor(parent_child_pairs_1, 7, 11)# => true
has_common_ancestor(parent_child_pairs_1, 6, 5) #=> true
has_common_ancestor(parent_child_pairs_1, 5, 6) # => true


has_common_ancestor(parent_child_pairs_2, 4, 12)# => true
has_common_ancestor(parent_child_pairs_2, 1, 6) #=> false
has_common_ancestor(parent_child_pairs_2, 1, 12)# => false



def findNodesWithZeroAndOneParents(parentChildPairs): # n pairs input
    node2pnum = {}
    for p, c in parentChildPairs: # O(n)
        if c in node2pnum:
            node2pnum[c] += 1
        else:
            node2pnum[c] = 1
        if p not in node2pnum:
            node2pnum[p] = 0
    zeropnodes = []
    onepnodes = []
    for node in node2pnum: # O(2n)
        if node2pnum[node] == 0:
            zeropnodes.append(node)
        elif node2pnum[node] == 1:
            onepnodes.append(node)
    return zeropnodes,onepnodes

### Time: O(n) + O(2n) -> O(n)
### Space: node2pnum=O(2n) + zeropnodes + onepnodes -> O(n)
### 2.8
# Check if a linkedList is a palindrome

## Questions
# empty linkedList
# 1 element linkedList

class Node:
    def __init__(self, val):
        self.val = val
        self.nextNode = None

class LinkedList:
    def __init__(self, hN):
        self.headNode = hN


# linear time implementation using stack and 2 pointer
def isPalindrome(ll):
    """checks whether linkedList is a Palindrome"""
    slow = ll.headNode # 1
    fast = ll.headNode # 1
    stack = [] # first in last out
    # 1. use 2 pointer to find midpoint, meanwhile saving first half in stack - O(n/2)
    while fast != None and fast.nextNode != None:
        stack.append(slow.val)
        slow = slow.nextNode
        fast = fast.nextNode.nextNode
    if fast != None: # odd number of elements
        slow = slow.nextNode # skip midpoint
     
    # 2. compare saved first half with remaining other half - O(n/2)
    while slow != None:
        if stack.pop() != slow.val:
            return False
        slow = slow.nextNode
    return True

### O(n/2) + O(n/2) -> O(n) ###

n1 = Node(1)
n2 = Node(2)
n3 = Node(2)
n4 = Node(1)
n1.nextNode = n2
n2.nextNode = n3
n3.nextNode = n4
ll = LinkedList(n1)
print(isPalindrome(ll))
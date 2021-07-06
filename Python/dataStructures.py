#################################################################

class SLLNode:
    '''Singly Linked List'''
    def __init__(self, val=None):
        self.val = val
        self.nextNode = None # initialize as null
        # self.prevNode (for doubly linked list)

    # NOTE: getVal(), getNextNode(), setNextNode()

class SLinkedList:
    def __init__(self):
        self.headNode = None # initialize as null

    def listPrint(self):
        """prints entire linked list"""
        printval = self.headNode
        while printval is not None:
            print (printval.val)
            printval = printval.nextNode

    def insertAtBeginning(self, newVal): # O(1)
        """Insert new node at the beginning"""
        newNode = SLLNode(newVal)
        newNode.nextNode = self.headNode
        self.headNode = newNode
        
    def insertAtEnd(self, newVal): # O(n)
        """Insert new node at the end"""
        newNode = SLLNode(newVal)
        if self.headNode is None:
            self.headNode = newNode
            return
        lastNode = self.headNode
        while(lastNode.nextNode): # iterate through linked list
            lastNode = lastNode.nextNode
        lastNode.nextNode=newNode
        
    def insertInMiddle(self,middle_node,newVal): # O(1) withe the reference to middle_node
        """Insert new node in the middle
        
        parameters:
        middle_node(SLLNode): the existing node after which the new node will be inserted
        newVal: val of new SLLNode
        """

        if middle_node is None:
            return
        newNode = SLLNode(newVal)
        middle_node.nextNode = newNode
        newNode.nextNode = middle_node.nextNode
        
    def removeNode(self, valBeforeRemoval): # O(n)
        """Remove node from linked list
        
        parameters:
        valBeforeRemoval(SLLNode): val of the node whose following node is removed 
        """

        headNode = self.headNode
        if (headNode is None):
            return
        # first node needs to be removed - O(1)
        if (headNode.val == valBeforeRemoval):
            self.headNode = headNode.nextNode
            return
        else:  # other node needs to be removed - O(n)
            while (headNode is not None):
                if headNode.val == valBeforeRemoval:
                    break
                prev = headNode # will run through at least once
                headNode = headNode.nextNode
            prev.nextNode = headNode.nextNode

def sll():
    list1 = SLinkedList()
    list1.headNode = SLLNode("Mon")
    e2 = SLLNode("Wed")
    e3 = SLLNode("Thu")
    list1.headNode.nextNode = e2 # Link first SLLNode to second node
    e2.nextNode = e3 # Link second SLLNode to third node
    # list1.listPrint() # => Mon Wed Thu

    list1.insertAtBeginning("Sun")
    # list1.listPrint() # => Sun Mon Wed Thu

    list1.insertAtEnd("Fri")
    # list1.listPrint() # => Sun Mon Wed Thu Fri

    list1.insertInMiddle(list1.headNode.nextNode,"Tue")
    # list1.listPrint() # => Sun Mon Tue Wed Thu Fri

    list1.removeNode("Wed")
    # list1.listPrint() # => Sun Mon Tue Thu Fri


#################################################################

class DLLNode:
    def __init__(self, next=None, prev=None, data=None):
        self.data = data
        self.next = next # reference to next node in DLL
        self.prev = prev # reference to previous node in DLL

class DLinkedList:
    def __init__(self):
        self.headNode = None

    def insertAtBeginning(self, new_data): # O(1)
        """ insert at front of list, returning ref to head"""
        new_node = DLLNode(data= new_data)
        new_node.next = self.headNode
        new_node.prev = None
        if (self.headNode != None): # change prev of head node to new node
            self.headNode.prev = new_node
        self.headNode = new_node # reassigning head_ref
        return self.headNode
    
    def insertBefore(self, next_node, new_data): # O(1)
        """ insert before a given node, returning ref to head"""
        if (next_node == None):
            print("the given next node cannot be NULL")
            return
        new_node = DLLNode(data=new_data)

        new_node.prev = next_node.prev
        if (new_node.prev != None):
            new_node.prev.next = new_node
        else:
            self.headNode = new_node

        new_node.next = next_node
        next_node.prev = new_node
        return self.headNode
    
    # def insertAtEnd(self, new_data): # O(n)
    #     """ insert at end of list, returning ref to head"""
        # still need to iterate to get to the last node -> set prev and next pointers

    def deleteNode(self, dele): # O(1) with reference given
        """delete the node referenced by dele"""
        if self.headNode is None or dele is None:
            return
        # If node to be deleted is head node
        if self.headNode == dele:
            self.headNode = dele.next
        if dele.next is not None: # not the last node
            dele.next.prev = dele.prev
        if dele.prev is not None: # not the first node
            dele.prev.next = dele.next
        # Free the memory occupied by dele by calling python garbage collector (import gc)
        # gc.collect() 


    def printList(self, node):
        """prints DLL starting from the given node"""
        last = None
        print("Traversal in forward direction ")
        while (node != None):
            print(node.data, end=" ")
            last = node
            node = node.next
        print("\nTraversal in reverse direction ")
        while (last != None):
            print(last.data, end=" ")
            last = last.prev
 
def dll():
    dll = DLinkedList()
    dll.insertAtBeginning(7)
    dll.insertAtBeginning(1)
    dll.insertAtBeginning(4)
    # Insert 8, before 1. So linked list becomes 4.8.1.7.NULL
    head = dll.insertBefore(dll.headNode.next, 8)
    print("Created DLL is: ")
    dll.printList(head) # 4817 7184


    dll = DLinkedList()
    # 10 <-> 8 <-> 4 <-> 2
    dll.insertAtBeginning(2)
    dll.insertAtBeginning(4)
    dll.insertAtBeginning(8)
    dll.insertAtBeginning(10)
    print("\n---Original Linked List:")
    dll.printList(dll.headNode)
    dll.deleteNode(dll.headNode) # 8 <-> 4 <-> 2
    dll.deleteNode(dll.headNode.next) # 8 <-> 2
    dll.deleteNode(dll.headNode.next) # NULL<-8->NULL
    print("\n---Modified Linked List:")
    dll.printList(dll.headNode)


#################################################################


class Stack:
    '''Stack (FILO) that keeps track of min (CTCI 3.2)
    keeps track of min through a minStack that is sometimes updated in push() and pop()
    '''
    def __init__(self):
        self.stack = []
        self.minStack = [] # addition for Min
        self.size = 0

    def push(self,ele):
        """adds item to top - O(1)"""
        self.stack.append(ele)
        self.size += 1
        currMin = self.getMin() #  addition for Min
        if currMin is None or ele <= currMin: # <= for ease with removal in pop()
            self.minStack.append(ele)

    def pop(self):
        """removes item from the top (most recently added) and returns it - O(1)"""
        if self.size == 0:
            return None
        valueToPop = self.stack.pop() # not None
        self.size -= 1
        if valueToPop == self.getMin(): #  addition for Min, other just return self.stack.pop()
            self.minStack.pop()
        return valueToPop

    def peek(self):
        """returns first element of stack (does not remove) - O(1)"""
        if self.size == 0:
            return None
        return self.stack[-1]

    def isEmpty(self):
        """returns whether stack is empty - O(1)"""
        return self.getSize() == 0

    def getMin(self):
        if len(self.minStack) == 0:
            return None
        return self.minStack[-1]

    def getSize(self):
        return self.size


#################################################################


class Queue():
    ''' Queue (FIFO) implemented with 2 Stacks(FILO) (CTCI 3.5)
    frontStack (reversed, for dequeue) + backStack (normal, for enqueue) = queue
    dump all from backStack into frontStack (reversing order) when frontStack is empty
    ''' 
    # dequeue <- + 1, + 2, + 3 <- enqueue
    def __init__(self):
        self.frontStack = Stack() # +3, +2, +1
        self.backStack = Stack()

    def loadBacktoFront(self):
        """ load backStack into frontStack in reverse order"""
        # ensures that frontStack + backStack = queue
        # O(len of backStack)
        while self.backStack.isEmpty() is not True:
            self.frontStack.push(self.backStack.pop())

    def enqueue(self, ele):
        """inserts item to the back of queue - O(1) """
        self.backStack.push(ele)

    def dequeue(self):
        """removes the first/front item (earliest added)"""
        if self.frontStack.getSize() == 0:
            self.loadBacktoFront()
        return self.frontStack.pop()

    def peek(self):
        """returns first/front element (earliest added)"""
        if self.frontStack.getSize() == 0:
            self.loadBacktoFront()
        return self.frontStack.peek()

    def isEmpty(self):
        """ returns whether queue is empty - O(1) """
        return self.frontStack.getSize() + self.backStack.getSize() == 0
    
    def getSize(self):
        return self.frontStack.getSize() + self.backStack.getSize()

def queue():
    queue = Queue()
    queue.enqueue(10)
    queue.enqueue(20)
    print("2 expected:", queue.getSize())
    queue.enqueue(30)
    print("10 expected:", queue.peek())
    print("3 expected:", queue.getSize())
    print("False expected:", queue.isEmpty())
    print("10 expected:", queue.dequeue())
    print("20 expected:", queue.dequeue())
    print("1 expected:", queue.getSize())
    queue.enqueue(100)
    print("30 expected:", queue.dequeue())
    print("100 expected:", queue.dequeue())


#################################################################


class BinHeap: # min heap
    def __init__(self):
        self.heapList = [0] # 0 there so i // 2 -> parent's index
        self.currentSize = 0 # not equal to len(heapList)

    def siftUp(self,i): # O(log n)
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                self.heapList[i // 2], self.heapList[i] = self.heapList[i], self.heapList[i // 2] # swap
            i = i // 2
        # expensive for node at bottom of tree

    def insert(self,k): # O(log n)
        self.heapList.append(k)
        self.currentSize += 1
        self.siftUp(self.currentSize)

    def siftDown(self,i): # O (log n)
        """start at index i downwards, sift down until the bottom"""
        while (i * 2) <= self.currentSize: # has children
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]: # violating order
                self.heapList[i], self.heapList[mc] = self.heapList[mc], self.heapList[i] # swap
            # NOTE: Should be able to add else here to break if <= minChild (assume children subtrees already sorted)
            ## 1/2 called in removeMin (indeed all sorted)
            ## 2/2 called in buildHeap from middle up
                # middle: children subtree=leaves -> children subtree sorted
                # further up: called from bottom first -> children subtree sorted
            # with current implementation: expensive for node at top
            i = mc
           
    def minChild(self,i): # O(1)
        """ return index of the smaller of the 2 children"""
        if i * 2 + 1 > self.currentSize: # only 1 child
            return i * 2
        else:
            if self.heapList[i*2] < self.heapList[i*2+1]:
                return i * 2
            else:
                return i * 2 + 1

    def removeMin(self): # O(log n)
        retval = self.heapList[1]
        # move last item to root - O (1)
        self.heapList[1] = self.heapList[self.currentSize]
        self.heapList.pop()
        self.currentSize-=1
        # resort - O(log n)
        self.siftDown(1)
        return retval

    def buildHeap(self,alist): # O(n/2 * log n)= O(n log n) => actually O(n)-most nodes sift down depth close to 1
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        i = len(alist) // 2 # for [ len(aList)//2 + 1, len(aList) ], no child & won't enter loop in siftDown
        # start in the middle of list / near bottom of tree (bottom most non-leaf node)
        #  -> work our way back toward front of list / root of tree
        while (i > 0): # when i = 1, at root
            self.siftDown(i)
            i -= 1

def binHeap():
    bh = BinHeap()
    bh.buildHeap([9,5,6,2,3])
    print(bh.removeMin())
    print(bh.removeMin())
    print(bh.removeMin())
    print(bh.removeMin())
    print(bh.removeMin())

    bh.insert(1)
    bh.insert(4)
    bh.insert(5)
    bh.insert(2)
    bh.insert(3)
    print(bh.removeMin())
    print(bh.removeMin())
    print(bh.removeMin())
    print(bh.removeMin())
    print(bh.removeMin())


#################################################################


class BinaryNode:
    '''enough for a binary tree'''
    def __init__(self, val=None, left=None, right=None):
        self.nodeVal = val
        self.left = left
        self.right = right

    def insert(self, value):
        if not value:
            return
        if value < self.value:
            if self.left is None:
                self.left = BinaryNode(value)
            else:
                self.left.insert(value)
        elif value > self.value:
            if self.right is None:
                self.right = BinaryNode(value)
            else:
                self.right.insert(value)
        else:
            pass
    
    def preOrderTrav(root):
        '''DFS-Pre Order Traversal 前序遍历: node -> left subtree -> right subtree (most familiar)'''
        def pre_rec(root, trav):
            if root:
                trav.append(root.nodeVal)
                if root.left: pre_rec(root.left, trav) # if saves fun calls: base case = leaf node instead of None
                if root.right: pre_rec(root.right, trav)
        trav = []
        pre_rec(root, trav)
        return trav

    def inOrderTrav(root):
        '''DFS-In Order Traversal 中序遍历:  left subtree -> node -> right subtree
        recursive implementation'''
        def in_rec(root, trav):
            if root:
                if root.left: in_rec(root.left, trav)
                trav.append(root.nodeVal)
                if root.right: in_rec(root.right, trav)
        trav = []
        in_rec(root, trav)
        return trav
        ### time: traverse each node * O(1) per node -> O(n) ###
        ### space: tree height -> worst O(n)/average O(log n) ###
    def inorderTraversal_iter(root):
        ### REVIEW
        '''Iterative implementation using stack
        keeps appending left until None, then node, then right; resume with stack when finish right leaf
        '''
        stack, trav = [], []
        currN = root
        while currN or stack:
            if currN:
                stack.append(currN)
                currN = currN.left
            else: # stack
                node = (stack.pop())
                trav.append(node.nodeVal)
                currN = node.right
        return trav

    def postOrderTrav(root):
        '''DFS-Post Order Traversal 后序遍历: left subtree -> right subtree -> node'''
        def post_rec(root, trav):
            if root:
                if root.left: post_rec(root.left, trav)
                if root.right: post_rec(root.right, trav)
                trav.append(root.nodeVal)
        trav = []
        post_rec(root, trav)
        return trav

    def bfs():
        return

def binaryTree():
    print(BinaryNode.inorderTraversal_iter(None))
    tree1 = BinaryNode(2, BinaryNode(1), BinaryNode(3))
    print(BinaryNode.inorderTraversal_iter(tree1))

binaryTree()
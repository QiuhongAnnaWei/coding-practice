# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next # another node
    
    @classmethod
    def makeFromList(cls, l):
        """creating a ListNode from a list"""
        if len(l) == 0:
            return None 
        headNode = cls(l[0])
        currNode = headNode
        for num in l[1:]:
            currNode.next = cls(num)
            currNode = currNode.next
        return headNode

    def __str__(self):
        """
        Overriden for print.
        Won't be called on None, which is represented as [] on leetcode.
        """
        curNode = self
        s_list = []
        while curNode:
            s_list.append(str(curNode.val))
            curNode = curNode.next
        return ", ".join(s_list)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Q1: # EASY | def twoSum(self, nums: List[int], target: int) -> List[int]:
    # https://leetcode.com/problems/two-sum/
    # Given an array of integers nums and an integer target,
    # return indices of the two numbers such that they add up to target. 
    # You may assume that each input would have exactly one solution,
    # and you may not use the same element twice.
    # You can return the answer in any order.
    
    ## Optimized Approach: one-pass hash table
    ### 1. Directly check for complement's existence, which requires using num as key (num: numIdx)
    ### 2. Duplicate key concern solved: there are only duplicate numbers if the two duplicates add up to the target ([1, 2, 2, 5] for target 4),
    ### otherwise > 1 solution exists which is invalid ([1, 2, 2, 5] for target 7 not possible)
    def twoSum(self, nums, target):
        candidates = {}
        for numIdx, num in enumerate(nums):
            complement = target - num
            if complement in candidates:
                return [candidates[complement], numIdx]
            else:
                candidates[num] = numIdx
    
    ## Initial Approach 
    def twoSum_initial(self, nums, target):   
        candidates = {}
        for numIdx, num in enumerate(nums):
            # check num's combination with all past candidates
            for candidateIdx, candidate in candidates.items():
                if num + candidate == target:
                    return [candidateIdx, numIdx]
            # does not find target combination
            candidates[numIdx] = num
    
    @staticmethod
    def test():
        q1 = Q1()
        print(q1.twoSum([1, 2, 3], 4))
    
    @classmethod
    def test_cls(cls):
        q1 = cls()
        print(q1.twoSum([1, 2, 3], 4))

# Q1.test()
# Q1.test_cls()


class Q2: # MEDIUM | def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    # https://leetcode.com/problems/add-two-numbers/
    # You are given two non-empty linked lists representing two non-negative integers.
    # The digits are stored in reverse order, and each of their nodes contains a single digit.
    # Add the two numbers and return the sum as a linked list.

    # You may assume the two numbers do not contain any leading zero, except the number 0 itself.

    ## While loop implementation: reduced memory use (stack)
    def addTwoNumbers(self, l1, l2):
        dummyHead = ListNode(0)
        currNode = dummyHead
        nextAddition = 0
        while l1 or l2 or nextAddition:
            val1 = l1.val if l1 else 0 
            val2 = l2.val if l2 else 0
            nextAddition, val = divmod(val1+val2+nextAddition, 10) # quotient, remainder
            currNode.next = ListNode(val)
            currNode = currNode.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return dummyHead.next

    ## Complexity ##
    # time: max(l1len,l2len) times through the loop * O(1) per loop = O( max(l1len,l2len) )
    # space: linked list length = max(l1len,l2len) + 1 -> O( max(l1len,l2len) )

    ## Recursion implementation
    def addTwoNumbers_rec(self, l1, l2):
        return self.addTwoNumbers_rec_helper(l1, l2, 0) # initial digit no addition

    def addTwoNumbers_rec_helper(self, l1, l2, addition):
        # base case: both None
        if l1 == None and l2 == None:
            if addition > 0:
                return ListNode(1, None)
            else:
                return None
        # general case: l1, or l2, or both not None
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        curNodeVal = val1 + val2 + addition # 10
        if curNodeVal >= 10: # max: 9+9+1=19
            n = ListNode(curNodeVal % 10)
            nextAddition = 1
        else:
            n = ListNode(curNodeVal)
            nextAddition = 0
        next1 = l1.next if l1 else None
        next2 = l2.next if l2 else None 
        n.next = self.addTwoNumbers_rec_helper(next1, next2, nextAddition)
        return n
       
    ## Complexity ##
    # time: max(l1len,l2len) calls * O(1) -> O( max(l1len,l2len) )
    # space:
    #   linked list length = max(l1len,l2len) + 1 -> O( max(l1len,l2len) )
    #   recursion tree depth = max(l1len,l2len) -> O( max(l1len,l2len) )

    @staticmethod  
    def test():
        q2 = Q2()
        l1 = ListNode.makeFromList([2, 4, 3])
        l2 = ListNode.makeFromList([5, 6, 4])
        print("expected: 7 0 8 |", q2.addTwoNumbers(l1, l2)) # (342+465=807)
        l3 = ListNode.makeFromList([0])
        l4 = ListNode.makeFromList([0])
        print("expected: 0 |", q2.addTwoNumbers(l3, l4))
        l5 = ListNode.makeFromList([9,9])
        l6 = ListNode.makeFromList([1])
        print("expected: 0 0 1 |", q2.addTwoNumbers(l5, l6)) # 10,009,998
        l7 = ListNode.makeFromList([9,9,9,9,9,9,9])
        l8 = ListNode.makeFromList([9,9,9,9])
        print("expected: 8 9 9 9 0 0 0 1 |", q2.addTwoNumbers(l7, l8)) # 10,009,998

# Q2.test()


class Q3: # MEDIUM | def lengthOfLongestSubstring(self, s: str) -> int:
    # https://leetcode.com/problems/longest-substring-without-repeating-characters/
    # Given a string s, find the length of the longest substring without repeating characters.

    ## sliding window: expand window one character by one, and if it caused
    ## repeating substring, then shrink window from the left until all unique again
    def lengthOfLongestSubstring(self, s):
        if len(s) == 0:
            return 0
        
        winStart = 0
        winEnd = 0 # inclusive
        charSet = {} # {char:index}
        maxLen = 0
        
        while winEnd < len(s):
            if s[winEnd] not in charSet: # can safely expand window
                charSet[s[winEnd]] = winEnd
            else: # found repeating character, need to shrink window 
                newWinStart = charSet[s[winEnd]]+1
                for i in range(winStart, newWinStart):
                    del charSet[s[i]]
                winStart = newWinStart
                charSet[s[winEnd]] = winEnd
            if winEnd-winStart+1 > maxLen:
                maxLen = winEnd-winStart+1
            winEnd+=1
        
        return maxLen
        
    ### Time: max O(2n) -> O(n) ###
        
    @staticmethod
    def test():
        q3 = Q3()
        print("expected 0:", q3.lengthOfLongestSubstring(""))
        print("expected 3:", q3.lengthOfLongestSubstring("abcabcbb"))
        print("expected 1:", q3.lengthOfLongestSubstring("bbbbb"))
        print("expected 3:", q3.lengthOfLongestSubstring("pwwkew"))

# Q3.test()


## TODO: DID NOT FINISH OPTIMIZED SOLUTION
class Q4: # HARD | def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    # https://leetcode.com/problems/median-of-two-sorted-arrays/
    # Given two sorted arrays nums1 and nums2 of size m and n respectively
    # return the median of the two sorted arrays.
    
    ## Helper functions for both approaches
    def medianOfArray(self, arr):
        if len(arr) % 2 ==0: # even length
            half = int(len(arr)/2)
            return (arr[half-1] + arr[half])/2
        else: # odd length
            return arr[int(len(arr)/2)]  
    def flexMed(self, smallerEle, largerEle, isEven):
        if isEven:
            return (smallerEle+largerEle)/2
        else:
            return largerEle   
    
    ## TODO: NOT FINISHED: last ele of ListToSearch may not always be the larger of the 2 ending eles
    ## OPTIMIZED APPROACH: binary search to find lists' number of contributing elements to the half sorted mergeArr
    ## 1) mergeArray's length is fixed (number of contributions from num1 and num2 sum to a fixed number) 
    ## 2) a simple comparison can verify if an ele from num1 and an ele from num2 are the last contributing nums
    ## https://medium.com/@hazemu/finding-the-median-of-2-sorted-arrays-in-logarithmic-time-1d3f2ecbeb46
    def findMedianSortedArrays(self, nums1, nums2) -> float:
        # Choose the smaller of the two list to do binary search on
        if nums1 == []:
            return self.medianOfArray(nums2)
        elif nums2 == []:
            return self.medianOfArray(nums1)

        halfLen = int((len(nums1)+len(nums2))/2)+1 # >= 2
        isEven = (len(nums1)+len(nums2))%2==0
        listToSearch = nums1 if len(nums1)<=len(nums2) else nums2 # assign reference
        otherList = nums2 if len(nums1)<=len(nums2) else nums1
        minCon = 0 if len(otherList) >= halfLen else halfLen-len(otherList) # number of contributions, not index
        maxCon = min(halfLen, len(listToSearch)) # number of contributions, not index

        # nums1/listToSearch = [1,2,5], nums2/OtherList = [3,4,6,7],8
        # halfLen = 5
        # minCon = 0
        # maxCon = 3
        # Binary search
        while minCon < maxCon: # have not located the number of contributions
            con = int((minCon + maxCon)/2) # 1 -> 1
            if con == 0: # guaranteed that: len(otherList) >= halfLen >= 2
                if otherList[halfLen-1] <= listToSearch[0]:
                    return self.flexMed(otherList[halfLen-2], otherList[halfLen-1], isEven)
                else:
                    minCon = con 
            else:
                otherCon = halfLen - con # 4 -> 7
                if len(otherList)<=otherCon: # only happen if nums1 and nums2 have same length
                    if listToSearch[con-1]>=otherList[otherCon-1]:
                        return self.flexMed(otherList[otherCon-1], listToSearch[con-1], isEven)
                    else:
                        minCon = con 
                else:
                    if listToSearch[con-1]>otherList[otherCon-1] and listToSearch[con-1]<otherList[otherCon]:
                        pass
                
    ### Time: O(log( min(m,n) ))  ###    


    ## INITIAL APPROACH: merging the 2 sorted arrays to midpoint,iteratively selecting the smaller head of the 2 arrays    
    def findMedianSortedArrays_actualMerge(self, nums1, nums2) -> float:
        if nums1 == []:
            return self.medianOfArray(nums2)
        elif nums2 == []:
            return self.medianOfArray(nums1)
        
        # both nums1 and nums2 have at least 1 element
        mergeArr = [] # min length 2
        stopLen = int((len(nums1)+len(nums2))/2)+1 # 4->3, 5->3
        isEven = (len(nums1)+len(nums2))%2==0
        nums1Idx = 0
        nums2Idx = 0
        while len(mergeArr) < stopLen:
            # 1. ending case: runs out of elements in nums1 or nums2
            remainingLen = stopLen-len(mergeArr)  # >= 1
            # median must be in nums2 - optimization
            if nums1Idx >= len(nums1):
                if remainingLen == 1:
                    return self.flexMed(mergeArr[-1], nums2[nums2Idx], isEven)
                else: # > 1
                    secLast = nums2[nums2Idx+remainingLen-2]
                    last = nums2[nums2Idx+remainingLen-1]
                    return self.flexMed(secLast, last, isEven)
            # median must be in nums1 - optimization
            if nums2Idx >= len(nums2):
                if remainingLen == 1:
                    return self.flexMed(mergeArr[-1], nums1[nums1Idx], isEven)
                else: # > 1
                    secLast = nums1[nums1Idx+remainingLen-2]
                    last = nums1[nums1Idx+remainingLen-1]
                    return self.flexMed(secLast, last, isEven)
            
            # 2. general case: iteratively appends smaller head of the 2 arrays
            if nums1[nums1Idx] <= nums2[nums2Idx]:
                mergeArr.append(nums1[nums1Idx])
                nums1Idx+=1
            else:
                mergeArr.append(nums2[nums2Idx])
                nums2Idx+=1
        
        secLast, last = mergeArr[-2:]
        return self.flexMed(secLast, last, isEven)

    ### Time: max iterate through half of merged array=(m+n)/2 elements -> O(m+n) ###
            
    @staticmethod
    def test():
        q4 = Q4()
        print("Q4 expected 2:", q4.findMedianSortedArrays([1,3], [2]))
        print("Q4 expected 2.5:", q4.findMedianSortedArrays([1,2], [3, 4]))
        print("Q4 expected 0:", q4.findMedianSortedArrays([0, 0], [0, 0]))
        print("Q4 expected 1:", q4.findMedianSortedArrays([], [1]))
        print("Q4 expected 1:", q4.findMedianSortedArrays([1], []))
        print("Q4 expected 25:", q4.findMedianSortedArrays([10, 20, 30, 40], []))
        print("Q4 expected 15:", q4.findMedianSortedArrays([1], [10, 20, 30]))
        
# Q4.test()


class Q5: # MEDIUM | def longestPalindrome(self, s: str) -> str: 
    # https://leetcode.com/problems/longest-palindromic-substring/ 
    # Given a string s, return the longest palindromic substring in s.
    # 1 <= s.length <= 1000

    ## Linear traversal with the current ele/space as midpoint of palindrome
    def longestPalindrome(self, s: str) -> str:
        
        startIdx = 0 # initialized to first letter
        endIdx = 0 # inclusive
        for midIdx in range(1, len(s)*2-1): # space is odd
            if midIdx % 2 == 0: # letter center
                leftIdx = int(midIdx/2)-1
                rightIdx = int(midIdx/2)+1
            else: # space center
                leftIdx = int(midIdx/2)
                rightIdx = int(midIdx/2)+1
            # Find maximum palindromic substring centered at midIdx
            while leftIdx>=0 and rightIdx<len(s) and s[leftIdx] == s[rightIdx]:
                leftIdx-=1
                rightIdx+=1
            if rightIdx-leftIdx+1-2 > endIdx-startIdx+1:
                # +1: since it is 0-indexed; -2: to correct for last loop
                startIdx = leftIdx+1
                endIdx = rightIdx-1
       
        return s[startIdx:endIdx+1]
        # b_a(1)_b(4)_a(3)_b_d
        # b_a(1)_(3)b(2)_a_d
    
    ### Time: 2n centers, each center traverses at max n/2 elements -> O(n^2)

    @staticmethod
    def test():
        q5 = Q5()
        print("expected 'bab/aba':", q5.longestPalindrome("babad"))
        print("expected 'bb:", q5.longestPalindrome("cbbd"))
        print("expected 'a':", q5.longestPalindrome("a"))
        print("expected 'a/b/c/d/e':", q5.longestPalindrome("abcde"))
        print("expected 'abcdeedcba':", q5.longestPalindrome("abcdeedcba")) # even
        print("expected 'abcdedcba':", q5.longestPalindrome("abcdedcba")) # odd
    
# Q5.test()


class Q11: # MEDIUM | def maxArea(self, height: List[int]) -> int:
    # https://leetcode.com/problems/container-with-most-water/
    # Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai)
    # n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0).
    # Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

    # RELATED: Q42, Q84

    ## Only worth it to shrink width if height can increase (move lower of the 2 ends)
    ## Area may or may not increase with the shrink -> necessary to try the solution space
    def maxArea(self, height) -> int:
        leftPter = 0 # index
        rightPter = len(height)-1 # index
        maxArea = 0
        while leftPter < rightPter:
            area = (rightPter-leftPter) * min(height[leftPter], height[rightPter])
            if area > maxArea:
                maxArea = area
            # move lower of the 2 ends 
            if height[leftPter] <= height[rightPter]:
                leftPter +=1
            else:
                rightPter-=1
        return maxArea
        
    ### Time: O(n) - 2 pointers together traverse entire list once ###
        
    ## BRUTE FORCE: iterate through each left side of container, expanding right side one by one
    ## n-1 + n-2 +... 2 + 1 containers / n Choose 2 = (n-1+1)*(n-1)/2 = n*(n-1)/2 -> O(n^2)
        
        
    @staticmethod
    def test():
        q11=Q11()
        print("11 Expected 49", q11.maxArea([1,8,6,2,5,4,8,3,7]))
        print("11 Expected 1", q11.maxArea([1, 1]))
        print("11 Expected 16", q11.maxArea([4,3,2,1,4]))
        print("11 Expected 2", q11.maxArea([1,2,1]))

        print("11 Expected 8", q11.maxArea([1, 8, 8, 1, 1, 1, 1]))
        print("11 Expected 3", q11.maxArea([1, 4, 3]))
        print("11 Expected 4", q11.maxArea([1, 1, 1, 1, 1]))
        print("11 Expected 8", q11.maxArea([1, 8, 8, 1, 1, 1, 1]))
        print("11 Expected 32", q11.maxArea([8, 1, 8, 1, 8]))
        print("11 Expected 0", q11.maxArea([0, 0, 0, 0]))

# Q11.test()


class Q15: # MEDIUM | def threeSum(self, nums: List[int]) -> List[List[int]]:
    # https://leetcode.com/problems/3sum/
    # Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]]
    # such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
    # Notice that the solution set must not contain duplicate triplets.


    ## Sort (easier deduplication), then do 2sum
    def threeSum(self, nums):
        if len(nums)<3:
            return []
        tripArr = []
        
        nums.sort() # sort in place - O(n log n)
        lastele1 = None
        for ele1Idx in range(0, len(nums)-2): # last 2 num don't have enough num to its right
            if nums[ele1Idx] > 0: # target<0, but not possible with remaining elements
                return tripArr
            if lastele1 is not None and nums[ele1Idx] == lastele1:
                continue # move to distinct value
            target = nums[ele1Idx]*-1
            # 2sum problem with 2 pointer
            lPter = ele1Idx+1
            rPter = len(nums)-1
            while lPter < rPter:
                sum = nums[lPter] + nums[rPter]
                if sum<target:
                    lPter+=1
                elif sum>target:
                    rPter-=1
                else:
                    tripArr.append([nums[ele1Idx],nums[lPter],nums[rPter]])
                    # move to innermost occurence of lPter and rPter's values 
                    while lPter<len(nums)-1 and nums[lPter] == nums[lPter+1]:
                        lPter+=1
                    while rPter>ele1Idx+1 and nums[rPter] == nums[rPter-1]:
                        rPter-=1
                    lPter+=1
                    rPter-=1
                    
            lastele1 = nums[ele1Idx] # for moving to distinct value in the next iteration           
        
        return tripArr

    ## Time: nlogn + [n-1 + n-2 + ... + 2 -> n^2] = O(nlogn + n^2) = O(n^2) ##



    ## Initial Approach: calcualte sum of pairs then iterate through each ele -> deduplication too costly
    def threeSum_mapOf2Sum(self, nums):
        if len(nums)<3:
            return []
        tripArr = []
        pairSums = {} #sum:{smallerVal1,smallerVal2,...}
        countCache = {} # num:count
        # 1.calculate sum for all pairs n(n-1)/2 -> O(n^2)
        for idx1 in range(len(nums)):
            for num in nums[idx1+1:len(nums)]:
                sum = nums[idx1]+num
                smallerNum = nums[idx1] if nums[idx1]<num else num
                # check if 3rd element equals one of the 2 (ex: [2, 2, -4] with only one 2 in nums)
                thirdEle = sum*-1
                if thirdEle == nums[idx1] or thirdEle == num:
                    neededCount=[thirdEle, nums[idx1], num].count(thirdEle) # 3 only if sum=num=0
                    if thirdEle not in countCache:
                        countCache[thirdEle] = nums.count(thirdEle) # O(n)
                    actualCount = countCache[thirdEle] 
                    if neededCount > actualCount:
                        continue # do not add to pairSums
                
                if sum in pairSums:
                    pairSums[sum].add(smallerNum)
                else:
                    pairSums[sum] = {smallerNum}
        # 2. Iterate through each num and find if its negative exists as a sum, meanwhile deduplicating - O(n)
        for num in nums:
            target = -1*num
            if target in pairSums:
                for tripVal in pairSums[target]:
                    tripArr.append([num, tripVal, target-tripVal])
                    # Deduplicate for duplicate appearances of num in nums and triplet in output
                    if (-1*tripVal)!= target and (-1*tripVal) in pairSums:
                        pairSums[-1*tripVal].remove(min(num, target-tripVal))
                    if (tripVal-target)!=target and (tripVal-target)!=(-1*tripVal) and (tripVal-target) in pairSums:
                        pairSums[tripVal-target].remove(min(num, tripVal))
                del pairSums[target]
                
        return tripArr

        ## Supposedly about O(n^2) ###
    
    @staticmethod
    def test():
        q15 = Q15()
        print("15 expected [[-1,-1,2],[-1,0,1]]:", q15.threeSum([-1,0,1,2,-1,-4])) 
        print("15 expected []:", q15.threeSum([]))  
        print("15 expected []:", q15.threeSum([0]))  
        print("15 expected [[0,0,0]]:", q15.threeSum([0,0,0]))
        print("15 expected [[0,0,0]]:", q15.threeSum([0,0,0,0,0]))   
        print("15 expected []:", q15.threeSum([0,0,1]))   
        print("15 expected [-2,1,1]:", q15.threeSum([1,1,1,1,1,1,-2,1,1,1]))   

# Q15.test()


class Q17: # MEDIUM | letterCombinations(self, digits: str) -> List[str]:
    # https://leetcode.com/problems/letter-combinations-of-a-phone-number/
    # Given a string containing digits from 2-9 inclusive, return all possible letter combinations
    # that the number could represent. Return the answer in any order.

    ## storing all the letter combos so far and for every next letter candidate, append to each combo
    def letterCombinations(self, digits: str):
        numToLetter = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"] # index = digit
        if len(digits) == 0:
            return []
        combos = ['']
        for digit in digits:
            updatedCombos = []
            for letter in numToLetter[int(digit)]: # a, b, c
                for comboSoFar in combos:
                    updatedCombos.append(comboSoFar+letter)
            combos = updatedCombos
        return combos

    ## Time: 4^1+4^2+...+4^n < n * 4^n -> O(n*4^n): n for each of the 4^n combinations ## 

    @staticmethod
    def test():
        q17 = Q17()
        digits = ""
        expected = []
        print(sorted(expected)==sorted(q17.letterCombinations(digits)), "| expected", sorted(expected), " | ", sorted(q17.letterCombinations(digits)))
        digits = "4"
        expected = ["g", "h", "i"]
        print(sorted(expected)==sorted(q17.letterCombinations(digits)), "| expected", sorted(expected), " | ", sorted(q17.letterCombinations(digits)))
        digits = "23"
        expected = ["ad","ae","af","bd","be","bf","cd","ce","cf"]
        print(sorted(expected)==sorted(q17.letterCombinations(digits)), "| expected", sorted(expected), " | ", sorted(q17.letterCombinations(digits)))

# Q17.test()


class Q19: # MEDIUM | def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    # https://leetcode.com/problems/remove-nth-node-from-end-of-list/
    # Given the head of a linked list, remove the nth node from the end of the list
    # and return its head.
    # Follow up: Could you do this in one pass?


    ## one pass: using 2 pointers of distance n in between;  dummy node pointing to head + None at the end
    def removeNthFromEnd(self, head, n: int):
        if head.next == None: # n must be 1
            return None
        
        # at least 2 nodes in linked list
        dummyHead = ListNode(None, head)
        first_pter = dummyHead
        second_pter = dummyHead
        # a. make the 2 pointers n nodes apart - traverses n nodes
        for _ in range(n+1):
            first_pter = first_pter.next
        # b. move both pointers until first_pter is None - traverses totalNodes - n nodes
        while first_pter is not None:
            first_pter = first_pter.next
            second_pter = second_pter.next
        second_pter.next = second_pter.next.next
        return dummyHead.next
    ## Time complexity: O(totalNodes) | Space complexity: O(1) of 2 pointers ##
        
        
    ## create a list of all nodes and then remove the node by changing next of adjacent nodes
    def removeNthFromEnd_first(self, head, n: int):
        if head.next == None: # n must be 1
            return None
        
        # at least 2 nodes in linked list
        # a. create a list of the nodes in order - O(totalNodes)
        currNode = head
        nodes = []
        while currNode:
            nodes.append(currNode)
            currNode = currNode.next
        # b. removing the node and returning the head - O(1)
        if n == 1:
            nodes[-2].next = None
        elif n == len(nodes):
            return head.next
        else: # at least 3 nodes in linked list
            nodes[-1*n-1].next = nodes[-1*n+1]
        return head
    ## Time complexity: O(totalNodes) | Space complexity: O(totalNodes) of the list nodes ##
        # Can make space complexity to be O(1) by changing it to 
        ## a. keep a count of the total number of nodes - O(totalNodes) | O(1) 
        ## b. traverse (total number - n) times to find the node to remove - O(totalNodes) | O(1)
    
    @staticmethod
    def test():
        q19 = Q19()
        head1 = ListNode.makeFromList([1, 2, 3, 4, 5])
        print("expected [1, 2, 3, 5] | ", q19.removeNthFromEnd(head1, 2))
        head2 = ListNode.makeFromList([1])
        print("expected [] | ", q19.removeNthFromEnd(head2, 1))
        head3 = ListNode.makeFromList([1, 2])
        print("expected [1] | ", q19.removeNthFromEnd(head3, 1))
        head3 = ListNode.makeFromList([1, 2])
        print("expected [2] | ", q19.removeNthFromEnd(head3, 2))

# Q19.test()

class Q20: # EASY | def isValid(self, s: str) -> bool:
    # https://leetcode.com/problems/valid-parentheses/
    # Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', 
    # determine if the input string is valid. An input string is valid if:
        # 1. Open brackets must be closed by the same type of brackets.
        # 2. Open brackets must be closed in the correct order.


    ## keep a stack of opening characters
    def isValid(self, s):
        if len(s) % 2 == 1: # odd number of elements
            return False
        opens = {"(", "{", "["}
        closes = {")", "}", "]"}
        openChars = []
        for char in s:
            if char in opens:
                openChars.append(char)
            elif char in closes:
                if len(openChars) == 0:
                    return False
                else:
                    lastOpen = openChars.pop()
                    valid = (lastOpen == "(" and char == ")") or \
                            (lastOpen == "{" and char == "}") or \
                            (lastOpen == "[" and char == "]")
                    if not valid:
                        return False
        if len(openChars) == 0:
            return True
        else:
            return False
            
    ### Time: O(len(s)) ###

    @staticmethod
    def test():
        q20 = Q20()
        print("expected True", q20.isValid("()"))
        print("expected True", q20.isValid("()[]{}"))
        print("expected True", q20.isValid("({[]})"))
        print("expected True", q20.isValid("[([]){}]"))
        print("expected False", q20.isValid("[)"))
        print("expected False", q20.isValid("([)]"))
        print("expected False", q20.isValid("[])"))
        print("expected False", q20.isValid("[["))
        print("expected False", q20.isValid("))"))
        print("expected False", q20.isValid(")("))

# Q20.test()


class Q21: # EASY | def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    # https://leetcode.com/problems/merge-two-sorted-lists/
    # Merge two sorted linked lists and return it as a sorted list.
    # The list should be made by splicing together the nodes of the first two lists.
    
    ## compare the heads of the 2 lists and append the smaller of the two
    def mergeTwoLists(self, l1, l2):
        l1curNode = l1
        l2curNode = l2
        dummyHead = ListNode()
        outputcurNode = dummyHead 
        while l1curNode is not None or l2curNode is not None:
            if l1curNode is None:
                outputcurNode.next = l2curNode
                break
            elif l2curNode is None:
                outputcurNode.next = l1curNode
                break
            if l1curNode.val <= l2curNode.val:
                outputcurNode.next = ListNode(l1curNode.val, None)
                l1curNode = l1curNode.next
            else:
                outputcurNode.next = ListNode(l2curNode.val, None)
                l2curNode = l2curNode.next
            outputcurNode = outputcurNode.next
        
        return dummyHead.next
            
    ### Time: O(len(l1)+len(l2)) ###

    @staticmethod
    def test():
        q21 = Q21()
        print("expected [] | ", q21.mergeTwoLists(ListNode.makeFromList([]), ListNode.makeFromList([])))
        print("expected [1, 2, 3] | ", q21.mergeTwoLists(ListNode.makeFromList([]), ListNode.makeFromList([1, 2, 3])))
        print("expected [1, 2, 3] | ", q21.mergeTwoLists(ListNode.makeFromList([1, 2, 3]), ListNode.makeFromList([])))
        print("expected [1, 2, 3, 4, 5] | ", q21.mergeTwoLists(ListNode.makeFromList([1, 2, 3]), ListNode.makeFromList([4, 5])))
        print("expected [1, 2, 3, 4, 5] | ", q21.mergeTwoLists(ListNode.makeFromList([4, 5]), ListNode.makeFromList([1, 2, 3])))
        print("expected [1, 1, 1, 1] | ", q21.mergeTwoLists(ListNode.makeFromList([1, 1, 1]), ListNode.makeFromList([1])))
        print("expected [1, 1, 2, 3, 4] | ", q21.mergeTwoLists(ListNode.makeFromList([1, 1, 3]), ListNode.makeFromList([2, 4])))
        print("expected [-2, 0, 1, 1, 3, 4] | ", q21.mergeTwoLists(ListNode.makeFromList([1, 1, 3]), ListNode.makeFromList([-2, 0, 4])))
        print("expected [1, 1, 2, 3, 4, 4] | ", q21.mergeTwoLists(ListNode.makeFromList([1, 2, 4]), ListNode.makeFromList([1, 3, 4])))


# Q21.test()


class Q22: # MEDIUM | def generateParenthesis(self, n: int) -> List[str]:
    # https://leetcode.com/problems/generate-parentheses/
    # Given n pairs of parentheses, 
    # write a function to generate all combinations of well-formed parentheses.

    ## combos of 3 parantheses = 2 parentheses combinations + placement of additional ()
    ## -> dynamic programming: dp[i] = [ all combinations with i pairs of parentheses ]
    def generateParenthesis(self, N):
        dp = [ [] for _ in range(N+1) ]
        dp[0] = [""]
        for pnum in range(1, N+1):
            for i in range(pnum): # how many pairs to place inside additional (): [0, n-1]
                dp[pnum]+=[ "(" + a + ")"+ b for a in dp[i] for b in dp[pnum-1-i]]
        return dp[N] # dp[-1]
    ### Time: through all 4^N/(sqrt(N)) combinations * O(1) -> O( 4^N/(sqrt(N)) ) ###
    # iterate through all combinations from 0 to N: (4^N)/(N*sqrt(N)) + that for N-1 + ... + 0 -> (4^N)/(N*sqrt(N)) * N/2 -> 4^N/(sqrt(N))
    ### Space: dp -> O(N) ###

    ## at each step, can either open an untouched pair, or close an opened pair
    def generateParenthesis_rec(self, N):
        untouchedPairCt = N
        openedPairCt = 0
        if N == 1:
            return ["()"]
        else:
            return self.generateParenthesisRec([], untouchedPairCt, openedPairCt, [])
        
    def generateParenthesisRec(self, comboSoFar, untouchedPairCt, openedPairCt, listSoFar):       
        # 1. base case
        if untouchedPairCt == 0 and openedPairCt == 0:
            listSoFar.append(''.join(comboSoFar)) # pass by reference
            return listSoFar
        # 2. sequential: not modifying comboSoFar and counters between first and second 'if'
        if untouchedPairCt > 0:
            comboSoFar.append('(')
            listSoFar = self.generateParenthesisRec(comboSoFar, untouchedPairCt-1, openedPairCt+1, listSoFar)
            comboSoFar.pop()
        if openedPairCt > 0 and len(comboSoFar)>0: # not the starting symbol
            comboSoFar.append(')')
            listSoFar = self.generateParenthesisRec(comboSoFar, untouchedPairCt, openedPairCt-1, listSoFar)
            comboSoFar.pop()
        return listSoFar
    
    ### Time: 4^N/(sqrt(N)) calls * 1 per call = O[ (4^N)/(sqrt*N) ] ###
    ### (4^N)/(N*sqrt(N)) valid combinations * 2N calls max for each = 4^N/(sqrt(N)) calls in total
        
    @staticmethod
    def test():
        q22 = Q22()
        print('expected ["()"]', q22.generateParenthesis(1))
        print('expected ["(())","()()"]', q22.generateParenthesis(2))
        print('expected ["((()))","(()())","(())()","()(())","()()()"]', q22.generateParenthesis(3))

# Q22.test()


from queue import PriorityQueue
class Q23: # HARD | def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    # https://leetcode.com/problems/merge-k-sorted-lists/
    # You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. 
    # Merge all the linked-lists into one sorted linked-list and return it.

    ## Optimize mergeKLists_list by using a priority queue to get min each time
    def mergeKLists(self, lists):
        # to fix " '<' not supported between instances of 'ListNode' and 'ListNode' "
        ListNode.__lt__=lambda self,other: self.val<=other.val
        if len(lists) == 0:
            return None
        dummyhead = ListNode()
        outputNode = dummyhead
        q = PriorityQueue()
        for listnode in lists:
            if listnode: # remove None
                q.put( (listnode.val, listnode) ) 
        while q.qsize() > 0: # as long as any ListNode in lists is not None
            nodeVal, node = q.get() # remove and return the node with minimum val
            outputNode.next = node # will reassign node's next in the next iteration (save memory)
            outputNode = outputNode.next # next is None
            if node.next:
                q.put( (node.next.val, node.next) )
        return dummyhead.next
    ### N: number of ListNodes across all k linked lists
    ### Time: N nodes in final list * remove from priority queue to select each O(logk) -> O(Nlogk) ###
    ### Space: output takes O(N) + O(k) priority queue -> O(N) + O(k) ###
    
    
    ### maintain current value of each of the list, appending the smallest among them at each step
    def mergeKLists_list(self, lists):
        if len(lists) == 0:
            return None
        dummyhead = ListNode()
        outputNode = dummyhead
        lists = [ currNode for currNode in lists if currNode] # remove None
        while len(lists) > 0: # as long as any ListNode in lists is not None
            minNodeIdx = 0
            minVal = lists[0].val
            for i in range(1, len(lists)):
                if lists[i].val < minVal:
                    minVal = lists[i].val
                    minNodeIdx = i
            outputNode.next = lists[minNodeIdx] # will reassign .next in the next iteration (save memory)
            outputNode = outputNode.next # next is None
            if lists[minNodeIdx].next is None:
                lists.pop(minNodeIdx)
            else:
                lists[minNodeIdx] = lists[minNodeIdx].next
        return dummyhead.next
    ### N: number of ListNodes across all k linked lists
    ### Time: N nodes in final list * iterate thorugh ≤ k eles of ‘lists' to select each -> O(kN) ###
    ### Space: output takes O(N) + O(1) minNodeIdx and minVal -> O(N) ###

    @staticmethod
    def test():
        q23 = Q23()
        print("expected None |", q23.mergeKLists([]))
        print("expected None |", q23.mergeKLists([ None ]))
        print("expected [1, 2, 3] |", q23.mergeKLists([ ListNode.makeFromList([1, 2, 3]), None ]))
        print("expected [1, 1, 2, 2, 3, 3] |", q23.mergeKLists([ ListNode.makeFromList([1, 2, 3]), ListNode.makeFromList([1, 2, 3]) ]))
        lists = [ListNode.makeFromList([-1,0,2,3]), ListNode.makeFromList([-10,2,2]), ListNode.makeFromList([5,5]), ListNode.makeFromList([1])]
        print("expected [-10, -1, 0, 1, 2, 2, 2, 3, 5, 5] |", q23.mergeKLists(lists))

# Q23.test()


class Q31: # MEDIUM | def nextPermutation(self, nums: List[int]) -> None:
    # https://leetcode.com/problems/next-permutation/
    # Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.
    # If such an arrangement is not possible, it must rearrange it as the lowest possible order (i.e., sorted in ascending order).
    # The replacement must be in place and use only constant extra memory.


    ## find the number with larger numbers behind that is closest to the end, 
    # change it to the smallest and closest-to-the-end-among-same of the larger numbers, 
    # and reverse the numbers after it
    def nextPermutation(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """ 
        if len(nums) == 1:
            return nums     
        # 1 [2] 4 3 [3] 1 -> 1 3 4 3 2 1 -> 1 3 [1 2 3 4]
        smallerIdx = -1 
        largerIdx = -1
        # 1. find the number with larger numbers behind that is closest to the end - O (n)
        max = -1
        for idx, num in reversed(list(enumerate(nums))):
            if max > num:
                smallerIdx = idx
                break
            elif max < num:
                max = num
        if smallerIdx == -1:
            nums.sort()
            return
        # 2. find the smallest (and closest-to-the-end-among-same) of the larger numbers - O (n)
        largerIdx = smallerIdx+1 # always defined as smallerIdx cannot refer to last element
        for idx in range(smallerIdx+2, len(nums)): # may be empty
            if nums[idx] > nums[smallerIdx] and nums[idx] <= nums[largerIdx]: 
                ## ≤ ensures closest-to-the-end-among-same -> keeps the descending order -> reverse
                largerIdx = idx
        # 3. swap the 2 and reverse the numbers after the original smallerIdx (at least 1 num) - O (n)
        nums[smallerIdx], nums[largerIdx] =  nums[largerIdx], nums[smallerIdx]
        left = smallerIdx+1
        right = len(nums)-1
        while (left<right):
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
 
    ### Time: O(n) + O(n) + O(n) => O(n) ###
    ### Space: O(1) ### 

    ## find the number with larger numbers behind that is closest to the end, change it to the smallest of the larger numbers, and sort the numbers after it
    def nextPermutation_notconstantmemory(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """ 
        if len(nums) == 1:
            return nums     
        # 1 [2] 4 [3] 3 1 -> 1 3 4 2 3 1 -> 1 3 [1 2 3 4]
        smallerIdx = -1 
        largerIdx = -1
        # 1. find the number with larger numbers behind that is closest to the end - O (n)
        max = -1
        for idx, num in reversed(list(enumerate(nums))):
            if max > num:
                smallerIdx = idx
                break
            elif max < num:
                max = num
        if smallerIdx == -1:
            nums.sort()
            return
        # 2. find the smallest (and closest-to-the-front-among-same) of the larger numbers - O (n)
        largerIdx = smallerIdx+1 # always defined as smallerIdx cannot refer to last element
        for idx in range(smallerIdx+2, len(nums)): # may be empty
            if nums[idx] > nums[smallerIdx] and nums[idx] < nums[largerIdx]: 
                ## < not ≤ ensures largerIdx is closest-to-the-front-among-same -> ruins the descending order -> extra memory
                largerIdx = idx
        # 3. swap the 2 and sort the numbers after the original smallerIdx - O (n log n)
        smallTemp = nums[smallerIdx]
        nums[smallerIdx] = nums[largerIdx]
        nums[largerIdx] = smallTemp
        nums[smallerIdx+1:] = sorted(nums[smallerIdx+1:])

    ### Time: O(n) + O(n) + O(n log n) => O(n log n) ###
    ### Space: nums[smallerIdx+1:] - O(n) ###

    @staticmethod
    def test():
        q31 = Q31()
        nums = [1]
        q31.nextPermutation(nums)
        print("expected [1] |", nums)
        nums = [3, 2, 1]
        q31.nextPermutation(nums)
        print("expected [1, 2, 3] |", nums)
        nums = [1, 2, 3]
        q31.nextPermutation(nums)
        print("expected [1, 3, 2] |", nums)
        nums = [1, 2, 3, 3]
        q31.nextPermutation(nums)
        print("expected [1, 3, 2, 3] |", nums)
        nums = [1, 2, 4, 3, 3]
        q31.nextPermutation(nums)
        print("expected [1, 3, 2, 3, 4] |", nums)
        nums = [1, 2, 4, 3, 3, 2, 1, 0]
        q31.nextPermutation(nums)
        print("expected [1, 3, 0, 1, 2, 2, 3, 4] |", nums, [1, 3, 0, 1, 2, 2, 3, 4]==nums)

# Q31.test()



class Q32: # HARD | def longestValidParentheses(self, s: str) -> int:
    # https://leetcode.com/problems/longest-valid-parentheses/
    # Given a string containing just the characters '(' and ')', 
    # find the length of the longest valid (well-formed) parentheses substring.

    ## dp: stores length of longest substring ending at index
    def longestValidParentheses(self, s):
        if len(s) < 2:
            return 0
        dp = [0 for _ in range(len(s))]
        for i in range(1, len(s)): # dp[0]=0
            if s[i] == ")":
                if s[i-1] == "(":
                    ### NOTE: core of dp 1 - for (()) (), {2 of additional ()} + {4 of (()) before} = 6
                    dp[i] = 2+dp[i-2] if i-2>=0 else 2 # check for [(())]()
                elif s[i-1] == ")":
                    if i-dp[i-1]-1>= 0 and s[ i-dp[i-1]-1 ] == "(": # else 0
                        ### NOTE: core of dp 2 - for () (()), {2 of additional ()} + {inner()'s 2} + {2 of () before} = 6
                        dp[i] = 2+dp[i-1]+dp[i-dp[i-1]-2] if i-dp[i-1]-2>=0 else 2+dp[i-1] # check for [()](())
        return max(dp)
    ### Time: iterate through each character to fill dp, O(1) at each -> O(n) ###
    ### Space: dp -> O(n) ###
    
    ## Further optimized solution with O(1) space
    def longestValidParentheses_O1Space(self, s):
        l,r = 0,0
        maxLen = 0
        for i in range(len(s)): # forward traversal
            if s[i] == '(':
                l +=1
            else:
                r +=1
            if r == l:
                maxLen = max(maxLen, l+r)
            elif r > l: # current substring not valid, to avoid ())(
                l = r = 0
        l,r = 0,0
        for i in range(len(s)-1,-1,-1): # backward traversal - account for cases of r<l like ( ((()))
            if s[i] == '(':
                l += 1
            else:
                r += 1
            if l == r:
                maxLen = max(maxLen, l+r)
            elif l > r: # current substring not valid, to avoid )(()
                l = r = 0
        return maxLen
    ### Time: 2 traversal through s, constant at each char -> O(n) ###
    ### Space: l, r, maxLen -> O(1) ###
    
    @staticmethod
    def test():
        q32 = Q32()
        print(0==q32.longestValidParentheses(""))
        print(0==q32.longestValidParentheses("("))
        print(0==q32.longestValidParentheses(")"))
        print(2==q32.longestValidParentheses("()"))
        print(0==q32.longestValidParentheses(")("))
        print(0==q32.longestValidParentheses(")))"))
        print(0==q32.longestValidParentheses("(("))
        print(2==q32.longestValidParentheses("(()"))
        print(4==q32.longestValidParentheses("(())"))
        print(6==q32.longestValidParentheses("(()())"))
        print(6==q32.longestValidParentheses("((()())"))
        print(6==q32.longestValidParentheses("(()()))"))
        print(4==q32.longestValidParentheses(")))()()"))
        print(4==q32.longestValidParentheses(")))()()((("))
        print(4==q32.longestValidParentheses(")))()())))"))
        print(4==q32.longestValidParentheses("(()))())("))
        
# Q32.test()      



class Q33: # MEDIUM | def search(self, nums: List[int], target: int) -> int:
    # https://leetcode.com/problems/search-in-rotated-sorted-array/
    # There is an integer array nums sorted in ascending order (with distinct values).
    # Prior to being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length)
    # such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
    # For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].
    # Given the array nums after the rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.
    # You must write an algorithm with O(log n) runtime complexity.

    ## use binary search to find pivot, then use binary search on the relevant half to find target
    def search(self, nums, target):
        # 1. use binary search to find pivot (pair where ele_i > ele_i+1 ) - O(log n)
        if len(nums) == 1:
            if nums[0] == target:
                return 0
            else:
                return -1
        else:
            left = 0
            right = len(nums) - 1
            pivot = None  # start of second half
            while left < right: # < to ensure mid2 is within bounds
                mid1 = (left + right) // 2
                mid2 = mid1 + 1
                if nums[mid1] > nums[mid2]: # found the pivot
                    pivot = mid2
                    break
                else:
                    if nums[mid1] > nums[right]:
                        left = mid1
                    else:
                        right = mid1
        
        # 2. use binary search to find target in the relevant half - O(log n)
        left, right = 0, len(nums) - 1
        if pivot is not None: # has 2 parts, can narrow search range
            if target < nums[0]:
                left = pivot
                right = len(nums) - 1
            else:
                left = 0
                right = pivot - 1
        while left <= right: 
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    ### Time: O(log n) ###
    ### Space: O(1) ###
    
    @staticmethod
    def test():
        q33 = Q33()
        print(0 == q33.search([1], 1))
        print(-1 == q33.search([1], 10))
        print(1 == q33.search([1, 2, 3], 2))
        print(-1 == q33.search([1, 2, 3], -2))
        print(2 == q33.search([4, 1, 2, 3], 2)) # pivot in left half
        print(1 == q33.search([1, 2, 3, 0], 2)) # pivot in right half
        print(2 == q33.search([3, 4, 1, 2], 1)) # pivot in middle, target is at pivot
        print(0 == q33.search([4, 5, 6, 7, 8, 9, -1, 0, 3], 4))
        print(8 == q33.search([4, 5, 6, 7, 8, 9, -1, 0, 3], 3))
        print(-1 == q33.search([4, 5, 6, 7, 8, 9, -1, 0, 3], 20))

# Q33.test()



class Q34: # MEDIUM | def searchRange(self, nums: List[int], target: int) -> List[int]:
    # https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/


    ## The 2 solutions about the same

    ## use binary search, find first apperance of target with special break condition, then find lastIdx
    ## (<tar, tar) -> find 2nd of the pair (first appearance) ; (tar, >tar) -> find 1st of the pair (last appearance)
    def searchRange(self, nums, target):
        if len(nums) == 0:
            return [-1, -1]
        # 1. search in entire nums list for first apperance of target (firstIdx) - O(log n)
        l = 0
        r = len(nums)-1
        firstIdx = -1
        while l <= r:
            mid = (l+r)//2
            if nums[mid]==target:
                if mid==0 or nums[mid-1]<target: # won't be out of bounds because of or
                    ## NOTE: instead of using mid1 and mid2 as below, use this conidition
                    firstIdx = mid
                    break
                else: # nums[mid-1] must also be target
                    r = mid - 1
            elif nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        if firstIdx == -1:
            return [-1, -1]
        # 2. target found, search in right half (all >= target) for lastIdx - O(log n)
        else:
            l = firstIdx
            r = len(nums)-1
            while l <= r:
                mid = (l+r)//2
                if nums[mid]==target:
                    if mid==len(nums)-1 or nums[mid+1]>target: # won't be out of bounds because of or
                        return [firstIdx, mid]
                    else: # nums[mid+1] also target
                        l = mid + 1
                else: # nums[mid] > target
                    r = mid - 1
        
    ### Time: 2 binary searches: O(log n) ###
    ### Space: O(1)  - (less space than the solution below) ### 


    ## find one instance of the target, then binary search on the 2 halves for startIdx and endIdx
    def searchRange_3search(self, nums, target):
        if len(nums) == 0:
            return [-1, -1]
        # 1. use binary search to find if target is in nums - O(log n)
        l = 0
        r = len(nums)-1
        targetI = -1
        while l <= r:
            mid = (l+r)//2
            if nums[mid] == target:
                targetI = mid
                break
            elif nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        if targetI == -1:
            return [-1, -1]
        # 2. use binary search on the 2 halves to find startIdx (find pair of (<tar, tar)) and endIdx (find pair of (tar, >tar)) - 2 * O(log(n/2))
        else:
            startIdx = 0 # default: start of nums
            l = 0
            r = targetI
            while l < r: # at least 2 elements, all <= target
                mid1 = (l+r)//2
                mid2 = mid1+1
                if nums[mid1] < target:
                    if nums[mid2] == target:
                        startIdx = mid2
                        break
                    elif nums[mid2] < target:
                        l = mid2
                elif nums[mid1] == target: # nums[mid2] must also == target
                    r = mid1
            endIdx = len(nums)-1 # default: start of nums
            l = targetI
            r = len(nums)-1
            while l < r: # at least 2 elements, all >= target
                mid1 = (l+r)//2
                mid2 = mid1+1
                if nums[mid2] > target:
                    if nums[mid1] == target:
                        endIdx = mid1
                        break
                    elif nums[mid1] > target:
                        r = mid1
                elif nums[mid2] == target: # nums[mid1] must also == target
                    l = mid2
            return [startIdx, endIdx]
        
    ### Time: 3 binary searches: O(log n) ###
    ### Space: O(1) - (more space than the solution above) ### 
    

    @staticmethod
    def test():
        q34 = Q34()
        print("Q34")
        print([-1,-1]==q34.searchRange([], 0))
        print([-1, -1]==q34.searchRange([1], 0))
        print([0, 0]==q34.searchRange([0], 0))
        print([0, 1]==q34.searchRange([-3, -3], -3))
        print([0, 2]==q34.searchRange([3, 3, 3], 3))
        print([3, 3]==q34.searchRange([1, 3, 4, 5, 6], 5))
        print([3, 5]==q34.searchRange([1, 3, 4, 5, 5, 5, 6], 5))
        print([-1, -1]==q34.searchRange([1, 3, 4, 5, 5, 5, 7], 6))
        print([-1, -1]==q34.searchRange([1, 3, 4, 5, 5, 5, 7], -10))

# Q34.test()

        

class Q39: # MEDIUM | def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    # https://leetcode.com/problems/combination-sum/
    # Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations
        # of candidates where the chosen numbers sum to target. You may return the combinations in any order.
    # The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if
    #   the frequency of at least one of the chosen numbers is different.
    # It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

    ## recursion: append each candidate to comboSoFar and changes remainingTar accordingly
    def combinationSum(self, candidates, target):
        combos = []
        candidates.sort() # - O(c log c)
        self.combinationSumRec(candidates, [], target, 0, combos)
        return combos
    
    def combinationSumRec(self, candidates, comboSoFar, remainingTar, idx, combos):
        '''
        candidates: sorted original list of all candidates
        comboSoFar: list of numbers in the combination so far
        remainingTar: remaining value to sum to
        idx: only consider condidates[idx:] to avoid duplicates
        combos: the final output list modified throughout the recursive calls
        '''
        # base case
        if remainingTar == 0:
            combos.append(comboSoFar)
            return
        # general case:
        for i in range(idx, len(candidates)):
            if candidates[i] > remainingTar: # since candidates is sorted
                return # break out of the loop
            else:
                self.combinationSumRec(candidates, comboSoFar+[candidates[i]], remainingTar-candidates[i], i, combos)

    ### Time: O(c log c) + O(candidates_len^comboSoFar_len) branches * O(1) at each node ###
    ### Space: comboSoFar - O(target=max_comboSoFar_len), combos - O(number of combos) ###

    @staticmethod
    def test():
        q39 = Q39()
        print("expected [] | ", q39.combinationSum([2, 3, 6, 7], 1))
        print("expected [] | ", q39.combinationSum([2, 4, 6, 7], 5))
        print("expected [[1]] | ", q39.combinationSum([1], 1))
        print("expected [[1, 1, 1, 1, 1]] | ", q39.combinationSum([1], 5))
        print("expected [[1]] | ", q39.combinationSum([3, 1, 6, 7], 1))
        print("expected [[1, 1]] | ", q39.combinationSum([1], 2))
        print("expected [[1, 1], [2]] | ", q39.combinationSum([3, 2, 1], 2))
        print("expected [[2, 2, 3], [7]] | ", q39.combinationSum([2, 3, 6, 7], 7))
        print("expected [[2, 2, 3], [7]] | ", q39.combinationSum([7, 3, 6, 2], 7))
        print("expected [[2, 2, 2, 2],[2, 3, 3],[3, 5]] | ", q39.combinationSum([2, 3, 5], 8))

# Q39.test()



class Q41: # HARD | def firstMissingPositive(self, nums: List[int]) -> int:
    # https://leetcode.com/problems/first-missing-positive/
    # Given an unsorted integer array nums, find the smallest missing positive integer.
    # You must implement an algorithm that runs in O(n) time and uses constant extra space.

    ## Could not solve and looked at solution
    ## NOTE: As indices are continuous and ascending, swap nums' elements in place to find the first index that does not have a corresponding value
    def firstMissingPositive(self, nums):
        for i in range(len(nums)):
            value = nums[i]
            while value>0 and value<=len(nums) and value != nums[value-1]:
                nums[i], nums[value-1] = nums[value-1], nums[i]
                value = nums[i] # original nums[value-1]
        for i in range(len(nums)):
            if nums[i] != i+1:
                return i+1
        return len(nums)+1
    
    ### Time: around O(n) + O(<=n) -> O(n) ###
    ### Space: O(1) ###

    @staticmethod
    def test():
        q41 = Q41()
        print(1==q41.firstMissingPositive([2]))
        print(2==q41.firstMissingPositive([1]))
        print(1==q41.firstMissingPositive([-1]))
        print(1==q41.firstMissingPositive([0]))
        print(4==q41.firstMissingPositive([1, 2, 3]))
        print(6==q41.firstMissingPositive([4, 1, 2, 5, 3]))
        print(4==q41.firstMissingPositive([0, 2, 3, 1]))
        print(4==q41.firstMissingPositive([3, 1, 2, -4, 2]))
        print(1==q41.firstMissingPositive([5, 4, 3]))
        print(2==q41.firstMissingPositive([1, 1, 1]))
        print(2==q41.firstMissingPositive([4, 1, -10, 0, 3]))
        print(3==q41.firstMissingPositive([1, 2, 2, 1]))

# Q41.test()



class Q42: # HARD | def trap(self, height: List[int]) -> int:
    # https://leetcode.com/problems/trapping-rain-water/
    # Given n non-negative integers representing an elevation map where the width of each bar is 1,
    # compute how much water it can trap after raining.

    ## stores (decreaseIdx:decreaseHeight) in stack, pop all < next height when encounter increase add corresponding water volume
    def trap(self, height):
        water = 0
        decreaseStack = []
        for i in range(len(height)-1): # only enter the loop if len(height)>=2
            change = height[i+1] - height[i]
            if change<0: # decrease
                decreaseStack.append((i, abs(change)))
            elif change>0: # increase
                while len(decreaseStack)>0:
                    decreaseIdx, h = decreaseStack[-1]
                    if height[decreaseIdx] <= height[i+1]:
                        decreaseStack.pop() # all the water that can be trapped already added (shorter of the 2 decides this)
                        water += (i-decreaseIdx)*h
                    else: # first one where height[decreaseIdx] > height[i+1]
                        diff = height[decreaseIdx] - height[i+1]
                        water+= (i-decreaseIdx)*(h-diff)
                        decreaseStack[-1] = (decreaseIdx,diff)
                        break
        return water

    ### n = length of height
    ### Time: traverse through height-O(n) + through this iteration in total append and pop all decrease-O(≤n)+O(≤n)=O(n) -> O(n) ###
    ### space: decreaseStack - O(n) ###

    @staticmethod
    def test():
        q42 = Q42()
        print(0==q42.trap([]))
        print(0==q42.trap([1]))
        print(0==q42.trap([1, 1, 1, 1]))
        print(6==q42.trap([0,1,0,2,1,0,1,3,2,1,2,1]))
        print(9==q42.trap([4,2,0,3,2,5]))

# Q42.test()



class Q45: # MEDIUM | def jump(self, nums: List[int]) -> int
    # https://leetcode.com/problems/jump-game-ii/
    # Given an array of non-negative integers nums, you are initially positioned at the first index of the array.
    # Each element in the array represents your maximum jump length at that position.
    # Your goal is to reach the last index in the minimum number of jumps.
    # You can assume that you can always reach the last index.

    ## find range of each number of jumps
    def jump(self, nums):
        if len(nums) == 1:
            return 0
        l, r = 0, nums[0] # inclusive index range of a patriuclar numJumps
        numJumps = 1
        while r < len(nums)-1: # hasn't reached the end yet
            farthest = max(i+nums[i] for i in range(l, r+1)) # inclusive index
            l, r = r, farthest
            numJumps += 1
        return numJumps # r>=len(nums)-1
    
    ### Time: go through each index once (except for overlap of ranges - max <2n) -> O(n) ###

    ## find leftmost that can reach destination, then find min number of jumps to get there, repeat recursively (base case is index==0)
    def jump_rec(self, nums):
        if len(nums) == 1:
            return 0
        mins = [-1 for _ in range(len(nums))] # each index stores min number of jumps to get there
        self.jump_helper(nums, len(nums)-1, mins)
        return mins[-1]    
    def jump_helper(self, nums, destIdx, mins):
        '''populates mins[destIdx], the minimum number of jumps to get to destIdx'''
        if destIdx == 0:
            mins[0] = 0
            return
        leftIdx = destIdx
        for i in range(destIdx-1, -1, -1):  # find leftmost that can reach destIdx
            if destIdx-i <= nums[i]: # possible to reach destIdx
                leftIdx = i
        # guaranteed to find a leftIdx
        if mins[leftIdx] < 0: # has not been calculated before
            self.jump_helper(nums, leftIdx, mins) # populate mins[leftIdx]
        mins[destIdx] = min(mins[destIdx], mins[leftIdx]+1) if mins[destIdx]>0 else mins[leftIdx]+1
    
    @staticmethod
    def test():
        q45 = Q45()
        print(0==q45.jump([0]))
        print(0==q45.jump([1]))
        print(1==q45.jump([1, 1]))
        print(2==q45.jump([1, 1, 1]))
        print(1==q45.jump([2, 1, 1]))
        print(2==q45.jump([2,3,1,1,4]))
        print(2==q45.jump([2,3,0,1,4]))

# Q45.test()



class Q46: # MEDIUM | def permute(self, nums: List[int]) -> List[List[int]]:
    # https://leetcode.com/problems/permutations/
    # Given an array nums of distinct integers, return all the possible permutations.
    # You can return the answer in any order.

    ## Build permutation one num at a time, recurisvley go down each path
    def permute(self, nums):
        if len(nums) == 1:
            return [nums]
        perms = []
        self.permute_rec(nums, [], perms)
        return perms
    def permute_rec(self, remainNums, permSoFar, perms):
        if len(remainNums) == 1: # base case (optimized from len==0)
            perms.append(permSoFar+remainNums)
            return
        for i in range(len(remainNums)):
            self.permute_rec(remainNums[:i]+remainNums[i+1:], permSoFar+[remainNums[i]], perms)
    
    ### Time: dominated by bottom layer -> O(n!*1) -> O(n!) ###
    ### number of calls/branches: n - n(n-1) - ... - n! at bottom layer
    ### remainNum: if layer has n(n-1) nodes, it will take (n-2); bottom layer takes O(1)
    ### Space: O(n!) + O(n) -> O(n!) ###
    ### perms takes O(n!)
    ### call stack: # of function calls at any one time=height of tree=n - O(n) ###

    @staticmethod
    def test():
        q46 = Q46()
        print("expected [[1]] |", q46.permute([1]))
        print("expected [[1, 2], [2, 1]] |", q46.permute([1, 2]))
        print("expected [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]] |", q46.permute([1,2,3]))

# Q46.test()



class Q48: # MEDIUM | def rotate(self, matrix: List[List[int]]) -> None:
    # https://leetcode.com/problems/rotate-image/
    # You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
    # You have to rotate the image in-place, which means you have to modify the input 2D matrix directly.
    # DO NOT allocate another 2D matrix and do the rotation.

    ## orig(r, c) -> orig(c, sideLen-r-1)
    ## for each square ring, for each set of 4 symmetric values, circularly reassign
    def rotate(self, matrix):
        """
        Do not return anything, modify matrix in-place instead.
        """
        if len(matrix) == 1:
            return matrix
        side = len(matrix)
        for ringIdx in range(side//2): # top left of the ring = (ringIdx, ringIdx)
            for i in range(ringIdx, side-ringIdx-1):
                r, c = ringIdx, i
                toreplace = matrix[r][c]
                for _ in range(4): # 4 sides
                    replaced = matrix[c][side-r-1]
                    matrix[c][side-r-1] = toreplace
                    toreplace = replaced
                    r, c = c, side-r-1             
    ### Time: at max go through each cell once -> O(number of cells) ###
    ### Space: O(1) ###
        
    @staticmethod
    def test():
        q48=Q48()
        matrix = [[1]]
        q48.rotate(matrix)
        print([[1]] == matrix)

        matrix = [[1,2],[3,4]]
        q48.rotate(matrix)
        print([[3,1],[4,2]]== matrix)

        matrix = [[1,2,3],[4,5,6],[7,8,9]]
        q48.rotate(matrix)
        print([[7,4,1],[8,5,2],[9,6,3]] == matrix)

        matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
        q48.rotate(matrix)
        print([[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]] == matrix)

# Q48.test()



class Q49: # MEDIUM | def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    # https://leetcode.com/problems/group-anagrams/
    # Given an array of strings strs, group the anagrams together. You can return the answer in any order.
    # An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
    # typically using all the original letters exactly once.
    
    def groupAnagrams(self, strs):
        if len(strs) == 1:
            return [strs]
        d = {}
        for str in strs: # O(n * )
            key = ''.join(sorted(str))
            if key in d.keys():
                d[key].append(str)
            else:
                d[key] = [str]
        return list(d.values()) # O(n)
    
    ### Time: for each of n str in strs, sorted+join -> n * (slogs+s) -> O(n * slogs)###

    @staticmethod
    def test():
        q49=Q49()
        print([[""]]==q49.groupAnagrams([""]))
        print([["a"]]==q49.groupAnagrams(["a"]))
        print('expected [["bat"],["nat","tan"],["ate","eat","tea"]] |', q49.groupAnagrams(["eat","tea","tan","ate","nat","bat"]))

# Q49.test()



class Q53: # EASY | def maxSubArray(self, nums: List[int]) -> int:
    # https://leetcode.com/problems/maximum-subarray/
    # Given an integer array nums, find the contiguous subarray (containing at least one number)
    # which has the largest sum and return its sum.

    ## maxSum(int) to store the max value so far (bigger of curSum+nums[i] and nums[i])
    def maxSubArray(self, nums):
        if len(nums) == 1:
            return nums[0]
        maxSum, curSum = nums[0], nums[0] # subarray must have ≥ 1 number
        for i in range(1, len(nums)):
            curSum = max(curSum+nums[i], nums[i])
            maxSum = max(maxSum, curSum)
        return maxSum
            
    ## maxSoFar(list) to store max up until each index
    def maxSubArray_list(self, nums):
        if len(nums) == 1:
            return nums[0]
        maxSoFar = [nums[0]]
        for i in range(1, len(nums)):
            maxSoFar.append(max(maxSoFar[i-1]+nums[i], nums[i]))
        return max(maxSoFar) # O(n)
            
    ### Time(for both): traverse through list, constant at each step (+ max(), whihc is O(n)) -> O(n) ###
    ### Space: O(1) vs. O(n) ###

    @staticmethod
    def test():
        q53=Q53()
        print(1==q53.maxSubArray([1]))
        print(6==q53.maxSubArray([2, 2, 2]))
        print(4==q53.maxSubArray([2, 2, -2, 2]))
        print(6==q53.maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))
        print(23==q53.maxSubArray([5,4,-1,7,8]))

# Q53.test()



class Q55: # MEDIUM | def canJump(self, nums: List[int]) -> bool:
    # https://leetcode.com/problems/jump-game/
    # Given an array of non-negative integers nums, you are initially positioned at the first index of the array.
    # Each element in the array represents your maximum jump length at that position.
    # Determine if you are able to reach the last index.

    ## traverse nums from the back, decreasing destIdx as we find a way to reach it
    def canJump_back(self, nums):
        if len(nums) == 1:
            return True
        destIdx = len(nums)-1
        for i in range(len(nums)-1)[::-1]:
            if i+nums[i] >= destIdx:
                destIdx = i # new destination
        return destIdx==0 # arrived at the start of the list
    ### Time: O(n) ###

    ## stores max index that can be reached
    def canJump_forward(self, nums):
        if len(nums) == 1:
            return True
        maxIdx = nums[0]
        if maxIdx >= len(nums)-1:
            return True
        for i in range(1, len(nums)-1):
            if i > maxIdx: # NOTE: beyond what can be reached
                return False
            maxIdx = max(maxIdx, i + nums[i])
            if maxIdx >= len(nums)-1:
                return True
        return False
    ### Time: O(n) ###
        
    ## take a range's step one jump at a time until the range of the step >= end (similar to Q45's solution)
    def canJump_step(self, nums):
        if len(nums) == 1:
            return True
        destIdx = len(nums)-1
        l, r = 0, nums[0] # inclusive indices of range of a particular number of jumps, initialized to range of the first jump
        while r < destIdx:
            farthestIdx = r
            for i in range(l, r+1):
                if i+nums[i]>=destIdx:
                    return True
                else:
                    farthestIdx = max(farthestIdx, i+nums[i])
            if farthestIdx == r: # cannot move any further
                return False
            else:
                l, r = r, farthestIdx
        return True # r >= destIdx
    ### Time: O(n) ###

    @staticmethod
    def test():
        q55 = Q55()
        print(True==q55.canJump([1]))
        print(True==q55.canJump([0]))
        print(False==q55.canJump([0, 0]))
        print(True==q55.canJump([1, 0]))
        print(True==q55.canJump([2,3,1,1,4]))
        print(True==q55.canJump([1,1,1,1,1,1,1,1]))
        print(False==q55.canJump([3,2,1,0,4]))
    
# Q55.test()



class Q56: # MEDIUM | def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    # https://leetcode.com/problems/merge-intervals/
    # Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals,
    # and return an array of the non-overlapping intervals that cover all the intervals in the input.
    
    # sort by start_i and then traverse comparing end of interval_i to start of interval_i+1
    def merge(self, intervals):
        if len(intervals) == 1:
            return intervals
        intervals.sort(key=lambda interval:interval[0]) # sort in place by starting index - O(n log n)
        output = []
        s=0 # index of start for current output interval
        e=0 # index of end for current output interval
        for i in range(1, len(intervals)):
            if intervals[e][1]<intervals[i][0]: # can start new interval in output
                output.append([intervals[s][0], intervals[e][1]])
                s, e = i, i
            else: # overlap
                if intervals[i][1]>=intervals[e][1]:
                    e=i # keep end as the max among overlapping intervals
        output.append([intervals[s][0], intervals[e][1]])
        return output
    ### Time: n=len(intervals); O(n log n) + O(n) -> O(n log n) ###
    ### Space: output-O(n) + s,e - O(1) -> O(n) ###
    
    @staticmethod
    def test():
        q56=Q56()
        print([[0,6]] == q56.merge([[0,6]]))
        print([[0,0], [1,1], [3,6]] == q56.merge([[0,0], [1,1], [3,6]]))
        print([[0,0], [1,1]] == q56.merge([[0,0], [1,1], [1,1]]))
        print([[1,5]] == q56.merge([[1,4], [4,5]]))
        print([[1,2]] == q56.merge([[1,2],[1,2]]))
        print([[1,3]] == q56.merge([[1,2],[1,3]]))
        print([[1,3]] == q56.merge([[1,3],[1,2]]))
        print([[1,5]] == q56.merge([[1,5],[2,3]])) 
        print([[1,5]] == q56.merge([[1,5],[2,3],[4,5]])) 
        print([[1,6]] == q56.merge([[2,3],[4,6],[1,5]])) 
        print([[1,6],[10,11]] == q56.merge([[10,11],[2,3],[1,5],[4,6]])) 
        print([[10, 28]] == q56.merge([[10,12],[12,25],[22,28]]))
        print([[10, 25]] == q56.merge([[10,12],[12,25],[22,24]]))
        print([[1,6], [8,10], [15,18]] == q56.merge([[1,3],[2,6],[8,10],[15,18]]))
        print([[1,6], [8,10], [15,18]] == q56.merge([[2,6],[8,10],[15,18],[1,3]]))  

# Q56.test()



class Q62: # MEDIUM |  def uniquePaths(self, m: int, n: int) -> int:
    # https://leetcode.com/problems/unique-paths/
    # A robot is located at the top-left corner of a m x n grid.
    # The robot can only move either down or right at any point in time. 
    # The robot is trying to reach the bottom-right corner of the grid
    # How many possible unique paths are there?
    
    ## Math approach: a path can be represented as D D R R R -> 2 ways to think about it
        ## 1. permutations with repeats: arrange (m-1)D's and (n-1)R's -> [(m-1+n-1)!]/[(m-1)!*(n-1)!]
        ## 2. combinations: select (m-1) or (n-1) positions out of (m-1+n-1) -> [(m-1+n-1)!]/[(m-1)!*(n-1)!]
    
    ## optimized dp: bottom row and rightmost column always 1
    def uniquePaths(self, m, n):
        dp=[ [1 for _ in range(n)] for _ in range(m)]
        for m_i in range(1, m): # row by row from bottom up
            for n_i in range(1, n): # right to left of each row
                    dp[m_i][n_i]= dp[m_i-1][n_i] + dp[m_i][n_i-1] ## NOTE: core of dp: down + right
        return dp[m-1][n-1]
    ## dp[m][n]: number of ways to reach bottom right starting at m above and n to the left
    ## NOTE: core of dp -> dp[m][n] = 1*ways from 1 down ([m-1][n]) + 1*ways from 1 right ([m][n-1])
    ## To fill dp[m][n]: need to fill [<m, <n] - all other cells in the rectangle where m,n is top left
    def uniquePaths_dpinit(self, m, n):
        # m: rows/height/downs, n:columns/width/right
        dp=[ [0 for _ in range(n)] for _ in range(m)]
        dp[0][0] = 1
        for m_i in range(m): # row by row from bottom up
            for n_i in range(n): # right to left of each row
                if not (m_i==0 and n_i==0):
                    down = dp[m_i-1][n_i] if m_i > 0 else 0 # 1 way to go down
                    right = dp[m_i][n_i-1] if n_i > 0 else 0 # 1 way to go right
                    dp[m_i][n_i]= down + right
        return dp[m-1][n-1]
    ### Time: O(mn) - iterate through each cell once ### 
    ### Space: O(mn) - dp ###

    @staticmethod
    def test():
        q62 = Q62()
        print(1==q62.uniquePaths(1,1))
        print(3==q62.uniquePaths(3,2))
        print(3==q62.uniquePaths(2,3))
        print(6==q62.uniquePaths(3,3))
        print(28==q62.uniquePaths(7,3))

# Q62.test()


class Q64: # MEDIUM | def minPathSum(self, grid: List[List[int]]) -> int:
    # https://leetcode.com/problems/minimum-path-sum/
    # Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right,
    # which minimizes the sum of all numbers along its path.
    # Note: You can only move either down or right at any point in time.

    # RELATED: similar set up to Q62


    ## dp[r][c]=min path sum from (0, 0) -> can directly use grid for it (modify in place)
    ## fill all to (r, c)'s left and top first: (r, c) is like bottom right of a smaller rectangle
    ## NOTE: core of dp - dp[r][c] = min(dp[r][c-1], dp[r-1][c]) + grid[r][c]
    def minPathSum(self, grid):
        for r in range(len(grid)): # row by row from top down
            for c in range(len(grid[0])): # left to right
                if r == 0 and c==0:
                    continue
                if r==0:
                    grid[r][c] += grid[r][c-1]
                elif c==0:
                    grid[r][c] += grid[r-1][c]
                else:
                    grid[r][c] += min(grid[r][c-1], grid[r-1][c]) ## NOTE: core of dp - min(left, top)
        return grid[len(grid)-1][len(grid[0])-1]
    
    ### Time: traverse through each node once * O(1) at each traversal ->  O(mn) ###
    ### Space: O(1) ###

    @staticmethod
    def test():
        q64 = Q64()
        print(1==q64.minPathSum([[1]]))
        print(7==q64.minPathSum([[1,3,1],[1,5,1],[4,2,1]]))
        print(12==q64.minPathSum([[1,2,3],[4,5,6]]))
        print(10==q64.minPathSum([[4,5,6],[1,2,3]]))

# Q64.test()



class Q70: # EASY | def climbStairs(self, n: int) -> int:
    # https://leetcode.com/problems/climbing-stairs/
    # You are climbing a staircase. It takes n steps to reach the top.
    # Each time you can either climb 1 or 2 steps. 
    # In how many distinct ways can you climb to the top?


    ## NOTE: core of dp -> ways(n) = ways(n-1) + ways(n-2)
    ## To reach step n: reach n-1 then 1 step + reach n-1 then 2 steps
    def climbStairs(self, n):
        if n < 4:
            return n
        first, second = 2, 3 # for n=2, n=3 (since we only need to keep the last two)
        for n in range(4, n):
            first, second = second, first+second ## NOTE: core of dp
        return first+second
        
    ### Time: (n-4) loops * O(1) per loop -> O(n)  ### 
    ### Memory: first + second + n -> O(1)

    @staticmethod
    def test():
        q70 = Q70()
        print(1==q70.climbStairs(1))
        print(2==q70.climbStairs(2))
        print(3==q70.climbStairs(3))
        print(5==q70.climbStairs(4))
        print(8==q70.climbStairs(5))
        print(987==q70.climbStairs(15))

# Q70.test()



class Q72: # HARD | def minDistance(self, word1: str, word2: str) -> int:
    # https://leetcode.com/problems/edit-distance/
    # Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.
    # You have the following three operations permitted on a word:
        # Insert a character
        # Delete a character
        # Replace a character

    # NOTE about edit distance:
        # At each char, if more of both words: 0 operation {go to next char}(match) or 1 opertaion {insert, delete, replace}(mismatch)
        # The 3 operations all correspond to a change of the index pointers of the two words
    
    ## 1. DP: remember minDistance(w1i,w2i) for (w1i,w2i) bottom up (smaller -> larger string)
    ## NOTE: core of dp - dp[(w1i,w2i)]=topleft if match else 1+ min(topleft, left, top)
    # filling from top left to bottom right; dp[w2ct][w1ct]: w2ct chars of w2 remains to be matched and w1ct chars of w2 remain
    def minDistance_dp(self, word1, word2):
        dp = [ [0 for _ in range(len(word1)+1)] for _ in range(len(word2)+1)] # each row: 1 particular remaning numChar for w2
        for w1ct in range(1, len(word1)+1):
            dp[0][w1ct] = w1ct
        for w2ct in range(1, len(word2)+1):
            dp[w2ct][0] = w2ct
        
        for w2ct in range(1, len(word2)+1): # row by row top down
            for w1ct in range(1, len(word1)+1): # left to right
                if word1[len(word1)-w1ct] == word2[len(word2)-w2ct]:
                    dp[w2ct][w1ct] = dp[w2ct-1][w1ct-1] ## NOTE
                else:
                    dp[w2ct][w1ct] = 1 + min(dp[w2ct-1][w1ct-1], dp[w2ct][w1ct-1], dp[w2ct-1][w1ct]) ## NOTE
        return dp[len(word2)][len(word1)] # bottom right
    ### Time: traverse through each cell * O(1) at each cell -> O(len(w1) * len(w2)) ###
    ### Space: dp -> O(len(w1) * len(w2)) ###


    ## 2. Memoiaztion: remember minDistance(w1i,w2i) for (w1i,w2i) top down: if remember return, else fill through recursion
    def minDistance_memoization(self, word1, word2):
        mem = {} # {(w1i,w2i):edit distance} - mutable/pass by reference
        self.minDistance_memHelper(word1, word2, 0, 0, mem)
        return mem[(0,0)]

    def minDistance_memHelper(self, w1, w2, w1i, w2i, mem):
        if (w1i,w2i) in mem:
            return mem[(w1i,w2i)]
        else: # not in mem: fill through recursion
            # base case
            if w1i==len(w1) and w2i==len(w2):
                ans = 0
            elif w1i==len(w1): # more of w2 remains
                ans =  len(w2)-w2i
            elif w2i==len(w2): # more of w1 remains
                ans = len(w1)-w1i
            # general case: more of both words remain
            elif w1[w1i]==w2[w2i]:
                ans = self.minDistance_memHelper(w1,w2,w1i+1,w2i+1, mem)
            else: # all operations done to word1
                insert = 1 + self.minDistance_memHelper(w1, w2, w1i, w2i+1, mem)
                delete = 1 + self.minDistance_memHelper(w1, w2, w1i+1, w2i, mem)
                replace = 1 + self.minDistance_memHelper(w1, w2, w1i+1, w2i+1, mem)
                ans = min(insert, delete, replace)
            mem[(w1i,w2i)] = ans
            return mem[(w1i,w2i)]
            

    ## 3. Recursion: at each mismatched char, start 3 recursive branches, then find min of these 3 branches
    ## A lot of overlapping operations
    def minDistance_recursive(self, word1, word2):
        return self.minDistance_recHelper(word1, word2, 0, 0)
        
    def minDistance_recHelper(self, w1, w2, w1i, w2i):
        # base case
        if w1i==len(w1) and w2i==len(w2):
            return 0
        elif w1i==len(w1): # more of w2 remains
            return len(w2)-w2i
        elif w2i==len(w2): # more of w1 remains
            return len(w1)-w1i
        # general case: more of both words remain
        if w1[w1i]==w2[w2i]:
            return self.minDistance_recHelper(w1,w2,w1i+1,w2i+1)
        else: # all operations done to word1
            insert = 1 + self.minDistance_recHelper(w1, w2, w1i, w2i+1)
            delete = 1 + self.minDistance_recHelper(w1, w2, w1i+1, w2i)
            replace = 1 + self.minDistance_recHelper(w1, w2, w1i+1, w2i+1)
            return min(insert, delete, replace)

    
    @staticmethod
    def test():
        q72 = Q72()
        print(0==q72.minDistance_dp("",""))
        print(1==q72.minDistance_dp("","a"))
        print(2==q72.minDistance_dp("ab",""))
        print(3==q72.minDistance_dp("horse","ros"))
        print(5==q72.minDistance_dp("intention","execution"))

# Q72.test()



### REVIEW: sort
class Q75: # MEDIUM | def sortColors(self, nums: List[int]) -> None:
    # https://leetcode.com/problems/sort-colors/
    # Given an array nums with n objects colored red, white, or blue, sort them in-place so that 
    # objects of the same color are adjacent, with the colors in the order red, white, and blue.
    # We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.
    # You must solve this problem without using the library's sort function.

    ## one pass: keep track of next0pos(rightmost) and next2pos(leftmost)
    ## REVIEW
    def sortColors(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) == 1:
            return
        next0pos = 0 # <= curr, all to its left == 0
        next2pos = len(nums)-1 # >= curr, all to its right == 2
        curr = 0
        while curr <= next2pos: # when =, knows all to the right == 2
            if nums[curr] == 0: # move 0 to the block of 0s at the left
                nums[curr], nums[next0pos] = nums[next0pos], nums[curr]
                next0pos += 1
                curr += 1 # new nums[curr] only possible to be 0 or 1
            elif nums[curr] == 1:
                curr += 1
            elif nums[curr] == 2:
                nums[curr], nums[next2pos] = nums[next2pos], nums[curr]
                next2pos -= 1 # do not know what nums[curr] may be, cannot += 1
    ### Time: sinlge pass -> O(n) ###
    ### Space: 3 pointser -> O(1) ###
    
    ## brute force: at each char, find min in the remaining ones
    def sortColors_assignEachPos(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) == 1:
            return
        for i in range(len(nums)):
            min = nums[i]
            minIdx = i
            for j in range(i, len(nums)):
                if nums[j] < min:
                    minIdx = j
                    min = nums[j]
            nums[i], nums[minIdx] = nums[minIdx], nums[i]
    ### Time: n+n-1+n-2+...+2+1 -> (n+1)*(n)/2 -> O(n^2) ###
    ### Space: i, min, minIdx, j -> O(1) ###
        
    @staticmethod
    def test():
        q75=Q75()
        nums = [1]
        q75.sortColors(nums)
        print([1]==nums)
        nums = [0,1,2]
        q75.sortColors(nums)
        print([0,1,2]==nums)
        nums = [2,0,1]
        q75.sortColors(nums)
        print([0,1,2]==nums)
        nums = [1,2,0]
        q75.sortColors(nums)
        print([0,1,2]==nums)
        nums = [2,1,0]
        q75.sortColors(nums)
        print([0,1,2]==nums)

        nums = [0,0,0]
        q75.sortColors(nums)
        print([0,0,0]==nums)
        nums = [1,1,1]
        q75.sortColors(nums)
        print([1,1,1]==nums)
        nums = [2,2,2]
        q75.sortColors(nums)
        print([2,2,2]==nums)

        nums = [2,0,2,2]
        q75.sortColors(nums)
        print([0,2,2,2]==nums)
        nums = [2,2,2,1]
        q75.sortColors(nums)
        print([1,2,2,2]==nums)
        nums = [0,1,0,1]
        q75.sortColors(nums)
        print([0,0,1,1]==nums)
        
        nums = [2,0,2,1,1,0]
        q75.sortColors(nums)
        print([0,0,1,1,2,2]==nums)
        nums = [1,2,0,1,2,0,0,0,2,1,0]
        q75.sortColors(nums)
        print([0,0,0,0,0,1,1,1,2,2,2]==nums)

# Q75.test()



### REVIEW: sliding window
class Q76: # HARD | minWindow(self, s: str, t: str) -> str:
    # https://leetcode.com/problems/minimum-window-substring/
    # Given two strings s and t of lengths m and n respectively, return the minimum window substring of s 
    # such that every character in t (including duplicates) is included in the window. 
    # If there is no such substring, return the empty string "".
    # The testcases will be generated such that the answer is unique.
    # A substring is a contiguous sequence of characters within the string.

    ## Improved Sliding Window: for each char, 1) expand, 2) shrink if possible, 3) update min
    ## 1) currWin: {char:[idx1, idx2]} -> {char:count} since always need to iterate to shrink anyways
    ## 2) Cache checking of complete with metCharCt: if a char's count is newly met, metCharCt+=1 (checking becomes O(1))
    def minWindow(self, s, t):
        # 1. represent t as dictionary {char: count}
        vocab = {}
        for char in t:
            vocab[char] = vocab[char]+1 if (char in vocab) else 1
        # 2. keep a curWin as dictionary {char in vocab: count}
        l, r = 0,0
        currWin = {} 
        minLen, minl, minr = -1, -1, -1
        metCharCt = 0 # monotonically increase (once complete, always complete)
        while r < len(s):
            if s[r] in vocab:
                currWin[s[r]] = currWin[s[r]]+1 if (s[r] in currWin) else 1
                if currWin[s[r]] == vocab[s[r]]: # first time meeting the requirement
                    metCharCt += 1
                if currWin[s[r]] > vocab[s[r]] and s[r]==s[l]: # exceed and can shrink
                    l+=1
                    currWin[s[r]]-=1
                    while l<r:
                        if s[l] in vocab:
                            if currWin[s[l]] <= vocab[s[l]]:
                                break # cannot shrink further
                            else:
                                currWin[s[l]]-=1
                        l+=1
            else:
                if len(currWin) == 0: # finding initial left
                    l+=1
            # update min
            if metCharCt == len(vocab) and (minLen==-1 or r-l+1<minLen):
                minLen = r-l+1
                minl, minr = l,r
            r+=1
        if minLen == -1:
            return ""
        else:
            return s[minl:minr+1]
    ### Time: vocab = O(t_len) + add and remove each char (max twice) = O(s_len) -> O(t+s) ###
    ### Space: vocab = O(t) , currWin = O(t) -> O(t) ### 


    ## Sliding Window: for each char, 1) expand, 2) shrink if possible, 3) update min
    def minWindow_badSlidingWin(self, s, t):
        # 1. represent t as dictionary {char: count}
        vocab = {}
        for char in t:
            vocab[char] = vocab[char]+1 if (char in vocab) else 1
        # 2. keep a curWin as dictionary {char in vocab: [idx1, idx2]}
        l, r = 0,0
        currWin = {}
        minLen, minl, minr = -1, -1, -1
        while r < len(s):
            if s[r] in vocab:
                if s[r] in currWin:
                    currWin[s[r]].append(r)
                    if len(currWin[s[r]]) > vocab[s[r]] and currWin[s[r]][0]==l: # exceed and can shrink
                        l+=1
                        currWin[s[r]].pop(0)
                        while l<r:
                            if s[l] in vocab:
                                if len(currWin[s[l]]) <= vocab[s[l]]:
                                    break # cannot shrink further
                                else:
                                    currWin[s[l]].pop(0)
                            l+=1
                else:
                    currWin[s[r]] = [r] # won't exceed count
            else:
                if len(currWin) == 0: # finding initial left
                    l+=1
                    
            # update min
            complete = True
            for char in vocab:
                if char not in currWin or len(currWin[char]) < vocab[char]:
                    complete = False
            if complete and (minLen==-1 or r-l+1<minLen):
                minLen = r-l+1
                minl, minr = l,r
            r+=1
        if minLen == -1:
            return ""
        else:
            return s[minl:minr+1] 
    ### Time: vocab = O(t_len) + add and remove each char (max twice) = O(s_len) + each char check complete O(s*t) -> O(st) ###
    ### Space: vocab = O(t) , currWin = O(s) -> O(s+t) ### 

    @staticmethod
    def test():
        q76 = Q76()
        print("a"==q76.minWindow("a","a"))
        print(""==q76.minWindow("a","b"))
        print(""==q76.minWindow("a","aa"))
        print("BANC"==q76.minWindow("ADOBECODEBANC","ABC"))
        print("BAZA"==q76.minWindow("BAZA","BAA"))
        print("BAZA"==q76.minWindow("BAZA","ABA"))
        print("BAZA"==q76.minWindow("ZZBAZAZ","BAA"))
        print("AAB"==q76.minWindow("BAZAABZAA","BAA"))
        print("BA"==q76.minWindow("AZBBA","AB"))
        print("BA"==q76.minWindow("ZBAZBBZA","AB"))

# Q76.test()



class Q78: # MEDIUM | def subsets(self, nums: List[int]) -> List[List[int]]:
    # https://leetcode.com/problems/subsets/
    # Given an integer array nums of unique elements, return all possible subsets (the power set).
    # The solution set must not contain duplicate subsets. Return the solution in any order.

    ## absent/present  for each num -> 2^(nums_len) -> recursion
    def subsets(self, nums):
        ps = [ [], [nums[0]]]
        for i in range(1, len(nums)):
            for subset in ps.copy(): # avoid infinite loop
                ps.append(subset+[nums[i]]) # don't want to modify subset in place
                # if absent, does not modify original subset
        return ps
        
    ### Time: iterate through each of 2^nums_len subsets once * O(1) for each -> O(2^nums_len) ###
    ### Memory: ps is O(2^nums_len) + ps.copy() is bounded by 2^nums_len-1 ->  O(2^nums_len) ###
        
    @staticmethod
    def test():
        q78 = Q78()
        print([[],[1]]==q78.subsets([1]))
        print([[],[1],[2],[1,2]]==q78.subsets([1,2]))
        print([[],[1],[2],[1,2],[-3],[1,-3],[2,-3],[1,2,-3]]==q78.subsets([1,2,-3]))

# Q78.test()



class Q79: # MEDIUM | def exist(self, board: List[List[str]], word: str) -> bool:
    # https://leetcode.com/problems/word-search/
    # Given an m x n grid of characters board and a string word, return true if word exists in the grid.
    # The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are 
    # horizontally or vertically neighboring. The same letter cell may not be used more than once.

    ## improved dfs: if cur cell matches: go on with word[1:], else return False and go to next start/down next path
    ### 1. visited set -> modifying board in place
    ### 2. word[1:] -> index of word
    ### 3. put checking loop of matching_char into exist(); r,c not last matching index, but curr to check if match
    def exist(self, board, word):
        for r in range(len(board)):
            for c in range(len(board[0])):
                if self.exist_dfs(board, word, 0, r, c):
                    return True
        return False

    def exist_dfs(self, board, word, wIdx, r, c):
        '''r, c are the indices to check'''
        if wIdx == len(word):
            return True
        else:
            if word[wIdx] != board[r][c]:
                return False
            else:
                board[r][c]="_" # to avoid using the same cell more than once
                if (r>0 and self.exist_dfs(board, word, wIdx+1, r-1, c)) or \
                    (r<len(board)-1 and self.exist_dfs(board, word, wIdx+1, r+1, c)) or \
                    (c>0 and self.exist_dfs(board, word, wIdx+1, r, c-1)) or \
                    (c<len(board[0])-1 and self.exist_dfs(board, word, wIdx+1, r, c+1)) or \
                    (wIdx+1==len(word)): # for len(board) and len(board[0]) = 1
                    return True
                board[r][c]=word[wIdx] # changing it back
                return False
    ### Time: mn starting char for recursion * (branches=4^depth=word_len) calls *  O(1) per call -> O(mn * 4^word_len) ###
    ### Space: heap = O(1) since visited is modifying board in place; call stack = depth=word_len -> O(word_len) ###

    ## dfs: find matching first cell, then recursively check 4 adjcent directions for each cell
    def exist_initialdfs(self, board, word):
        r, c = self.matching_char(board, 0, -1, word[0])     
        while r is not None:
            if self.exist_rec(board, word[1:], r, c, {(r, c)}):
                return True # else continue looping
            r, c = self.matching_char(board, r, c, word[0])
        return False
            
    def exist_rec(self, board, word, r, c, visited):
        '''r, c are the indices of the last matching character'''
        if len(word) == 0:
            return True
        else:
            if r != 0 and ((r-1,c) not in visited) and board[r-1][c] == word[0]: # top
                visited.add((r-1,c))
                if (self.exist_rec(board, word[1:], r-1, c, visited)):
                    return True
                visited.remove((r-1,c))
            if r != len(board)-1 and ((r+1,c) not in visited) and board[r+1][c] == word[0]: # bottom
                visited.add((r+1,c))
                if (self.exist_rec(board, word[1:], r+1, c, visited)):
                    return True
                visited.remove((r+1,c))
            if c != 0 and ((r,c-1) not in visited) and board[r][c-1] == word[0]: # left
                visited.add((r,c-1))
                if (self.exist_rec(board, word[1:], r, c-1, visited)):
                    return True
                visited.remove((r,c-1))
            if c != len(board[0])-1 and ((r, c+1) not in visited) and board[r][c+1] == word[0]: # right
                visited.add((r,c+1))
                if (self.exist_rec(board, word[1:], r, c+1, visited)):
                    return True
                visited.remove((r,c+1))
            return False
        
    def matching_char(self, board, r_idx, c_idx, target):
        '''r_idx, and c_idx are the indices of the last found target'''
        # check row r_idx
        for c in range(c_idx+1, len(board[0])):
            if board[r_idx][c] == target:
                return r_idx, c
        # check the other rows
        for r in range(r_idx+1, len(board)):
            for c in range(len(board[0])):
                if board[r][c] == target:
                    return r, c
        return None, None
    
    @staticmethod
    def test():
        q79 = Q79()
        print(True == q79.exist([["A"]], "A"))
        print(True == q79.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "C"))
        print(True == q79.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABC"))
        print(True == q79.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "CCB"))
        print(True == q79.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SA"))
        print(True == q79.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED"))
        print(True == q79.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SEE"))
        print(True == q79.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SEECCE"))
        print(False == q79.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCD"))
        print(False == q79.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCB"))
    
# Q79.test()



### REVIEW: difficult solution
class Q84: # HARD | def largestRectangleArea(self, heights: List[int]) -> int:
    # https://leetcode.com/problems/largest-rectangle-in-histogram/
    # Given an array of integers heights representing the histogram's bar height where 
    # the width of each bar is 1, return the area of the largest rectangle in the histogram.

    # RELATED: Q85
    
    ## Find maximum width for each bar's height (naive: linear search for each -> O(n^2))
    ## l,r bound=first bar in that direction shorter than the bar, makes use of sequential location info -> stack
    ## Within the width (between stack[-1] and curr), the bar is the minimum height
    ## Once found a bar that is shorter, the bar's height's rectangle is done -> pop
    def largestRectangleArea(self, heights):
        maxArea = 0
        stack = [-1] # stack of indices of candidate heights whose rectangle is not yet determined
        for i in range(len(heights)): # i is like the right bound
            # check whether current height terminates any candidate
            while stack[-1]!=-1 and heights[i] < heights[stack[-1]]: # current height terminates last candidate
                heightIdx = stack.pop()
                width = i-stack[-1]-1
                maxArea = max(maxArea, width*heights[heightIdx])
            stack.append(i)
        for _ in range(len(stack)-1): # deal with leftovers in stack
            heightIdx = stack.pop()
            width = len(heights)-stack[-1]-1
            maxArea = max(maxArea, width*heights[heightIdx])  

        return maxArea
    ### Time: each bar added once (start considering it) and popped once (found its rect) -> O(num_bars) ###
    ### Space: stack bounded by O(num_bars) -> O(num_bars) ###



# REVIEW: left/right's representation + how dp is constructed
class Q85: # HARD | def maximalRectangle(self, matrix: List[List[str]]) -> int:
    # https://leetcode.com/problems/maximal-rectangle/
    # Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle
    # containing only 1's and return its area.

    ## 1. For each row, find the width correspondining to each column's height
    ## left[i] = left bound/leftmost idx(inclusive) for height[i] where all h in between ≥ height[i]; similar for right
    ## dp: 1. height[i] in row = height[i] in row above + 1 if is 1, else if is 0, =0
        #  2. left[i] constrained by {unchanged from row-1 if this row has all 1 in area below, where consecutive 1 start this row}
        #     If at 0 / height[i]=0: left[i]=0 and right[i]=len(row)-1 -> doesn't affect row below
    def maximalRectangle(self, matrix):
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0
        maxArea = 0
        # don't need to keep row above as update maxArea after traversal through each row
        rowLen = len(matrix[0])
        left = [0 for _ in range(rowLen)]
        right = [rowLen-1 for _ in range(rowLen)]
        height = [0 for _ in range(rowLen)]
        for r in range(len(matrix)):
            lconsec1,rconsec1 = 0, rowLen-1 # where consecutive ones start 
            for c in range(rowLen):
                if matrix[r][c]=='0':
                    height[c]=0
                    left[c]=0
                    lconsec1 = c+1
                else:
                    height[c]+=1
                    left[c]=max(left[c], lconsec1) # dp CORE: 2 constraints of row above and this row
            for c in reversed(list(range(rowLen))):
                if matrix[r][c]=='0':
                    right[c] = rowLen-1
                    rconsec1 = c-1
                else:
                    right[c] = min(right[c], rconsec1) # dp CORE: 2 constraints of row above and this row
            for c in range(rowLen):
                maxArea = max(maxArea, height[c]*(right[c]-left[c]+1))
        return maxArea
    ### Time: for each row, iterate through ele in row 3 times = O(nRow * 3*nCol) -> O(nRow * nCol) ###
    ### Space: left, right, height = O(3nCol) -> O(nCol) ###

    ## 2. Same as largestRectangleArea: each row=ground with histogram, whose height=count of consecutive 1s from top ending at that row 
    def maximalRectangle_largestRectangleArea(self, matrix):
        if not matrix or not matrix[0]:
            return 0
        n = len(matrix[0])
        height = [0] * (n + 1)
        ans = 0
        for row in matrix:
            for i in xrange(n):
                height[i] = height[i] + 1 if row[i] == '1' else 0
            stack = [-1]
            for i in xrange(n + 1):
                while height[i] < height[stack[-1]]:
                    h = height[stack.pop()]
                    w = i - 1 - stack[-1]
                    ans = max(ans, h * w)
                stack.append(i)
        return ans
    ### Time: for each row, iterate through ele in row 2 times = O(nRow * 2*nCol) -> O(nRow * nCol) ###
    ### Space: height, stack = O(3nCol) -> O(nCol) ###

    @staticmethod
    def test():
        print("### Testing for Q85 ###")
        q85=Q85()
        print(0==q85.maximalRectangle([]))
        print(0==q85.maximalRectangle([[]]))
        print(0==q85.maximalRectangle([["0"]]))
        print(1==q85.maximalRectangle([["1"]]))
        print(2==q85.maximalRectangle([["0","1","1","0","0"]]))
        print(6==q85.maximalRectangle([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]))

# Q85.test()



class Q96: # MEDIUM | def numTrees(self, n: int) -> int:
    # https://leetcode.com/problems/unique-binary-search-trees/
    # Given an integer n, return the number of structurally unique BST's (binary search trees) 
    # which has exactly n nodes of unique values from 1 to n.
    
    ## dp: [number of unique BST for index]
    ## dp[n] = sum(when each node is root=dp[leftTreeSize] * dp[rightTreeSize])
    def numTrees(self, n):
        dp = [1, 1] # 1 way to build a bst with 0 and 1 node
        for numN in range(2, n+1):
            numBST = 0
            for root in range(1, numN+1):
                numBST+= dp[root-1] * dp[numN-root] # leftTreeSize (node<root) * rightTreeSize (node>root)
            dp.append(numBST) # at index numN
        return dp[n]

    ### Time: 2 + 3 + ... + n ≈ (1+n)*n/2 -> O(n^2) ###
    ### Space: dp - O(n) ###
                
    @staticmethod
    def test():
        q96 = Q96()
        print(1 == q96.numTrees(1))
        print(2 == q96.numTrees(2))
        print(5 == q96.numTrees(3))
        print(14 == q96.numTrees(4))

# Q96.test()
        


# CTCI 4.5
class Q98: # MEDIUM | def isValidBST(self, root: TreeNode) -> bool:
    # https://leetcode.com/problems/validate-binary-search-tree/
    # Given the root of a binary tree, determine if it is a valid binary search tree (BST).
    # A valid BST is defined as follows:
        # The left subtree of a node contains only nodes with keys less than the node's key.
        # The right subtree of a node contains only nodes with keys greater than the node's key.
        # Both the left and right subtrees must also be binary search trees.

 
    
    def isValidBST(self, root):
        return self.isValidBST_rec(root, None, None)  # None so no min for left subtree or max for right subtree
    
    def isValidBST_rec(self, root, min, max):
        if (min is not None and root.val <= min) or (max is not None and root.val >= max):
            return False
        left, right = True, True
        # saves function calls
        if root.left: left = self.isValidBST_rec(root.left, min, root.val)
        if root.right: right = self.isValidBST_rec(root.right, root.val, max)
        return (left and right)
        
    ### Time: number of calls = number of nodes O(n)  * O(1) per call -> O(n) ###
    
    @staticmethod
    def test():
        q98 = Q98()
        print(True == q98.isValidBST(TreeNode()))
        n1 = TreeNode(1)
        n2 = TreeNode(2)
        n3 = TreeNode(3)
        n4 = TreeNode(4)
        n5 = TreeNode(5)
        n4.left = n2
        n2.left = n1
        n2.right = n3
        n4.right = n5
        print(True == q98.isValidBST(n4))
        n6 = TreeNode(6)
        n3.right=n6
        print(False == q98.isValidBST(n4))

# Q98.test()



class Q101: # EASY | def isSymmetric(self, root: TreeNode) -> bool:
    # https://leetcode.com/problems/symmetric-tree/
    # Given the root of a binary tree, check whether it is a mirror of itself
    # (i.e., symmetric around its center).

    ## going down both branches simultaneously, comparing the two
    def isSymmetric(self, root):
        if root.left and root.right:
            return self.isSymmetric_rec(root.left, root.right)
        elif root.left or root.right:
            return False
        else: # one node
            return True
    def isSymmetric_rec(self, node1, node2): # only called for not None
        if node1.val != node2.val:
            return False
        outter, inner = True, True # both leaves -> True
        if node1.left or node2.right:
            if node1.left and node2.right:
                outter = self.isSymmetric_rec(node1.left, node2.right)
            else:
                return False # structurally wrong
        if node1.right or node2.left:
            if node1.right and node2.left:
                inner = self.isSymmetric_rec(node1.right, node2.left)
            else:
                return False # structurally wrong
        return outter and inner
        
    ### Time: each node traversed once -> O(number of nodes) ###
    ### Space: call stack: tree depth -> O(log2n) ###
        
    @staticmethod
    def test():
        q101=Q101()
        print("Q101")
        print(True == q101.isSymmetric(TreeNode()))
        n1 = TreeNode(1)
        n2 = TreeNode(2)
        n22 = TreeNode(2)
        n3 = TreeNode(3)
        n32 = TreeNode(3)
        n4 = TreeNode(4)
        n42 = TreeNode(4)
        n1.left = n2
        n1.right = n22
        n2.left = n3
        n2.right = n4
        n22.left = n42
        n22.right = n32
        print(True == q101.isSymmetric(n1))
        n4.right = TreeNode(-100)
        print(False == q101.isSymmetric(n1))
        n4.right = None
        n22.left = n32
        n22.right = n42
        print(False == q101.isSymmetric(n1))

# Q101.test()



# Definition for a binary tree node.
class Q102: # MEDIUM | def levelOrder(self, root: TreeNode) -> List[List[int]]:
    # https://leetcode.com/problems/binary-tree-level-order-traversal/
    # Given the root of a binary tree, return the level order traversal of
    # its nodes' values. (i.e., from left to right, level by level).

    ## BFS: queue (FIFO) of [[level1 nodes], [level2 nodes]]
    def levelOrder(self, root):
        if root is None:
            return []
        levelQueue = [root] # storing one level at a time
        out = []
        while levelQueue:
            newQueue, levelout = [], []
            for node in levelQueue: # iterating: first in first out
                # do not check unvisited here because binary tree doesn't have loops 
                levelout.append(node.val)
                if node.left: newQueue.append(node.left)
                if node.right: newQueue.append(node.right)
            levelQueue = newQueue
            out.append(levelout)
        return out
    
    ### Time: visit each node once -> O(num_nodes) ###
    ### Space: out/levelout=O(num_nodes) + newQueue/levelQueue=2*O(one level nodes)=2*O(num_nodes) -> O(num_nodes) ###

    @staticmethod
    def test():
        print("Q102 test():")
        q102 = Q102()
        print([] == q102.levelOrder(None))
        print([[-100]] == q102.levelOrder(TreeNode(-100)))
        n1 = TreeNode(1)
        n2 = TreeNode(2)
        n3 = TreeNode(3)
        n4 = TreeNode(4)
        n5 = TreeNode(5)
        n4.left = n2
        n2.left = n1
        n2.right = n3
        n4.right = n5
        print([[4],[2,5],[1,3]] == q102.levelOrder(n4))
        n6 = TreeNode(6)
        n3.right=n6
        print([[4],[2,5],[1,3], [6]] == q102.levelOrder(n4))

# Q102.test()



# REVIEW: common question idea, did not think of optimal solution independently, simple optimal
class Q128: # MEDIUM | def longestConsecutive(self, nums: List[int]) -> int:
    # https://leetcode.com/problems/longest-consecutive-sequence/
    # Given an unsorted array of integers nums, return the length of the 
    # longest consecutive elements sequence.
    # You must write an algorithm that runs in O(n) time.

    ## Iterate once for each sequence contained in nums: traverse nums only if num is start of a seq
    ## To check if num is start: only need to check if num-1 is in nums
    def longestConsecutive(self, nums):
        if len(nums) == 0:
            return 0
        maxLen = 0
        num_set = set(nums) # also removes duplicates
        for num in num_set:
            if num-1 not in num_set: # start of a sequence
                seqLen, curNum = 1, num+1
                while curNum in num_set:
                    seqLen+=1
                    curNum+=1
                maxLen=max(maxLen, seqLen)
        return maxLen
    ### Time: traverse all seqs contained in nums once and each ele only in 1 seq -> O(n) ###
    ### Space: num_set -> O(n) ###

    @staticmethod
    def test():
        print("Q128")
        q128 = Q128()
        print("1", 0==q128.longestConsecutive([]))
        print("2", 4==q128.longestConsecutive([100,4,200,1,3,2]))
        print("3", 4==q128.longestConsecutive([100,4,101,1,3,2]))
        print("4", 4==q128.longestConsecutive([100,3,200,2,1,300,4]))
        print("5", 4==q128.longestConsecutive([100,-3,200,-2,-1,300,-4]))
        print("6", 9==q128.longestConsecutive([0,3,7,2,5,8,4,6,0,1]))
        print("7", 3==q128.longestConsecutive([3,-100,1,-200,2,-200,1]))

# Q128.test()


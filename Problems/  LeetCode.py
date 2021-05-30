# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
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

# Q1.test()


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
        for ele1Idx in range(0, len(nums)-2): #last 2 num don't have enough num to its right
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
                        decreaseStack.pop()
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
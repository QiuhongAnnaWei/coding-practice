## Sorting Algorithms

def bubble_sort(nums):
    swapped = True # set to True so the loop runs at least once
    while swapped: # False (termination) if last loop had no swap
        swapped = False
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i] # Swap the elements
                swapped = True # Set the flag to True so we'll loop again

# random_list_of_nums = [5, 2, 1, 3, 4]
# print(random_list_of_nums)
# bubble_sort(random_list_of_nums)
# print(random_list_of_nums)



def selection_sort(nums):
    for sorted_num in range(len(nums)): # sorted_num: how many values were sorted
        lowest_value_index = sorted_num # initialize first item of unsorted is the smallest
        for j in range(sorted_num + 1, len(nums)): # iterates over the unsorted items
            if nums[j] < nums[lowest_value_index]:
                lowest_value_index = j
        # Swap values of the lowest unsorted element with the first unsorted element
        nums[sorted_num], nums[lowest_value_index] = nums[lowest_value_index], nums[sorted_num]

# random_list_of_nums = [12, 8, 3, 20, 11]
# print(random_list_of_nums)
# selection_sort(random_list_of_nums)
# print(random_list_of_nums)



def insertion_sort(nums):
    # Start on the second element as we assume the first element is sorted
    for i in range(1, len(nums)): # i is index of first in unsorted
        item_to_insert = nums[i]
        prevIdx = i - 1 # keep a reference of the index of the previous element
        # Move all items of sorted back if they > item_to_insert (first in unsorted)   
        while prevIdx >= 0 and nums[prevIdx] > item_to_insert: # item_to_insert kept in separate ref
            nums[prevIdx + 1] = nums[prevIdx] # swap
            prevIdx -= 1
        # Insert the item
        nums[prevIdx + 1] = item_to_insert # necessary?

# random_list_of_nums = [9, 1, 15, 28, 6]
# print(random_list_of_nums)
# insertion_sort(random_list_of_nums)
# print(random_list_of_nums)



from heapq import heappop, heappush, heapify
# def heap_sort(nums): # using heappop 1. O(n*logn)
#     unsortedHeap = []
#     # 1. turn unsorted into heap
#     for element in nums: # unsorted = nums in the beginning
#         heappush(unsortedHeap, element)z
    
#     # 2. remove min from heap (automatically rebuild heap with 1 less value), add min to array
#     ordered = []
#     while unsortedHeap:
#         ordered.append(heappop(unsortedHeap))
#     return ordered

def heap_sort(nums): # using heapify - 1. O(n)
    # 1. turn unsorted (nums) into heap
    heapify(nums)
    
    # 2. remove min from heap (automatically rebuild heap with 1 less value), add min to array
    ordered = []
    while nums:
        ordered.append(heappop(nums))
    return ordered

# random_list_of_nums = [13, 21, 15, 5, 26, 4, 17, 18, 24, 2]
# print(random_list_of_nums)
# print(heap_sort(random_list_of_nums))



def merge(left_list, right_list): # O (len(left_list) + len(right_list))
    """Merge 2 sorted list:
    compare the smallest element of each list and add smaller of the two to final sorted_list"""
    sorted_list = []
    l = 0
    r = 0
    while len(sorted_list) < (len(left_list) + len(right_list)):
        if l < len(left_list) and r < len(right_list):
            # Add smaller of the two min (front of list) to sorted_list
            if left_list[l] <= right_list[r]:
                sorted_list.append(left_list[l])
                l += 1
            else:
                sorted_list.append(right_list[r])
                r += 1
        elif l == len(left_list): # Reached end of left list
            sorted_list.extend(right_list[r:])
        elif r == len(right_list): # Reached the end of right list
            sorted_list.extend(left_list[l:])
    return sorted_list

def merge_sort(nums):
    # If the list is a single element, return it
    if len(nums) <= 1:
        return nums
    # Sort and merge each half
    mid = len(nums) // 2
    left_list = merge_sort(nums[:mid]) # sorted left list
    right_list = merge_sort(nums[mid:]) # sorted right list
    # Merge the sorted lists into a new one
    return merge(left_list, right_list)

# random_list_of_nums = [120, 45, 68, 250, 176]
# print(random_list_of_nums)
# random_list_of_nums = merge_sort(random_list_of_nums) # return new list
# print(random_list_of_nums)



def partition(nums, low, high): # O(n)
    """
    Shift index of element >= pivot closest to the left (i) rightwards (+=1), and
    shift index of element <= pivot closest to the right (j) leftwards (-=1),
    making swaps along the way, until i>=j
    Parameters:
    low: starting left index
    high: starting right index
    """
    pivot = nums[(low + high) // 2]  # Other choices for pivot: first, last, median, random
    i = low - 1 # index of ele >= pivot closest to the left
    j = high + 1 # index of ele <= pivot closest to the right
    while True:
        i += 1
        while nums[i] < pivot: # start from left, find ele > pivot OR reach pivot
            i += 1
        j -= 1
        while nums[j] > pivot: # start from left, find ele < pivot OR reach pivot
            j -= 1
        if i >= j: # know pivot at right place!
            return j # index for ele at right spot or 1 next to it
        # If element at i (is/left of the pivot) > element at j (is/right of the pivot)
        nums[i], nums[j] = nums[j], nums[i] # swap


def quick_sort(nums):
    def _quick_sort(items, low, high): # helper function to be called recursively
        if low < high:
            # This is the index after the pivot, where our lists are split
            split_index = partition(items, low, high)
            _quick_sort(items, low, split_index) # no action if low=split_index (1 ele)
            _quick_sort(items, split_index + 1, high) # no action if split_index+1=high (1 ele)

    _quick_sort(nums, 0, len(nums) - 1)


# Verify it works
random_list_of_nums = [22, 5, 1, 18, 99]
quick_sort(random_list_of_nums)
print(random_list_of_nums)
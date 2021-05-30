# Given two arrays of integers, find all pairs from the arrays that sum to a certain target value
# A pair: a value from one array and a value from the other

## Question
# Duplicate values?

## Example
# [3, 1, 2]
# [10, 11]
# 12 = (1+11), (2+10)


## Optimized solution: given tarSum and ele1 of pair, know exactly what is looked for (tarSum-ele1)
def sumToTarget(arr1, arr2, tarSum):
    pairs = []
    arr1_eles = set()
    for num in arr1: ### O(arr1_len) ###
        arr1_eles.add(num)
    for num in arr2: ### O(arr2_len) ###
        tarHalf = tarSum - num
        if tarHalf in arr1_eles:
            pairs.append((tarHalf, num))
        # right now, if arr2 has duplicates, will have duplicate looking tuples in output
    return pairs

### Time: O(arr1_len) + O(arr2_len) ###
### Space: O(arr1_len) - len(arr1_eles) ≤ arr1_len ###

print(sumToTarget([], [], 1)) # => []
print(sumToTarget([1, 1, 1], [2, 2], 3)) # => [(1, 2), (1, 2)]
print(sumToTarget([3, 1, 2], [10, 11], 12)) # => [(2, 10), (1, 11)]
print(sumToTarget([3, 1, 2], [10, 11], 20)) # => []
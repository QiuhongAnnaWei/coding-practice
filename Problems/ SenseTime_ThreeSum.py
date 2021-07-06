
# three sum
# Input: [50 int without duplicate]
# can repeat
# Output: [ 3 numbers that sum to 0 ]

# [-1, 0, 1] => [[-1, 0, 1]]
# [0] => []
# [-1, 0, 1, 2]  should also output [[-1, 0, 1], [-1,-1,2]]

import numpy as np

def sum(nums):
    out = []
    nums.sort()
    for i in range(0, len(nums)):
        tar = -1 * nums[i]
        # 2 sum
        l, r = i, len(nums)-1
        while l<=r: # to allow for [0,0,0] for example
            sum = nums[l]+nums[r]
            if sum == tar:
                out.append([nums[i], nums[l], nums[r]])
                l+=1
                r-=1
            elif sum < tar:
                l+=1
            else:
                r-=1
    return out

# print(sum([-9, 0, 10, -5]))
# input = [np.random.randint(-100,100) for _ in range(20)]
# print(input)
# print(sum([input]))
# print(sum([-1,-49, 3, 10, -82, -2]))
print(sum([0]))
print(sum([-1,2]))
print(sum([-93, -62, -40, -1, -23, -98, -49, 97, -49, -78, -83, -86, 3, -2, -79, 48, 97, -60, 22, 45]))


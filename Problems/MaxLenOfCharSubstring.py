# Input: stirng s & integer k
# Return len(longest substring of s that contains at most k distinct characters)

# Example:
# input = eceba, k = 2
# output = 3 - 'ece'

# input = eceba, k = 0
# output = 0

# input = aaaaa, k > 0
# output = 5

# all lowercase

# ecebac, k = 2
def maxLenOfCharSubstring(s, k):
    maxLen = 0 # 3
    if len(s) == 0:
        return 0
    if k == 0:
        return 0
    substringChar = {}
    substrList = []
    for strCharIdx in range(len(s)):
        # 1. appends further character to substring
        if len(substringChar) <= k: 
            if s[strCharIdx] in substringChar:
                substringChar[s[strCharIdx]]+= 1
            else:
                substringChar[s[strCharIdx]] = 1
            substrList.append(s[strCharIdx])
        
        # 2. after appending, check if still within max char count requirement
        if len(substringChar) <= k:
            # potentially update max count
            if len(substrList) > maxLen:
                maxLen = len(substrList)
        else:
            # iteratively remove from front of substring until within requirement again
            while len(substringChar) > k:
                charToRemove = substrList[0]
                substringChar[charToRemove] -= 1
                if substringChar[charToRemove]==0:
                    del substringChar[charToRemove] # remove char from dict if substring no longer has it
                substrList = substrList[1:]
    return maxLen

print(maxLenOfCharSubstring('ecebaccccccc', 2))
### O(2n) -> O(n): each character can only be added once and removed once"

### optimization
# sliding window -> use start and end index instead
# hashmap substringChar keeps track of the rightmost index of each character - can skip over character

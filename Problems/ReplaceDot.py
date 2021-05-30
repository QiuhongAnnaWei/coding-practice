##  字节跳动（二面）
# s = '...'  n=3
# 用 * 去换 .
# 要求： 任意的两个* 不能相邻
# 求： 有多少种替换的可能 

def numWays(s):
    return replaceDot(len(s), canBeAsterik = True)

def replaceDot(sLen, canBeAsterik = True):
    # base case
    if sLen == 1:
        if canBeAsterik: 
            return 2
        else:
            return 1
    # recursion
    else:
        if canBeAsterik: # not replace + replace
             return replaceDot(sLen-1, True) + replaceDot(sLen-1, False)
        else: # not replace
            return replaceDot(sLen-1, True)

print(numWays("...."))


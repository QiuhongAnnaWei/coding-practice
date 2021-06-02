

# import sys 
# for line in sys.stdin:
#     s = line.strip()# abbbaaccb
#     if len(s) == 1:
#         print(str(0)+","+str(1))
#     # 1. find all chars
#     vocab = set()
#     for char in s:
#         vocab.add(char)

#     # 2. sliding window
#     l, r = 0, 0 # indices
#     winVocab = {} # {char:rightmost index}
#     startIdx = len(s)
#     length = len(s)+1
#     while l <= r and r <= len(s)-1:
#         if s[r] in winVocab:
#             winVocab[s[r]] = max(winVocab[s[r]], r)
#             while l<winVocab[s[l]]:
#                 l+=1
#         else:
#             winVocab[s[r]] = r
#         if len(winVocab) == len(vocab): # has all character
#             newLength = r-l+1
#             if newLength < length: # if ==, previous more to the left
#                 length = newLength
#                 startIdx = l
#         r += 1
#     print(str(startIdx)+","+str(length))


# #coding=utf-8
# # 本题为考试多行输入输出规范示例，无需提交，不计分。
# import sys

# def rec(forbidden, curP, cache, h, rel):
#     '''
#     forbidden: index of shirt of the superior of curP
#     curP: index of current person in consideration
#     cache: {(forbidden, curP): maxHappiness}
#     '''
#     # base case: no inferior
    
#     if curP not in rel:
#         return max(h[curP][:forbidden]+h[curP][forbidden+1:])
#     maxH = -1
#     for i in range(3):
#         if i != forbidden:
#             sum = h[curP][i]
#             for inferior in rel[curP]:
#                 if (i, inferior) in cache:
#                     sum += cache[(i, inferior)]
#                 else:
#                     happiness = rec(i, inferior, cache, h, rel)
#                     cache[(i, inferior)] = happiness
#                     sum += happiness
#             maxH = max(maxH, sum)
#     return maxH

# if __name__ == "__main__":
#     # 读取第一行的n
#     n = int(sys.stdin.readline().strip())
#     ans = 0
#     h = [] # 2d array where a row h[i] = [D[i], E[i], F[i]] for ith person
#     for i in range(n): # 2nd to n+1th rows -> happiness
#         line = sys.stdin.readline().strip()
#         temp = list(map(int, line.split())) # [1, 2, 3]
#         h.append(temp)
#     rel = {} # {superiorIdx:[inferior1Idx, inferior2Idx]}
#     for i in range(n-1): # relationship
#         line = sys.stdin.readline().strip()
#         temp = list(map(int, line.split(" ")))
#         if temp[0] in rel:
#             rel[temp[0]].append(temp[1])
#         else:
#             rel[temp[0]]=[temp[1]]
#     print(rec(-1, 0, {}, h, rel))
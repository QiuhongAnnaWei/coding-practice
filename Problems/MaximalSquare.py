### Leetcode 221. Maximal Square
### April 19 ByteDance Interview

# 最大正方形
# @param matrix char字符型二维数组 
# @return int整型
#

# 最大正方形
# 限定语言：Kotlin、Typescript、Python、C++、Groovy、Rust、Java、Go、C、Scala、Javascript、Ruby、Swift、Php、Python 3
# 给定一个由0和1组成的2维矩阵，返回该矩阵中最大的由1组成的正方形的面积
# 示例1
# 输入

# [[1,0,1,0,0]
#  [1,0,1,1,1],
#  [1,1,1,1,1],
#  [1,0,0,1,0]]

# 输出
# 4

import numpy as np
class Solution:
    def maxLen(self, matrix, rIdx, cIdx):
        widMax = 0
        # check if within bounds
        for c in matrix[rIdx][cIdx:]:
          if c == 1:
            widMax+=1
          else:
            break
           
        heightMax = 1
        rCounter = rIdx+1
        # check if within bounds
        while rCounter < len(matrix):
          if matrix[rCounter][cIdx] == 1:
            # print(rCounter, cIdx, heightMax)
            heightMax +=1
            rCounter +=1
          else:
            break
        return widMax, heightMax
      
    def wrongSolve(self, matrix):
        # write code here
        # 1. get sets of consecutive one strings in each row
        maxArea = 0
        for rowIdx in range(len(matrix)):
          for colIdx in range(len(matrix[0])):
            if matrix[rowIdx][colIdx] == 1:
              widMax, heightMax = self.maxLen(matrix, rowIdx, colIdx)
              squareDim = min(widMax, heightMax)
              if squareDim * squareDim > maxArea:
                maxArea = squareDim * squareDim
                print(maxArea, rowIdx, colIdx, widMax, heightMax)
        return maxArea

    def dynamicProgrammingSol(self, matrix):
        # stores maximum side length of the square the point is the bottom right corner of
        maxBRSqrLen = [ [0 for _ in range(len(matrix[0]))] for _ in range(len(matrix)) ]
        for rowIdx in range(len(matrix)):
            for colIdx in range(len(matrix[0])):
                currVal = matrix[rowIdx][colIdx]
                if rowIdx == 0 or colIdx == 0: # first row or first column
                    maxBRSqrLen[rowIdx][colIdx] = currVal # 1 or 0
                    continue # go to next element
                if currVal == 1: # not in first row or column
                    topLeft = maxBRSqrLen[rowIdx-1][colIdx-1]
                    top = maxBRSqrLen[rowIdx-1][colIdx]
                    left = maxBRSqrLen[rowIdx][colIdx-1]
                    ### NOTE: core relationship of the dp
                    maxBRSqrLen[rowIdx][colIdx] = min(topLeft, top, left) + 1 # NOTE: needs tl, t, l to all meet condition for a square (different for rectangle)
        maxPerRow = [max(row) for row in maxBRSqrLen]
        return max(maxPerRow) ** 2
                    

# [[1 0 1 0 0] # 0
#  [1 0 1 1 1] # 1
#  [1 1 1 1 1]
#  [1 0 0 1 0]]

# [[1 0 1 0 0] # 0
#  [1 0 ] # 1
#  [0 0 0]
#

sol = Solution()
a = [[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]]
b = [[1,0,1,1,1,1,1,0,0,0],[1,0,1,1,1,0,0,0,0,0],[1,0,1,0,1,1,1,0,0,0],[1,1,0,1,1,1,1,0,1,0],[1,1,1,1,1,1,1,1,0,0],[1,0,1,1,1,1,1,1,1,0],[1,0,1,1,1,1,1,1,1,0],[1,0,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1,0,0],[1,0,1,1,0,1,1,0,0,0]]
# print(np.array(a)) # 2 => 4
# print(np.array(b)) # 5 => 25
print(sol.dynamicProgrammingSol(b))
          

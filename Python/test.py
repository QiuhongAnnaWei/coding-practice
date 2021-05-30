# dictionary = {"key1":"value1", "key2":"value2", "key3":"value3"}
# for key, value in dictionary.items():
#     print(key, value)
# for key in dictionary:
#     print(key)
# for value in dictionary.values():
#     print(value)
# print(list(dictionary))
# print(list(dictionary.keys()))
# print("" in dictionary.values())

# dictionary = {10:"value1", 33:"value2", 71:"value3"}
# print(min(dictionary))
# del dictionary[33]
# print(dictionary)

# a=[1, 2, 3]
# print(a[-2])

# n = 3
# m = 2
# a = [ [ 0 for col in range(m)] for row in range(n) ]
# print(a)

# # a = {0: [3, 4], 1:[2]}
# # b = {1:[2], 0: [3, 4]}
# # print(a==b)
# print( int(''.join([str(1), str(0), str(3)])) )


# p = {} # {num: [[boolean, boolean]]} for non-last steps
# p[1] = [[True], [False]]
# for i in range(2, 4): # 1 to n-1
#     possibilities = []
#     pathsSoFar = p[i-1]
#     for pastWay in pathsSoFar:
#         if pastWay[-1] is True:
#             possibilities.append(pastWay+[True])
#             possibilities.append(pastWay+[False])
#         else: # 2 steps is the max
#             possibilities.append(pastWay+[True])
#     p[i] = possibilities
# print(p[1])
# print(p[2])
# print(p[3])


# def restoreArray(pairs):
#     if len(pairs)==1:
#         return pairs[0]

#     restoredArr = []    
#     # parse pairs into 
#     neighbors = {} # value: [neighbor1, neighbor2]
#     for pair in pairs:
#         if pair[0] in neighbors:
#             neighbors[pair[0]].append(pair[1])
#         else:
#             neighbors[pair[0]] = [pair[1]]
#         if pair[1] in neighbors:
#             neighbors[pair[1]].append(pair[0])
#         else:
#             neighbors[pair[1]] = [pair[0]]
    
#     startValue = None
#     for neighbor in neighbors:
#         if len(neighbors[neighbor]) == 1:
#             restoredArr = [neighbor, neighbors[neighbor][0]]
#         break
#     while len(restoredArr) < len(pairs) + 1:
#         currVal = restoredArr[-1]
#         currValneighbors = neighbors[currVal]
#         nextVal = currValneighbors[0] if currValneighbors[1]==restoredArr[-2] else currValneighbors[1]
#         restoredArr.append(nextVal)
    
#     return restoredArr

# pairs = [[3,5],[1,4],[2,4],[1,5]]
# print(restoreArray(pairs))


# s = ["1", "2", "3"]
# if "1" in s:
#     print('yes')
# else:
#     print(f'not in {s}')





# x = input('Enter your name:')
# print('Hello, ' + x) 

# for i in range(3):
    # input()



# import platform
# print(platform.python_implementation()) => CPython

# def myfunc1():
#   x = "myfunc1"
#   def myfunc2():
#     nonlocal x
#     x = "myfunc2"
#   myfunc2()
#   return x
# print(myfunc1()) => 'myfunc2', without 'nonlocal' 'myfunc1'


# def foo():
#     x = 20
#     def bar():
#         global x
#         x = 25

#     print("Before calling bar: ", x) # 20 (accessing foo's x)
#     bar()
#     print("After calling bar: ", x) # 20 (accessing foo's x)

# foo()
# print("x in main: ", x) # 25 (accessing bar's global x)

# table = [0, 1] # index is n, can also use dictionary {n: fib value}
# def fib(n):
#     if n < len(table): # using cache
#         return table[n]
#     for i in range(2, n+1):
#         table.append(table[i-1] + table[i-2])
#     return(table[n])

print(max(1, 2, 3))
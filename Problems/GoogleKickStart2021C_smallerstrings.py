
# Google Kick Start 2021 Round C


### Solution Attempt 1 - did not pass test set
# def buildPal(palSoFar, string, curIdx, strictlySmaller, alphabet, listSoFar):
#     if curIdx >= int((len(string)+1)/2): #handles both even and odd
#         pal = ''.join(palSoFar)
#         if (pal[curIdx:] < string[curIdx:] or curIdx==len(string)) and (pal!=string):
#             listSoFar.append(pal)
#         return listSoFar
#     for letter in alphabet:
#         if strictlySmaller is False or letter<= string[curIdx]:
#             palSoFar[curIdx]=letter
#             palSoFar[curIdx*-1 - 1]=letter
#             newSmaller = False if strictlySmaller and letter < string[curIdx] else strictlySmaller
#             listSoFar = buildPal(palSoFar, string, curIdx+1, newSmaller, alphabet, listSoFar)
#     return listSoFar

# numCase = int(input().strip())
# complete_alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
# for num in range(numCase):
#     n, k = map(int, input().strip().split())
#     string = input().strip()
#     pal = buildPal([None]*n, string, 0, True, complete_alphabet[:k], [])
#     print (f"Case #{num+1}: {int(len(pal)%(10**9+7)) }")




### Solution Attempt 2 based on analysis
# 1. find number of strings of size int((len(string)+1)/2) (halfpals) that are lexicographically smaller than that part of s
# this calculation = process of converting that part of s from base k (a=0, b=1...) to base 10
# Example: for K=4 and "da" -> 12 strings < da
    # 1) "**" - 3 choices for 1st letter * 4 choices for 2nd letter + "d*" - 0 chioce (since a is smallest)
    # 2) "da" = 30 in base 4 -> 3*4 + 0*1 = 12 
# 2. check if the string of size int((len(string)+1)/2) that = that part of that part of s is < complete s 

alph = "abcdefghijklmnopqrstuvwxyz"
mod_num = 10**9+7
numCase = int(input().strip())
for num in range(numCase):
    n, k = map(int, input().strip().split())
    string = input().strip()
    tot = 0
    # Step 1
    halflen = int((len(string)+1)/2)
    for char_idx in range(halflen): # 2 for 4; 3 for 5
        # tot += alph.index(string[char_idx]) * k**(halflen-char_idx-1)
        tot += alph.index(string[char_idx]) * pow(k, (halflen-char_idx-1), mod_num) # necessary to be within time limit
    # Step 2
    if n%2 == 0 and string[:halflen][::-1]<string[halflen:]: tot+=1
    if n%2 == 1 and string[:halflen][-2::-1]<string[halflen:]: tot+=1
    print (f"Case #{num+1}: {tot%(mod_num)}")
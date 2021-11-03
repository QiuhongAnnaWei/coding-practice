#a)  2 3 *         => 6
#b)  2 3 +         => 5
#c)  2 3 5 * +     => 17   # 2 + 3 * 5 
#d)  10 11 *       => 110
#e)  3 3 * 4 4 * + => 25   # （3 * 3) + (4 * 4)

#f)   1 1 1 ....  1 + ... +  => N
#             N       N-1


#g) x sqrt => sqrt(x)

#h) a0 ... an n avgn => |a0 ... an|

#i.1) 4 dup * => 16
#i.2) 10 dup * => 100

#j) [dup *] square define 10 square => 100

#k)  [1 +] inc define 5 inc => 6

#l) [square swap square sqrt] define pythagorean 3 4 => 5

#a1) Input is well-formed
#a2) Everything is an integer
#a3) A `tokenizer` is provided.

    

operators = {
    "*": lambda s: v =s.pop(); s[-1]*=v,
    "+": lambda s: v=s.pop(); s[-1]+=v,
    "dup": lambda s: s.append(s[-1]),
    "define": add_define,
}

def add_define(stack):
    global operators
    funName = stack.pop()
    funBody = stack.pop()
    operators[funName] = ???

def postfix(tokenizer):
# stack = [2, 3, 5] + intermediate = constant
# see *:
# pop 5 from stack=[2,3], save it in *'s right = 5 
# pop 3 from stack, 3*5 = 15
# (prev operator's) *'s left and (current operator)+'s right
    global operators
    stack = []
    val = next(tokenizer)
    while val:
        if val in operators: # operator
            operators[val](stack)
        else: # number
            val = int(val)
            stack.append(val)
        val = next(tokenizer)
    return stack[-1]

# xs 2 3 * => xs 6
# [xs 2 3] * => [xs 6]
        
    
# Given two strings, determine if they share a common substring (return Boolean)
# A substring may be as small as one character.

# Examples
# The words "a", "and", "art" share common substring
# The words "be" and "cat" do not share a substring

## Brainstorm
# if s1 and s2 share a substring > 1 character ('ab') => share 'a', and share 'b'
# this problem is really checking if s1 and s2 have overlap in letter


# Implementation using Hash Table

def tokenize(s): ### O(s) ###
    """returns a set of letters that appeared in s"""
    tokens = set()
    for char in s:
        tokens.add(char)
    return tokens

def ifShareSubStr(s1, s2):
    s1_tokens = tokenize(s1) ### O(s1) ###
    for char in s2: ### O(s2) ###
        if char in s1_tokens:
            return True
    return False

### Time: O(s1_len) + O(s2_len) - iterate through each character of both strings once ###
### Space: O(s1_len) - len(s1_tokens) <= number of characters in s1 ###

print(ifShareSubStr("rttttc", "and"))
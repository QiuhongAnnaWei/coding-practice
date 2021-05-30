### 1.8
# Assume you have a method isSubstring which checks if one word is a substring of another.
# Given two strings, s1 and s2, write code to check if s2 is a rotation of s1 using only
# one call to isSubstring (i.e., “waterbottle” is a rotation of “erbottlewat”). 

## Questions
# Uppercase, lowercase - rotation, isSubstring()
# spaces and non-letter characters - rotation
# left to right only?

## Clarification & Brainstorm
# 'abc' rotation: 'abc', 'cab', 'bca' - substrings of abcabc

import unittest

# random implementation
def isSubstring(smallstr, bigstr):
    """return if smallstr is a substring of bigstr"""
    if smallstr in bigstr:
        return True
    else:
        return False

def isRotation(s1, s2):
    """return if s2 is a rotation of s1"""
    if len(s1) != len(s2): #O(1)
        return False
    doubles1 = s1+s1 # O(2*s1_len) -> O(n)
    return isSubstring(s2, doubles1) # O(bigstr_len) probably -> O(2n) -> O(n)

### Time complexity ###
# n: len of s1+s2
# dependent on isSubstring(), if it is O(n)
# -> O(1) + O(n) + O(n) -> 2*O(n)

class Test(unittest.TestCase):
    def testRotStr(self):
        """True example given in problem statement"""
        self.assertTrue(isRotation("erbottlewat", "waterbottle"))

    def testEmptyStr(self):
        self.assertTrue(isRotation("", ""))
        self.assertTrue(isRotation(" ", " "))
        self.assertTrue(isRotation("  1", "1  "))
        self.assertTrue(isRotation("  1", " 1 "))
        self.assertFalse(isRotation("  1", " 2 "))
        self.assertFalse(isRotation("  1", " 1  "))

    def testRepetition(self):
        self.assertTrue(isRotation("aa", "aa"))
        self.assertTrue(isRotation("  ", "  "))
        self.assertTrue(isRotation("abab", "baba"))
        self.assertFalse(isRotation("abab", "abba"))

    def testContainmentNotRot(self):
        self.assertFalse(isRotation("waterbottle", "wat"))
        self.assertFalse(isRotation("waterbottle", "ew"))
        self.assertFalse(isRotation("waterbottle", "ewa"))
    
    def testSameStr(self):
        self.assertTrue(isRotation("waterbottle", "waterbottle"))
    
    def testSimilarStr(self):
        self.assertFalse(isRotation("waterbottle", "waterbottla"))
        self.assertFalse(isRotation("waterbottle", "awaterbottl"))
        self.assertFalse(isRotation("waterbottle", "wwaterbottl"))
        self.assertFalse(isRotation("abc", "wwaterbottl"))

if __name__ == "__main__":
    unittest.main()
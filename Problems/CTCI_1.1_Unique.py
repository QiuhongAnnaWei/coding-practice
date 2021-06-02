### 1.1
# Implement an algorithm to determine if a string has all unique characters.
# What if you can not use additional data structures?


## Questions
# Uppercase/lowercase
# Empty string?
# How many characters? 26? 128? 256? ASCII?

import unittest

def unique(str):
    occured_char = set()
    for char in str: # ≤ n
        if char in occured_char: # O(1)
            return False
        else:
            occured_char.add(char) # O(1)
    return True

### Time complexity ###
# n: len of str
# ≤n * O(1) -> O(n)

class Test(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):

    # @classmethod
    # def tearDownClass(cls):

    def testEmptyStr(self):
        """unique should return True with empty strings"""
        self.assertTrue(unique(""))

    def testSpaceStr(self):
        """unique should return True with string of one space"""
        self.assertTrue(unique(" "))

    def testMultipleSpaceStr(self):
        """unique should return True with string of more than one space"""
        self.assertFalse(unique("   "))

    def testSingleCharStr(self):
        """unique should return True with single character strings"""
        self.assertTrue(unique('a'))

    def testDoubleCharStr(self):
        """unique should return True with double character strings that don't have the same characters"""
        self.assertTrue(unique('ab'))

    def testStrWithTwoConsChar(self):
        """unique should return False with strings that have two same consecutive characters"""
        self.assertFalse(unique('abb'))

    def testStrWithTwoNonConsChar(self):
        """unique should return False with strings that have two same non consecutive characters"""
        self.assertFalse(unique('abac1!da'))

    def testStrWithNoConsChar(self):
        """unique should return True with strings that have all unique characters"""
        self.assertTrue(unique('abcd123e!fg'))
        
if __name__ == "__main__":
    unittest.main()



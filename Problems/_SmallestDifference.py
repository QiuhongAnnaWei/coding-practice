# Given two arrays of integers, compute the pair of values (one value in each array)
# with the smallest non-negative difference. Return the difference

## Questions
# sorted array?
# duplicate values?

## Clarification & Brainstorm
# [10, 20, 30], [-1, 0, 1]

import unittest

def smallest_difference(a1, a2):
    a1.sort()
    a2.sort()
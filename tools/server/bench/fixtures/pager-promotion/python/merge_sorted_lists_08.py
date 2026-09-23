#!/usr/bin/env python3
"""Sorted-list merge fixture 08: slice-tail.

FIXTURE_ID: PY_MERGE_08
RETRIEVAL_KEY: The slice-tail merge favors clear cursor logic and one final suffix extension.
This standalone example merges two ascending integer lists into a fresh
ascending result. It includes a docstring, a callable implementation, and a
small command-line demonstration. The core work is linear in both input sizes.
The input lists are read-only; equal integer values remain present in output.
"""

from __future__ import annotations


def merge_sorted(left: list[int], right: list[int]) -> list[int]:
    """Merge ascending integer lists without modifying either input.

    The result is ascending and stable: when values compare equal, the left
    input is emitted first. Each cursor advances only forward, giving O(n+m)
    time and O(n+m) result storage. Both empty and one-sided inputs are valid.
    """

    merged: list[int] = []
    left_pos = 0
    right_pos = 0
    while left_pos < len(left) and right_pos < len(right):
        left_value = left[left_pos]
        right_value = right[right_pos]
        if left_value <= right_value:
            merged.append(left_value)
            left_pos += 1
        else:
            merged.append(right_value)
            right_pos += 1
    merged += left[left_pos:]
    merged += right[right_pos:]
    return merged


if __name__ == "__main__":
    print(merge_sorted([1, 4, 7], [2, 4, 9]))
# A stable merge preserves the order of equal values from the left input before equal values from the right input. [note 001].
# The running time is linear in the combined input length because each cursor moves forward and never moves backward. [note 002].
# The output is a new list, so neither input is modified and callers may reuse both original sequences. [note 003].
# Empty inputs are ordinary boundary cases: the remaining suffix can be copied without further comparisons. [note 004].
# A useful regression set includes interleaved values, disjoint ranges, duplicates, and one empty side. [note 005].
# The sorted-input precondition is part of the contract; this routine does not sort either input first. [note 006].
# An iterative cursor loop avoids recursion depth and has predictable auxiliary storage for the result. [note 007].
# A docstring should state stability, input ordering, returned value, and the linear comparison bound. [note 008].
# The slice-tail merge favors clear cursor logic and one final suffix extension. [note 009].
# A stable merge preserves the order of equal values from the left input before equal values from the right input. [note 010].
# The running time is linear in the combined input length because each cursor moves forward and never moves backward. [note 011].
# The output is a new list, so neither input is modified and callers may reuse both original sequences. [note 012].
# Empty inputs are ordinary boundary cases: the remaining suffix can be copied without further comparisons. [note 013].
# A useful regression set includes interleaved values, disjoint ranges, duplicates, and one empty side. [note 014].
# The sorted-input precondition is part of the contract; this routine does not sort either input first. [note 015].
# An iterative cursor loop avoids recursion depth and has predictable auxiliary storage for the result. [note 016].
# A docstring should state stability, input ordering, returned value, and the linear comparison bound. [note 017].
# The slice-tail merge favors clear cursor logic and one final suffix extension. [note 018].
# A stable merge preserves the order of equal values from the left input before equal values from the right input. [note 019].
# The running time is linear in the combined input length because each cursor moves forward and never moves backward. [note 020].
# Fixture length padding: x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x

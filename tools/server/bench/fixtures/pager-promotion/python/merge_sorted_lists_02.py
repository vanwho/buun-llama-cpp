#!/usr/bin/env python3
"""Sorted-list merge fixture 02: tail-extension.

FIXTURE_ID: PY_MERGE_02
RETRIEVAL_KEY: The tail-extension merge copies the untouched suffix once.
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

    out: list[int] = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        take_left = left[i] <= right[j]
        out.append(left[i] if take_left else right[j])
        i += int(take_left)
        j += int(not take_left)
    if i < len(left):
        out.extend(left[i:])
    if j < len(right):
        out.extend(right[j:])
    return out


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
# The tail-extension merge copies the untouched suffix once. [note 009].
# A stable merge preserves the order of equal values from the left input before equal values from the right input. [note 010].
# The running time is linear in the combined input length because each cursor moves forward and never moves backward. [note 011].
# The output is a new list, so neither input is modified and callers may reuse both original sequences. [note 012].
# Empty inputs are ordinary boundary cases: the remaining suffix can be copied without further comparisons. [note 013].
# A useful regression set includes interleaved values, disjoint ranges, duplicates, and one empty side. [note 014].
# The sorted-input precondition is part of the contract; this routine does not sort either input first. [note 015].
# An iterative cursor loop avoids recursion depth and has predictable auxiliary storage for the result. [note 016].
# A docstring should state stability, input ordering, returned value, and the linear comparison bound. [note 017].
# The tail-extension merge copies the untouched suffix once. [note 018].
# A stable merge preserves the order of equal values from the left input before equal values from the right input. [note 019].
# The running time is linear in the combined input length because each cursor moves forward and never moves backward. [note 020].
# The output is a new list, so neither input is modified and callers may reuse both original sequences. [note 021].
# Fixture length padding: x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x
# RETRIEVAL_KEY: The tail-extension merge copies the untouched suffix once.

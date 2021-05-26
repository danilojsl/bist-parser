# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/

import tensorflow as tf
import numpy as np


def parse_proj(scores, gold=None):
    '''
    Parse using Eisner's algorithm.
    '''
    number_rows, number_columns = int(scores.shape.dims[0]), int(scores.shape.dims[1])
    if number_rows != number_columns:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = number_rows  # Number of words (excluding root).

    # Initialize CKY table.
    complete = np.zeros([N, N, 2])  # s, t, direction (right=1).
    complete_tf = tf.zeros([N, N, 2])
    # complete_tf = tf.reshape(scores, shape=(-1, -1, 2))
    incomplete = np.zeros([N, N, 2])  # s, t, direction (right=1).
    complete_backtrack = -np.ones([N, N, 2], dtype=int)  # s, t, direction (right=1).
    incomplete_backtrack = -np.ones([N, N, 2], dtype=int)  # s, t, direction (right=1).

    # Loop from smaller items to larger items.
    for k in range(1, N):
        for s in range(N - k):
            t = s + k

            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, 1] + complete[(s + 1):(t + 1), t, 0] + scores[t, s] + (
                0.0 if gold is not None and gold[s] == t else 1.0)
            incomplete[s, t, 0] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, 1] + complete[(s + 1):(t + 1), t, 0] + scores[s, t] + (
                0.0 if gold is not None and gold[t] == s else 1.0)
            incomplete[s, t, 1] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s, (s + 1):(t + 1), 1] + complete[(s + 1):(t + 1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

    heads = [-1 for _ in range(N)]
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N - 1, 1, 1, heads)

    return heads


def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    '''
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
    head of each word.
    '''
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s][t][direction]
        if direction == 0:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return

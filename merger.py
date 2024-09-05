#####################################################
### DOCUMENT PROCESSOR [MERGER]
#####################################################
# Jonathan Wang

# ABOUT:
# This project creates an app to chat with PDFs.

# This is the MERGER
# which defines how two lists with scores
# should be merged together into one list.
# (Useful for fusing things like keywords or textnodes)
#####################################################
## TODOS:
# We're looping through A/B more than necessary.

#####################################################
## IMPORTS:
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#####################################################
## CODE:

GenericType = TypeVar("GenericType")

### TODO(Jonathan Wang): Implement Maximum Marginal Relevance (MMR)
# https://en.wikipedia.org/wiki/Maximum_marginal_relevance
# def mmr(documents, query, scores, lambda_param=0.5):
#     """
#     Calculate Maximum Marginal Relevance (MMR) for a list of documents.

#     Parameters:
#     documents (list of np.array): List of document vectors.
#     query (np.array): Query vector.
#     scores (list of float): Relevance scores for each document.
#     lambda_param (float): Trade-off parameter between relevance and diversity.

#     Returns:
#     list of int: Indices of selected documents in order of selection.
#     """
#     selected = []
#     remaining = list(range(len(documents)))

#     while remaining:
#         if not selected:
#             # Select the document with the highest relevance score
#             idx = np.argmax(scores)
#         else:
#             # Calculate MMR for remaining documents
#             mmr_scores = []
#             for i in remaining:
#                 relevance = scores[i]
#                 diversity = max([np.dot(documents[i], documents[j]) for j in selected])
#                 mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
#                 mmr_scores.append(mmr_score)
#             idx = remaining[np.argmax(mmr_scores)]

#         selected.append(idx)
#         remaining.remove(idx)

#     return selected

def _merge_on_scores(
    a_list: Sequence[GenericType],
    b_list: Sequence[GenericType],
    a_scores_input: Sequence[float | np.float64 | None],
    b_scores_input: Sequence[float | np.float64 | None],
    use_distribution: bool = True,
    a_weight: float = 0.5,
    top_k: int = 5,
) -> Sequence[GenericType]:
    """
    Given two lists of elements with scores, fuse them together using "Distribution-Based Score Fusion".

    Elements which have high scores in both lists are given even higher ranking here.

    Inputs:
        a_list: list of elements for A
        a_scores: list of scores for each element in A. Assume higher is better. Share the same index.
        b_list: list of elements for B
        b_scores: list of scores for each element in B. Assume higher is better. Share the same index.
        use_distribution: Whether to fuse using Min-Max Scaling (FALSE) or Distribution Based Score Fusion (TRUE)

    Outputs:
        List: List of elements that passed the merge.
    """
    # Guard Clauses
    if ((len(a_list) != len(a_scores_input)) or (len(b_list) != len(b_scores_input))):
        msg = (
            f"""_merge_on_scores: Differing number of elements and scores!
a_list: {a_list}
a_scores: {a_scores_input}
b_list: {b_list}
b_scores: {b_scores_input}
"""
        )
        raise ValueError(msg)

    if (a_weight > 1 or a_weight < 0):
        msg = "_merge_on_scores: weight for the A list should be between 0 and 1."
        raise ValueError(msg)
    if (top_k < 0): # or top_k > :
        # TODO(Jonathan Wang): Find a nice way to get the number of unique elements in a list
        # where those elements are potentially unhashable AND unorderable.
        # I know about the n^2 solution with two lists and (if not in x), but it's a bit annoying.
        msg = "_merge_on_scores: top_k must be between 0 and the total number of elements."
        raise ValueError(msg)

    # 0. Convert to numpy arrays
    # NOTE: When using a SubQuestionQueryEngine, the subanswers are saved as NodesWithScores, but their score is None.
    # We want to filter these out, so we get citations when the two texts are very similar.
    a_scores: NDArray[np.float64] = np.array(a_scores_input, dtype=np.float64)
    b_scores: NDArray[np.float64] = np.array(b_scores_input, dtype=np.float64)

    # 1. Calculate mean of scores.
    a_mean = np.nanmean(a_scores)  # np.nan if empty
    b_mean = np.nanmean(b_scores)

    # 2. Calculate standard deviations
    a_stdev = np.nanstd(a_scores)
    b_stdev = np.nanstd(b_scores)

    # 3. Get minimum and maximum bands as 3std from mean
    # alternatively, use actual min-max scaling
    a_min = a_mean - 3 * a_stdev if use_distribution else np.nanmin(a_scores)
    a_max = a_mean + 3 * a_stdev if use_distribution else np.nanmax(a_scores)
    b_min = b_mean - 3 * b_stdev if use_distribution else np.nanmin(b_scores)
    b_max = b_mean + 3 * b_stdev if use_distribution else np.nanmax(b_scores)

    # 4. Rescale the distributions
    if (a_max > a_min):
        a_scores = np.array([
            ((x - a_min) / (a_max - a_min))
            for x in a_scores
        ], dtype=np.float64)
    if (b_max > b_min):
        b_scores = np.array([
            (x - b_min) / (b_max - b_min)
            for x in b_scores
        ], dtype=np.float64)

    # 5. Fuse the scores together
    full_dict: list[tuple[GenericType, float]] = []
    for index, element in enumerate(a_list):
        a_score = a_scores[index]
        if (element in b_list):
            # In both A and B. Fuse score.
            b_score = b_scores[b_list.index(element)]
            fused_score = a_weight * a_score + (1-a_weight) * b_score
            full_dict.append((element, fused_score))
        else:
            # Only in A.
            full_dict.append((element, a_weight * a_score))

    for index, element in enumerate(b_list):
        if (element not in a_list):
            b_score = b_scores[index]
            full_dict.append((element, (1-a_weight) * b_score))

    full_dict = sorted(full_dict, key=lambda item: item[1], reverse=True)
    output_list = [item[0] for item in full_dict]

    if (top_k >= len(full_dict)):
        return output_list

    # create final response object
    return output_list[:top_k]

#####################################################
# LIFT DOCUMENT INTAKE PROCESSOR [MERGER]
#####################################################
# Jonathan Wang (jwang15, jonathan.wang@discover.com)
# Greenhouse GenAI Modeling Team

# ABOUT: 
# This project automates the processing of legal documents 
# requesting information about our customers.
# These are handled by LIFT
# (Legal, Investigation, and Fraud Team)

# This is the MERGER
# which defines how two lists with scores
# should be merged together into one list.
# (Useful for fusing things like keywords or textnodes)
#####################################################
## TODOS:
# We're looping through A/B more than necessary.

#####################################################
## IMPORTS:
from typing import List
import numpy as np

#####################################################
## CODE:
def _merge_on_scores(
    a_list: List,
    b_list: List,
    a_scores: List[float],
    b_scores: List[float],
    use_distribution: bool = True,
    a_weight: float = 0.5,
    top_k: int = 5,
) -> List:
    """
    Given two lists of elements with scores, fuse them together using "Distribution-Based Score Fusion"
    I.e., elements which have high scores in both lists are given even higher ranking here.

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
    if ((len(a_list) != len(a_scores)) or (len(b_list) != len(b_scores))):
        raise Exception("_merge_on_scores: Differing number of elements and scores!")
    if (a_weight > 1 or a_weight < 0):
        raise Exception("_merge_on_scores: weight for the A list should be between 0 and 1.")
    if (top_k < 0): # or top_k > :
        # TODO: Find a nice way to get the number of unique elements in a list
        # where those elements are potentially unhashable AND unorderable.
        # I know about the n^2 solution with two lists and (if not in x), but it's a bit annoying.
        raise Exception("_merge_on_scores: top_k must be between 0 and the total number of elements.")

    # 1. Calculate mean of scores.
    a_mean = np.mean(a_scores)  # np.nan if empty
    b_mean = np.mean(b_scores)

    # 2. Calculate standard deviations
    a_stdev = np.std(a_scores)
    b_stdev = np.std(b_scores)

    # 3. Get minimum and maximum bands as 3std from mean
    # alternatively, use actual min-max scaling
    a_min = a_mean - 3 * a_stdev if use_distribution else np.min(a_scores)
    a_max = a_mean + 3 * a_stdev if use_distribution else np.max(a_scores)
    b_min = b_mean - 3 * b_stdev if use_distribution else np.min(b_scores)
    b_max = b_mean + 3 * b_stdev if use_distribution else np.max(b_scores)

    # 4. Rescale the distributions
    if (a_max > a_min):
        a_scores = [
            float((x - a_min) / (a_max - a_min))
            for x in a_scores
        ]
    if (b_max > b_min):
        b_scores = [
            float((x - b_min) / (b_max - b_min))
            for x in b_scores
        ]

    # 5. Fuse the scores together
    full_dict = []
    for index, element in enumerate(a_list):
        a_score = a_scores[index]
        if (element in b_list):
            # In both A and B. Fuse score.
            b_score = b_scores[b_list.index(element)]
            fused_score = a_weight * a_score + (1-a_weight) * b_score
            full_dict.append((element, fused_score))
        else:
            a_score = a_scores[index]
            full_dict.append((element, a_weight * a_score))

    for index, element in enumerate(b_list):
        if (element not in a_list):
            b_score = b_scores[index]
            full_dict.append((element, (1-a_weight) * b_score))

    full_dict = list(reversed(sorted(full_dict, key=lambda item: item[1])))
    output_list = [item[0] for item in full_dict]

    if (top_k >= len(full_dict)):
        return output_list

    # create final response object
    return output_list[:top_k]
# Inspired by https://github.com/ace19-dev/image-retrieval-pytorch/blob/master/retrieval/retrieve.py

import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity,manhattan_distances

#maybe add AP?

# def cosine_distance(a, b, data_is_normalized=False):
#     return cosine_similarity(a, b)


# def euclidean_distance(a, b):
#     return np.linalg.norm(a - b)

# def manhattan_distance(a,b):
#     return manhattan_distances(a,b)


# class NearestNeighborDistanceMetric(object):
#     def __init__(self, metric, matching_threshold=None, budget=None):
#         if metric == "euclidean":
#             self._metric = _nn_euclidean_distance
#         elif metric == "cosine":
#             self._metric = _nn_cosine_distance
#         else:
#             raise ValueError(
#                 "Invalid metric; must be either 'euclidean' or 'cosine'")
#         self.matching_threshold = matching_threshold
#         self.budget = budget    # Gating threshold for cosine distance
#         self.samples = {}

#     def distance(self, queries, galleries):
#         """Compute distance between galleries and queries.

#         Parameters
#         ----------
#         queries : ndarray
#             An LxM matrix of L features of dimensionality M to match the given `galleries` against.
#         galleries : ndarray
#             An NxM matrix of N features of dimensionality M.

#         Returns
#         -------
#         ndarray
#             Returns a cost matrix of shape LxN

#         """
#         return self._metric(queries, galleries)
    
#testing metrics
from statistics import fmean
from typing import Sequence


def reciprocal_rank(found: Sequence[str], ground_truth: str) -> float:
    try:
        rank = found.index(ground_truth) + 1
        rank = 1 / rank
    except ValueError:
        rank = 0.
    return rank


def average_precision(found: Sequence[str], ground_truth: str) -> float:
    groups = list(map(lambda index: (found[index], found[: index + 1]), range(len(found))))
    groups = list(filter(lambda group: group[0] == ground_truth, groups))
    precisions = list(map(lambda group: precision(found=group[1], ground_truth=group[0]), groups))
    return fmean(precisions) if precisions else 0.


def mean_average_precision(retrievals: Sequence[Sequence[str]], labels: Sequence[str]) -> float:
    average_precisions = list(
        map(lambda groups: average_precision(found=groups[0], ground_truth=groups[1]), list(zip(retrievals, labels)))
    )
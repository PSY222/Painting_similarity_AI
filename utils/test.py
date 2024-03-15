from itertools import product
from os.path import join

import polars as pl


from src.addons.data import load_database
from metrics import cosine_distance,euclidean_distance #metrics

finders = {"cosinus": cosine_distance, "euclidean": euclidean_distance, "manhattan": ManhattanFinder}


def test(input_path: str, feature_path: str, output_path: str):


if __name__ == "__main__":
    
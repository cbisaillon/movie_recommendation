import pandas as pd
import os

script_path = os.path.dirname(__file__)
file_path = os.path.join(script_path, "../big_files/ml-25m/ratings.csv")


def loadRatings(max_recommendations):
    """
    Loads the ratings in the dataset
    :param movieList: the list of movies to correctly attach a movie to its rating
    :return: the list of rating in the dataset
    """

    data = pd.read_csv(file_path, nrows=max_recommendations)

    return data

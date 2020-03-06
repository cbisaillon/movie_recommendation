import pandas as pd

file_path = "../big_files/ml-25m/ratings.csv"


def loadRatings():
    """
    Loads the ratings in the dataset
    :param movieList: the list of movies to correctly attach a movie to its rating
    :return: the list of rating in the dataset
    """

    data = pd.read_csv(file_path)
    return data

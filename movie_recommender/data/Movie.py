import pandas as pd
import os

file_path = "../big_files/ml-25m/movies.csv"


class Movie:
    """
    Class that contains the information about a movie
    """
    def __init__(self, id, title, genres):
        self.id = id
        self.title = title
        self.genres = genres

    def __str__(self):
        return "Movie #{}, titled: {}".format(self.id, self.title)


def loadMovies():
    """
    Loads the movies in the dataset
    :return: the list of movies in the dataset
    """

    data = pd.read_csv(file_path)
    movies = {}

    for i, row in data.iterrows():
        movie = Movie(row['movieId'], row['title'], row['genres'].split('|'))
        movies[row['movieId']] = movie

    return movies

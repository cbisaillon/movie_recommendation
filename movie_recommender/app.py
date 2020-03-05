import sys
from movie_recommender.data.Movie import loadMovies
from movie_recommender.data.Rating import loadRatings
from movie_recommender.data.User import User

import torch.nn as nn


def main():
    print("Loading movies")
    movies = loadMovies()

    user = User(nb_movies=len(movies))

    print("Loading ratings")
    ratings = loadRatings()




    print(len(movies))
    print(len(ratings))


class NetworkRecommender(nn.Module):
    """
    The network that learns how to make good movie recommendation
    """

    def __init__(self, nb_movies, hidden_size):
        self.nb_hidden = hidden_size

        self.i2h = nn.Linear(nb_movies, hidden_size)
        self.h2o = nn.Linear(hidden_size, nb_movies)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.i2h(input)
        output = self.i2o(output)
        return self.softmax(output)


if __name__ == "__main__":
    argv = sys.argv

    # todo: Parse arguments
    main()

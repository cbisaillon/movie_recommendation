import numpy

class User:
    def __init__(self, nb_movies):

        # Initialize a list containing the rating for each movies with -1 values
        self.ratings = numpy.full(nb_movies, -1)
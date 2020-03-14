import pandas as pd
import numpy as np
import os
from scipy import spatial

dirname = os.path.dirname(__file__)

max_users = 500
matrix_save_loc = os.path.join(dirname, "matrices/user_based")
not_rated_number = -1


def buildUserMovieMatrix(movies: pd.DataFrame, ratings: pd.DataFrame, movieId2Idx):
    print("Building user-movie matrix...")
    users = ratings.userId.unique()[:max_users]
    matrix = np.full((len(users), len(movies)), not_rated_number)

    for userRowIdx, userId in enumerate(users):
        ratings_for_user = ratings[ratings['userId'] == userId]
        for idx, rating in ratings_for_user.iterrows():
            movie_idx = int(movieId2Idx[rating['movieId']])
            rating_val = int(rating['rating'])
            matrix[userRowIdx][movie_idx] = rating_val

    print("Saving user movie matrix")
    np.save(matrix_save_loc, matrix)
    return matrix


class UserBasedRecommender:
    def __init__(self, movies, ratings, moviesId2Idx, load_from_file=True):
        if load_from_file:
            if os.path.isfile(matrix_save_loc + ".npy"):
                print("Loading user movie matrix from file")
                self.userMovieMat = np.load(matrix_save_loc + ".npy")
            else:
                self.userMovieMat = buildUserMovieMatrix(movies, ratings, moviesId2Idx)
        else:
            self.userMovieMat = buildUserMovieMatrix(movies, ratings, moviesId2Idx)

        self.user_movie_tree = spatial.KDTree(self.userMovieMat)

    def recommend(self, user_ratings, nb_recommendations=5):
        print("Making recommendations...")

        # Find the closest user based on the rating of the user
        similarities, closest_users = self.user_movie_tree.query(user_ratings, nb_recommendations)

        # Find items that those users have rated but I have not
        mean_rating = np.mean(user_ratings[user_ratings != not_rated_number])

        suggestion_score = np.full((user_ratings.shape[0]), not_rated_number)

        for movieIdx in range(user_ratings.shape[0]):
            summation_numerator = 0.0
            summation_denominator = 0.0

            for similarity, userId in zip(similarities, closest_users):
                summation_denominator += similarity
                summation_numerator += (similarity * (self.userMovieMat[userId][movieIdx] - np.mean(self.userMovieMat[userId])))

            suggestion_score[movieIdx] = mean_rating + summation_numerator / summation_denominator

        # Get the movies with highest recommendation score
        best_suggestions = suggestion_score.argsort()[-nb_recommendations:][::-1]

        return best_suggestions

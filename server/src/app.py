import sys
from data.Movie import loadMovies
from data.Rating import loadRatings
import torch
import numpy as np
from methods.network.Recommender import NetworkRecommender
from methods.collaborative.user_based import UserBasedRecommender
from flask import Flask, request
from flask_cors import CORS
import os

import torch.nn as nn
import torch.optim as optim
import json
import cProfile


identifier = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(identifier)
userMovie2Rating = {}
learning_rate = 0.001
weight_decay = 0.00001
n_epochs = 100
max_recommendations = 100000

script_dir = os.path.dirname(__file__)
movie_poster_dir = os.path.join(script_dir, "data/posters")

main = None

pr = cProfile.Profile()
pr.enable()

webapp = Flask(__name__)
CORS(webapp)

@webapp.route('/')
def index():
    # Get the ratings in the url
    user_ratings = request.args.get('ratings')
    if user_ratings:
        return processQuery(user_ratings)
    else:
        return "You have to specify your movie ratings in the following format: {movieId1}:{rating1},{movieId2}:{rating2},..."

@webapp.route('/list')
def listMovies():

    page = request.args.get('page')
    if page is None:
        page = 1
    else:
        page = int(page)

    per_page = 20

    movies = main.movies[main.movies['average_score'].notna()]
    movies = movies.sort_values(by=['average_score'], ascending=False)
    nbMovies = len(movies)

    start = (page - 1) * per_page
    end = page * per_page

    if start < 0:
        start = 0

    if end > nbMovies:
        end = nbMovies - 1

    return json.dumps(dict([(id, title) for id, title in zip(movies[start:end].movieId, movies[start:end].title)]))


def processQuery(query):
    user_ratings = query.split(',')
    movie_score = {}
    for rating in user_ratings:
        movie_id = int(rating.split(":")[0])
        score = int(rating.split(":")[1])

        if not movie_id in movie_score.keys():
            movie_score[movie_id] = score

    # Send the data for recommendation
    return json.dumps(main.getRecommendation(movie_score))


class Main:

    def __init__(self):
        print("Loading movies")
        self.movies, self.movieId2Idx, self.movieIdx2Id = loadMovies()

        print("Loading ratings")
        self.ratings = loadRatings(max_recommendations)

        print("Computing average score for each movies")

        self.movies['average_score'] = self.movies.apply(lambda row: (self.ratings[self.ratings['movieId'] == row['movieId']])['rating'].sum(), axis=1)

        self.recommender = UserBasedRecommender(self.movies, self.ratings, self.movieId2Idx)

    def getRecommendation(self, ratings):
        user_rating_mat = np.full((len(self.movies)), -1, dtype=np.byte)

        # Build the user's rating matrix
        for movieId, score in ratings.items():
            # print(movieId)
            movieIdx = self.movieId2Idx[movieId]
            user_rating_mat[movieIdx] = score

        recommendations = self.recommender.recommend(user_rating_mat)
        recommendation_names = self.movies.loc[self.movies['movieId'].isin(recommendations)]['title']

        return recommendation_names.tolist()

    def runServer(self):
        webapp.run(threaded=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
        return webapp


def oldMethod(users, movies, ratings, movie_id_to_index):
    net = NetworkRecommender(len(users), len(movies)).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # Train the network
    i = 0
    total_loss = 0

    for user in users:
        net.zero_grad()

        rated_movies = ratings[(ratings["userId"] == user)]
        average_loss_user = 0
        for idx, rating in rated_movies.iterrows():
            score = torch.Tensor([rating['rating'] / 5.0]).to(device)
            user_id = torch.LongTensor([rating['userId']]).to(device)
            movie_id = torch.LongTensor([movie_id_to_index[int(rating['movieId'])]]).to(device)

            prediction = net(user_id, movie_id).view(-1)
            loss = criterion(prediction, score)
            average_loss_user += loss

        total_loss += average_loss_user / len(rated_movies)

        loss.backward()
        optimizer.step()

        if (i % 100 == 0):
            print("{}/{}. Loss: {}".format(i, len(users), total_loss / 100))
            total_loss = 0

        i += 1


if __name__ == "__main__":
    argv = sys.argv

    main = Main()
    main.runServer()

pr.disable()
pr.print_stats(sort="calls")
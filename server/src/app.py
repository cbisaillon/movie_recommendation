import sys
from data.Movie import loadMovies
from data.Rating import loadRatings
import torch
import numpy as np
from methods.network.Recommender import NetworkRecommender
from methods.collaborative.user_based import UserBasedRecommender
from flask import Flask
import os

import torch.nn as nn
import torch.optim as optim
from web.routes import index_page

identifier = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(identifier)
userMovie2Rating = {}
learning_rate = 0.001
weight_decay = 0.00001
n_epochs = 100

script_dir = os.path.dirname(__file__)
movie_poster_dir = os.path.join(script_dir, "data/posters")


webapp = Flask(__name__)
webapp.register_blueprint(index_page)

def main():
    print("Loading movies")
    movies, movieId2Idx, movieIdx2Id = loadMovies()
    print("Loading ratings")
    ratings = loadRatings()

    # if not os.path.isdir(movie_poster_dir):
    #     print("Downloading movie posters... It will take a long time !")
    #     # os.makedirs(movie_poster_dir)
    #     downloadPosters(movies)

    user_ratings = np.full((len(movies)), -1)
    user_ratings[0] = 5
    user_ratings[100] = 1
    # user_ratings = np.random.randint(0, 5, (len(movies)))

    recommender = UserBasedRecommender(movies, ratings, movieId2Idx)

    webapp.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

    # recommendations = recommender.recommend(user_ratings)
    # recommendation_names = movies.loc[movies['movieId'].isin(recommendations)]['title']
    #



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

    # todo: Parse arguments
    main()

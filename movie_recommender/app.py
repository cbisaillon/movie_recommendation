import sys
from movie_recommender.data.Movie import loadMovies
from movie_recommender.data.Rating import loadRatings
import torch

import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import random

userMovie2Rating = {}
learning_rate = 0.001
weight_decay = 0.00001
n_epochs = 100
identifier = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(identifier)

def main():
    print("Loading movies")
    movies = loadMovies()
    movie_id_to_index = {}

    for movie in movies:
        movie_id_to_index[movie.id] = len(movie_id_to_index.keys())

    print("Loading ratings")
    ratings = loadRatings()
    users = ratings.userId.unique()

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


class NetworkRecommender(nn.Module):
    """
    The network that learns how to make good movie recommendation
    """

    def __init__(self, nb_users, nb_movies, nb_factors=50):
        super(NetworkRecommender, self).__init__()
        self.nb_hidden = 10

        self.userEmbedding = nn.Embedding(nb_users, nb_factors, sparse=True)
        self.movieEmbedding = nn.Embedding(nb_movies, nb_factors, sparse=True)
        self.drop = nn.Dropout(0.02)
        self.hidden1 = nn.Linear(100, 40)
        self.hidden2 = nn.Linear(40, 20)
        self.out = nn.Linear(20, 1)
        self.tanh = nn.Tanh()

    def forward(self, user, movie):
        user_emb = self.userEmbedding(user)
        movie_emb = self.movieEmbedding(movie)

        features = torch.cat([user_emb, movie_emb], 1)

        x = self.drop(features)
        x = self.hidden1(x)
        x = self.hidden2(x)
        out = self.out(x)

        return out

    def _init(self):
        def init(layer):
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

        self.userEmbedding.weight.data.uniform_(-0.05, 0.05)
        self.movieEmbedding.weight.data.uniform_(-0.05, 0.05)
        self.hidden1.apply(init)
        self.hidden2.apply(init)
        self.out.apply(init)

if __name__ == "__main__":
    argv = sys.argv

    # todo: Parse arguments
    main()

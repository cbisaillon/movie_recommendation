import torch
import torch.nn as nn


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

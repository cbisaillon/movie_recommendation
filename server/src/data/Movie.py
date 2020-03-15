import pandas as pd
import os
from urllib.parse import quote
import urllib.request as request
import json

script_path = os.path.dirname(__file__)

file_path = os.path.join(script_path, "../big_files/ml-25m/movies.csv")
movie_api_url = "https://api.themoviedb.org/3/search/movie?api_key=15d2ea6d0dc1d476efbca3eba2b9bbfb&query={query}"
image_api_url = "http://image.tmdb.org/t/p/w500/{img}"
poster_folder = os.path.join(script_path, "posters")


# I won't use this method since free tier Firebase hosting only allow 1GB
def downloadPosters(movies):
    for idx, movie in movies.iterrows():
        query = '+'.join(movie['title'].split(' ')[:-1])  # Remove the year (last item)
        query = quote(query)

        movie_info_url = movie_api_url.replace('{query}', query)

        with request.urlopen(movie_info_url) as response:
            data = json.loads(response.read())

            if len(data['results']) > 0 and data['results'][0]["poster_path"] is not None:
                image_url = image_api_url.replace("{img}", data['results'][0]["poster_path"])

                # Download the image and save it
                if not os.path.isdir(poster_folder):
                    os.makedirs(poster_folder)

                with open(os.path.join(poster_folder, str(movie['movieId']) + ".jpg"), 'wb') as file:
                    file.write(request.urlopen(image_url).read())


def loadMovies():
    """
    Loads the movies in the dataset
    :return: the list of movies in the dataset
    """
    movies = pd.read_csv(file_path)

    movieId2Idx = {}
    movieIdx2Id = []

    for movieId in movies['movieId']:
        if not movieId in movieId2Idx.keys():
            movieId2Idx[int(movieId)] = len(movieId2Idx.keys())
            movieIdx2Id.append(movieId)

    return movies, movieId2Idx, movieIdx2Id

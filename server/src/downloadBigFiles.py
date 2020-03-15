import urllib.request
import os
import zipfile

script_dir = os.path.dirname(__file__)

movie_dataset = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
save_location = os.path.join(script_dir, "big_files/ml-25m.zip")
export_location = os.path.join(script_dir, "big_files/")

print("Downloading Movie lens dataset...")

urllib.request.urlretrieve(movie_dataset, save_location)

print("Unzipping dataset...")
with zipfile.ZipFile(save_location, 'r') as zip_ref:
    zip_ref.extractall(export_location)
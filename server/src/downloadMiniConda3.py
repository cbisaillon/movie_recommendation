import urllib.request
import os

script_location = os.path.dirname(__file__)

url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
save_loc = os.path.join(script_location, "../miniconda3.sh")

print("Downloading MiniConda3...")

urllib.request.urlretrieve(url, save_loc)
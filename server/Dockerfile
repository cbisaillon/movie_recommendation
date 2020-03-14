FROM python:3.7

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app

RUN pip install Flask gunicorn

CMD exec miniconda3.sh -b
CMD exec export PATH=$PATH:~/miniconda3/bin
CMD exec conda create -y --name movie-recommendation python==3.7
CMD exec conda install -f -y -q --name movie-recommendation -c conda-forge --file requirements.txt
CMD exec conda activate movie-recommendation
COPY . /opt/app

COPY src/ /app
WORKDIR /app

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
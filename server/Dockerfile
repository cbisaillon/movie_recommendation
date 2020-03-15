FROM python:3.7

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app

RUN pip install Flask gunicorn wheel pandas
RUN pip install -r requirements.txt

CMD exec miniconda3.sh -b
CMD exec export PATH=$PATH:~/miniconda3/bin
CMD exec conda create -y --name movie-recommendation python==3.7
CMD exec conda activate movie-recommendation
CMD exec conda install pandas
#CMD exec conda install -f -y -q --name movie-recommendation -c conda-forge --file requirements.txt
COPY . /opt/app

COPY src/ /app
WORKDIR /app

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:webapp
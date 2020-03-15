FROM python:3.7

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app

RUN pip install Flask gunicorn wheel pandas && pip install -r requirements.txt

CMD exec python src/downloadMiniConda3.py
CMD exec miniconda3.sh -b \
&& export PATH=$PATH:~/miniconda3/bin \
&& exec conda create -y --name movie-recommendation python==3.7 \
&& exec conda activate movie-recommendation \
&& exec conda install pandas

#CMD exec conda install -f -y -q --name movie-recommendation -c conda-forge --file requirements.txt
COPY . /opt/app

COPY src/ /app
WORKDIR /app

ENV PORT 8080

CMD exec python downloadBigFiles.py
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:webapp
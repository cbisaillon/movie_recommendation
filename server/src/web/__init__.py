from flask import Flask

webapp = Flask(__name__)

from server.src.web import routes
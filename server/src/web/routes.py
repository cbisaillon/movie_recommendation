from flask import Flask, render_template, make_response
import time
from flask import Blueprint
import os

script_folder = os.path.dirname(__file__)


def format_server_time():
    server_time = time.localtime()
    return time.strftime("%I:%M:%S %p", server_time)


index_page = Blueprint('index_page', __name__, template_folder=os.path.join(script_folder, 'templates'))


@index_page.route('/')
@index_page.route('/index')
def index():
    context = {'server_time': format_server_time()}
    return render_template('index.html', context=context)

from server.src.web import webapp
from flask import Flask, render_template, make_response
import time


def format_server_time():
    server_time = time.localtime()
    return time.strftime("%I:%M:%S %p", server_time)


@webapp.route('/')
@webapp.route('/index')
def index():
    context = {'server_time': format_server_time()}
    return render_template('index.html', context=context)

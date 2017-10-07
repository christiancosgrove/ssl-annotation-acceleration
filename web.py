import numpy as np
from flask import Flask
from flask import request, redirect, send_from_directory
import os
import time
import threading
from data import DataReader

app = Flask(__name__)

reader = None

# num_images = 

@app.route('/')
def index():
    print('test')

def handle_results():
    print('hi')


@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory('images', path)

def start_server(data_reader):
    reader = data_reader
    app.run(host='0.0.0.0', port=5000)
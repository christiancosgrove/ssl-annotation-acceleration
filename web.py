import numpy as np
from flask import Flask
from flask import request, redirect
import os
import time
import threading
from data import DataReader

app = Flask(__name__)

reader = None

@app.route('/')
def index():
    print('test')

def handle_results():
    print('hi')

def start_server(data_reader):
    reader = data_reader
    app.run(host='0.0.0.0', port=5000)
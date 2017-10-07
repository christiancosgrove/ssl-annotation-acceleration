import numpy as np
from flask import Flask
from flask import request, redirect, send_from_directory
import os
import time
import threading
from data import DataReader
import base64
from model import SSLModel

app = Flask(__name__)

reader = None
model = None

num_images = 20

USE_SANDBOX = True
post_endpoint = 'https://workersandbox.mturk.com/mturk/externalSubmit' if USE_SANDBOX else 'https://www.mturk.com/mturk/externalSubmit'

@app.route('/')
def index():

    css = ''
    js = ''
    with open('style.css') as cssfile:
        css = cssfile.read()
    with open('interface.js') as jsfile:
        js = jsfile.read().replace('NUM_IMAGES', str(num_images))

    html = '<head><style>{}</style><script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script><script>{}</script></head><body>'.format(css, js)


    html += '<form id="form1" action="' + post_endpoint + '" method="post">'
    indices, names, predictions = reader.get_labeling_batch(num_images, model)
    html += '<input type="hidden" name="c" value="{}"></input>'.format(base64.urlsafe_b64encode(predictions).decode('ascii'))
    html += '<input type="hidden" name="s" value="{}"></input>'.format(base64.urlsafe_b64encode(indices).decode('ascii'))

    for i in range(num_images):
        style = "display:none" if i != 0 else ""
        iname = "i" + str(i)
        html += '<div style="' + style + '">' + reader.class_list[predictions[i]] + '<input type="checkbox" id="{}" name="{}"><label for="{}"><img src="{}" /></label></div>'.format(iname, iname, iname, names[i])
    html += '</form></body></html>'
    return html

def handle_results():
    print('hi')


@app.route('/submit', methods=['POST'])
def submit():
    arg = request.get_json(force=True)
    imgs = np.fromstring(base64.urlsafe_b64decode(arg['s']), dtype=np.int32, count=num_images)
    categories = np.fromstring(base64.urlsafe_b64decode(arg['c']), dtype=np.int32, count=num_images)
    responses = np.array(arg['responses'])

    #todo: check if assignment was valid

    for i, r in enumerate(responses):
        if r == 1:
            print('positive label {}'.format(imgs[i]))
            reader.label_image_positive(imgs[i], categories[i])
        elif r == -1:
            reader.label_image_negative(imgs[i], categories[i])

    return 'done'


@app.route('/images/<path:path>')
def send_images(path):
    return send_from_directory('images', path)

def start_server(data_reader, ssl_model):
    global reader
    global model
    reader = data_reader
    model = ssl_model
    app.run(host='0.0.0.0', port=5000)
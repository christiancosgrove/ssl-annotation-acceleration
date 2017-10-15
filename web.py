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
# model = None

num_images = 32

num_images_groundtruth = 16


USE_SANDBOX = True
post_endpoint = 'https://workersandbox.mturk.com/mturk/externalSubmit' if USE_SANDBOX else 'https://www.mturk.com/mturk/externalSubmit'

@app.route('/')
def index():

    css = ''
    js = ''
    with open('style.css') as cssfile:
        css = cssfile.read()
    with open('interface.js') as jsfile:
        js = jsfile.read().replace('NUM_IMAGES', str(num_images+num_images_groundtruth))

    html = '<head><style>{}</style><script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script><script>{}</script></head><body>'.format(css, js)


    html += '<form id="form1" action="' + post_endpoint + '" method="post">'
    indices, names, predictions = reader.get_labeling_batch(num_images)
    gt_indices, gt_names, gt_predictions, gt_positives = reader.get_labeling_batch_groundtruth(num_images_groundtruth)
    if indices is None or gt_indices is None:
        return "No work right now..."

    total_indices = np.append(indices, gt_indices)
    total_names = names + gt_names
    total_predictions = np.append(predictions, gt_predictions)

    total_positives = np.append([0] * len(indices), gt_positives)

    perm = np.random.permutation(len(total_indices))


    total_indices = total_indices[perm]
    total_names = [total_names[i] for i in perm]
    total_predictions = total_predictions[perm]
    total_positives = total_positives[perm]

    html += '<input type="hidden" name="c" value="{}"></input>'.format(base64.urlsafe_b64encode(total_predictions).decode('ascii'))
    html += '<input type="hidden" name="s" value="{}"></input>'.format(base64.urlsafe_b64encode(total_indices).decode('ascii'))
    html += '<input type="hidden" name="p" value="{}"></input>'.format(base64.urlsafe_b64encode(total_positives).decode('ascii'))

    for i, ind in enumerate(total_indices):
        style = "display:none" if i != 0 else ""
        iname = "i" + str(i)
        html += '<div style="' + style + '"><span class="heading">Is this a <strong>' + \
            reader.class_list[total_predictions[i]] + '</strong>?</span><br><input type="checkbox" id="{}" name="{}"><label for="{}"><img src="{}" /></label></div>'.format(iname, iname, iname, total_names[i])
    html += '<a class="nextbtn" href="#" onclick="nextItem(false)">No (shortcut <strong>N</strong>)</a>'
    html += '<a class="nextbtn" href="#" onclick="nextItem(true)">Yes (shortcut <strong>M</strong>)</a>'
    html += '<p>Say <strong>no</strong> for any <strong>vehicle interiors</strong>, or any case where <strong>the type of vehicle is unclear</strong>.</p>'
    html += '</form></body></html>'
    return html

@app.route('/submit', methods=['POST'])
def submit():
    arg = request.get_json(force=True)
    imgs = np.fromstring(base64.urlsafe_b64decode(arg['s']), dtype=np.int32, count=num_images+num_images_groundtruth)
    categories = np.fromstring(base64.urlsafe_b64decode(arg['c']), dtype=np.int32, count=num_images+num_images_groundtruth)
    positives = np.fromstring(base64.urlsafe_b64decode(arg['p']), dtype=np.int32, count=num_images+num_images_groundtruth)
    responses = np.array(arg['responses'])

    #todo: check if assignment was valid
    gt_count = 0
    gt_correct=0

    for i, r in enumerate(responses):
        if positives[i] == 1 or positives[i] == -1:
            gt_count+=1
            if positives[i] == r:
                gt_correct+=1
        else:
            if r == 1:
                print('positive label {}'.format(imgs[i]))
                reader.label_image_positive(imgs[i], categories[i])
            elif r == -1:
                reader.label_image_negative(imgs[i], categories[i])

    print("{}/{} correct".format(gt_correct, gt_count))

    return 'done'


@app.route('/images/<path:path>')
def send_images(path):
    return send_from_directory('images', path)

def start_server(data_reader):
    global reader
    # global model
    reader = data_reader
    # model = ssl_model
    app.run(host='0.0.0.0', port=5000)
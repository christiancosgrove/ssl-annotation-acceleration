import numpy as np
from flask import Flask
from flask import request, redirect, send_from_directory
import os
import time
import threading
from data import DataReader
import base64
from model import SSLModel
import json

app = Flask(__name__)

reader = None
# model = None

num_images = 32
num_images_groundtruth = 8
groundtruth_threshold = 0.75 #acceptance threshold for performance on ground-truth images


USE_SANDBOX = True
post_endpoint = 'https://workersandbox.mturk.com/mturk/externalSubmit' if USE_SANDBOX else 'https://www.mturk.com/mturk/externalSubmit'

@app.route('/')
def index():

    css = ''
    js = ''
    with open('style_cluster.css') as cssfile:
        css = cssfile.read()
    with open('interface_cluster.js') as jsfile:
        js = jsfile.read().replace('NUM_IMAGES', str(num_images+num_images_groundtruth))

    html = '<head><style>{}</style><script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script><script>{}</script></head><body>'.format(css, js)


    html += '<form id="form1" action="' + post_endpoint + '" method="post">'
    total_indices, total_names, total_predictions, total_clusters, total_positives = reader.get_labeling_batch_clustered(num_images, num_images_groundtruth)
    if total_indices is None or total_names is None or total_predictions is None or total_clusters is None or total_positives is None:
        return "No work right now..."
    cluster_keys = np.zeros(total_clusters.shape[0], dtype=np.int64)
    for (i, j), cluster in np.ndenumerate(total_clusters):
        cluster_keys[i] += np.power(2, (total_clusters.shape[1] - 1 - j)) * np.int64(cluster)

    print("cluster keys")
    print(cluster_keys)
    print(total_clusters)
    perm = np.array(sorted(list(range(len(total_indices))), key=lambda i: cluster_keys[i]))


    total_indices = total_indices[perm]
    total_names = [total_names[i] for i in perm]
    total_predictions = total_predictions[perm]
    total_positives = total_positives[perm]
    total_clusters = total_clusters[perm]

    html += '<input type="hidden" name="c" value="{}"></input>'.format(base64.urlsafe_b64encode(total_predictions).decode('ascii'))
    html += '<input type="hidden" name="s" value="{}"></input>'.format(base64.urlsafe_b64encode(total_indices).decode('ascii'))
    html += '<input type="hidden" name="p" value="{}"></input>'.format(base64.urlsafe_b64encode(total_positives).decode('ascii'))
    html += '<input type="hidden" name="l" value="{}"></input>'.format(json.dumps(total_clusters.tolist()))

    html += '<div><span class="heading">Is this a <strong>' + reader.class_list[total_predictions[0]] + '</strong>?<br>'
    line_width = int(np.ceil(np.sqrt(len(total_indices))))
    for i, ind in enumerate(total_indices):
        iname = "i" + str(i)
        html += '<input type="checkbox" id="{}" name="{}"><label for="{}"><img id="{}" src="{}" /></label></div>'.format(iname, iname, iname, "m" + str(i), total_names[i])
        if i % line_width == (line_width - 1):
            html += '<br>'
    html += '<a class="nextbtn" href="#" onclick="nextItem(false)">No (shortcut <strong>N</strong>)</a>'
    html += '<a class="nextbtn" href="#" onclick="nextItem(true)">Yes (shortcut <strong>M</strong>)</a>'
    # html += '<p>Say <strong>no</strong> for any <strong>vehicle interiors</strong>, or any case where <strong>the type of vehicle is unclear</strong>.</p>'
    # html += '<p>Say <strong>no</strong> for any <strong>vehicle interiors</strong>, or any case where <strong>the type of vehicle is unclear</strong>.</p>'
    html += '<p>Categories: {}</p>'.format(', '.join(reader.class_list))
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
    if gt_correct >= groundtruth_threshold * gt_count:
        for i, r in enumerate(responses):
            if positives[i] == 0:
                if r == 1:
                    print('positive label {}'.format(imgs[i]))
                    reader.label_image_positive(imgs[i], categories[i])
                elif r == -1:
                    reader.label_image_negative(imgs[i], categories[i])
        print("{}/{} correct, accept".format(gt_correct, gt_count))
    else:
        print("{}/{} correct, reject".format(gt_correct, gt_count))

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
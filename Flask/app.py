from flask import Flask, request, jsonify, render_template, url_for, redirect, flash, Response, session, app
import flask
import pandas as pd
from ast import literal_eval
from controller import *
import matplotlib.pyplot as plt
from datetime import timedelta
import random, string
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
from os import path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = r'C:\Users\91880\Desktop\GOOGLE_DRIVE\Flask_Car_Plate_Recognition_System\venv\Programs\Images_upload'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
unique_identify_user = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
new_file_name = ''


@app.route('/', methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        user_name = 'Team ML Pipelining'
        operation = ['Local System', 'Click Picture']
        return render_template('index.html', user_name=user_name, operation=operation)
    elif request.method == "POST":
        # unique_identify_user = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        if request.form["operation_choice"] == 'Local System':
            return render_template('uploading.html')
        elif request.form["operation_choice"] == 'Click Picture':
            return render_template('video.html')


@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        file = request.files['file']

        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(url_for('upload'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # print(os.path.splitext(filename)[0])
            # print(os.path.splitext(filename)[1])
            global new_file_name
            new_file_name = str(unique_identify_user) + '_' + str(os.path.splitext(filename)[0]) + str(os.path.splitext(filename)[1])
            saved_file = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
            print(saved_file)
            file.save(os.path.join(saved_file))

            # AWS CONNECT
            # upload_aws(img_upload)
            return redirect(url_for('index'))
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(url_for('upload'))


@app.route('/video', methods=["POST"])
def video():
    if request.method == "POST":
        print('INSIDE VIDEO')
        file_name = video_capture(app)
        global new_file_name
        new_file_name = file_name
        print(new_file_name)
        return redirect(url_for('index'))


@app.route('/output', methods=["POST"])
def output():
    if request.method == "POST":
        global new_file_name
        filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)

        file_path = yolo_bounding_box(filename)
        # read the data from the file
        print(file_path)
        with open(file_path, 'rb') as infile:
            buf = infile.read()

        # use numpy to construct an array from the bytes
        x = np.fromstring(buf, dtype='uint8')

        # decode the array into an image
        img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)

        # show it
        cv2.imshow("Bounding Box for the image", img)
        cv2.waitKey(0)
        return redirect(url_for('index'))


def about():
    return render_template("about.html")


# run Flask app
if __name__ == "__main__":
    app.run(debug=True)

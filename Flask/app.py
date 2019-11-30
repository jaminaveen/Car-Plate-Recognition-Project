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

app = Flask(__name__)

UPLOAD_FOLDER = r'C:\Users\91880\Desktop\GOOGLE DRIVE\Flask Car Plate Recognition System\venv\Programs\Images_upload'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
unique_identify_user = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
new_file_name = ''


@app.route('/', methods=["GET", "POST"])
# @app.route("/index", methods=["GET", "POST"])
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
        print(file)
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
            file.save(os.path.join(saved_file))
            return redirect(url_for('index'))
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(url_for('upload'))


@app.route('/output', methods=["POST"])
def output():
    if request.method == "POST":
        global new_file_name
        filename = os.path.join(app.config['UPLOAD_FOLDER'], new_file_name)
        # print(filename)
        # print(new_file_name)
        img = mpimg.imread(filename)
        imgplot = plt.imshow(img)
        plt.show()
        return redirect(url_for('index'))


@app.route('/video', methods=["POST"])
def video():
    if request.method == "POST":
        file_name = video_capture(app)
        global new_file_name
        new_file_name = file_name
        print(new_file_name)
        return redirect(url_for('index'))


def about():
    return render_template("about.html")


# run Flask app
if __name__ == "__main__":
    app.run(debug=True)

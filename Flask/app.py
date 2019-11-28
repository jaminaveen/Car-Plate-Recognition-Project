from flask import Flask, request, jsonify, render_template, url_for, redirect, flash
import pandas as pd
from ast import literal_eval
from controller import *
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = r'C:\Users\91880\Desktop\GOOGLE DRIVE\Flask Car Plate Recognition System\venv\Programs\Images_upload'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/', methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        user_name = 'Team AMD'
        operation = ['Local System', 'Click Picture']
        return render_template('index.html', user_name=user_name, operation=operation)
    elif request.method == "POST":
        if request.form["operation_choice"] == 'Local System':
            return render_template('uploading.html')


@app.route('/upload', methods=["POST"])
def upload():
    print('dcdc')
    if request.method == "POST":
        file = request.files['file']
        print(file)
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(url_for('upload'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('index'))
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(url_for('upload'))


def about():
    return render_template("about.html")


# run Flask app
if __name__ == "__main__":
    app.run(debug=True)

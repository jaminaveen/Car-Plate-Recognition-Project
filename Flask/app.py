from flask import Flask, request, render_template, url_for, redirect, flash, app
from controller import allowed_file, video_capture, yolo_bounding_box, plate_segmentation, predict_cnn

import random, string
import os.path
import cv2 as cv
import os
from werkzeug.utils import secure_filename
import numpy as np
#import matplotlib.pyplot as plt



app = Flask(__name__)

UPLOAD_FOLDER = r'C:\Users\91880\Desktop\GOOGLE_DRIVE\untitled\Programs\Images_upload'
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
            new_file_name = str(unique_identify_user) + '_' + str(os.path.splitext(filename)[0]) + str(
                os.path.splitext(filename)[1])
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

        img, digits = plate_segmentation(file_path)
        out = predict_cnn(digits)
        
        out[0] = ''.join(out[0])
        out[1] = ''.join(out[1])
        print(img)
        print(out[0])
        
        font = cv.FONT_HERSHEY_SIMPLEX 
        fontScale = 1
        color = (255, 0, 0)
        thickness = 3
        file_path_final = 'C:/Users/91880/Desktop/GOOGLE_DRIVE/untitled/Programs/Images_upload/'
        with open(file_path_final + new_file_name, 'rb') as infile:
            buf = infile.read()

# use numpy to construct an array from the bytes
        x = np.frombuffer(buf, dtype='uint8')

# decode the array into an image
        img = cv.imdecode(x, cv.IMREAD_UNCHANGED)
        img = cv.resize(img, (500,500))
        cv.putText(img,out[0], (250,250),font,fontScale, color, thickness) 
        #cv.rectangle(img, (500,500), (200,200), (255, 255, 255) , 4)
# show it
        cv.imshow("some window", img)
        cv.waitKey(0)
        #cv.putText(image_output,out[0], (10,10), font, fontScale, color, thickness,)
        #
        #cv2.putText(img, 'TITLE1',(100,100),out[1])
        #
        # with open(file_path, 'rb') as infile:
        #     buf = infile.read()
        #
        # # use numpy to construct an array from the bytes
        # x = np.fromstring(buf, dtype='uint8')
        #
        # # decode the array into an image
        # img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
        #
        # # show it
        # cv2.imshow("Bounding Box for the image", img)
        # cv2.waitKey(0)
        # plt.imshow(img)
        # plt.imshow(digits)
        return redirect(url_for('index'))


def about():
    return render_template("about.html")


# run Flask app
if __name__ == "__main__":
    app.run(debug=False)

import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from numpy.linalg import inv
import scipy.stats as ss
import tkinter
import tkinter.filedialog as tkFileDialog
import os
from werkzeug.utils import secure_filename
import cv2
import random, string
from boto3.s3.transfer import S3Transfer
import boto3

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def video_capture(app):
    unique_identify_user = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                file_name = str(unique_identify_user) + str('.png')
                file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                cv2.imwrite(filename=file_name, img=frame)
                webcam.release()
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    return file_name


def upload_aws(SOURCE_FILENAME):
    aws_id = 'AKIAJ4CSH5Z3BFHREHQQ'
    aws_secret = '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'
    S3 = boto3.client('s3', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret)
    S3.upload_file(SOURCE_FILENAME,'info7374', SOURCE_FILENAME)

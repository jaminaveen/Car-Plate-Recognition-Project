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

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def video_capture(app):
    unique_identify_user = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
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

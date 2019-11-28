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


def prediction():
    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window

    currdir = os.getcwd()
    tempdir = tkFileDialog.askdirectory(parent=root, initialdir=currdir, title='Please select a directory')
    if len(tempdir) > 0:
        print("You chose %s" % tempdir)
        return tempdir
    else:
        return None


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import argparse
import configparser
import os

import boto3
import numpy as np
import cv2 as cv
import CPR_utils as util
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


local_tmp = '../predictions_tmp'
mturk_tmp = '../mturk_tmp'

def get_digits(path):
    return pickle.load(open(path, "rb"))

def get_onehot_encoder():
    # one hot encoding
    alphabets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    classes = []
    for a in alphabets:
        classes.append([a])
    ohe = OneHotEncoder(handle_unknown='ignore', categorical_features=None)
    ohe.fit(classes)

    return ohe

def predict_one_plate(digits, ohe, model, fname, confidence_threshold = 0, mturk_confidence_threshold = 0.5):
    prediction = {}
    preds = []
    confidences =[]

    for dig in digits:
        d = np.reshape(dig, (1, 28, 28, 1))
        out = model.predict(d)
        # Get max pre arg
        p = []
        confidence = 0
        for i in range(len(out)):
            z = np.zeros(36)
            z[np.argmax(out[i])] = 1.
            confidence = max(out[i])
            p.append(z)
        prediction = np.array(p)
        pred = ohe.inverse_transform(prediction)

        char_pred = pred[0][0] #A
        if confidence <= mturk_confidence_threshold:
            # save dig to local first
            char_save_name = fname + "_" + char_pred + "_" + str(confidence).replace('.','') + ".png"
            dig_path = os.path.join(mturk_tmp, char_save_name)
            cv.imwrite(dig_path, dig)

            obj_name =f'{char_pred}/' + char_save_name
            client.upload_file(dig_path, config['buckets']['predicted'], obj_name)

        if confidence > confidence_threshold:
            preds.append(pred[0][0])  # pred[0][0] is predicted character  eg: preds.append('A')
            confidences.append(confidence)  # confidence
            print('Prediction : ' + str(pred[0][0]) + ' , Confidence : ' + str(confidence))

    prediction['preds'] = preds #predicted character list
    prediction['confidence'] = confidences #confidence list
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide amazon credentials')
    parser.add_argument('--conf', default="../conf/config.cfg", help='the path of config.cfg')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.conf)

    aws_access_key_id = config['AWS_access_credentials']['aws_access_key_id']
    aws_secret_access_key = config['AWS_access_credentials']['aws_secret_access_key']

    client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    # Load model
    model_key = "cnn_classifier.h5"
    local_path = "../conf/"

    model_local_path = os.path.join(local_path, model_key)

    #download from s3 and update local model
    client.download_file(config['buckets']['models'], model_key, model_local_path)
    model = load_model(model_local_path)

    digit_urls = util.get_objecturls_from_bucket(client, config['buckets']['digits'])

    # predict
    ohe = get_onehot_encoder()

    for url in digit_urls:
        fname = url.split(".com/")[1].split('.')[0]

        digits = get_digits(url)
        prediction = predict_one_plate(digits, ohe, model, fname)

        with open(os.path.join(local_tmp, fname), 'wb') as pf:
            pickle.dump(prediction, pf)

    # upload to s3
    files_to_upload = os.listdir(local_tmp)
    for f in files_to_upload:
        client.upload_file(f, config['buckets']['cnnpredictions'], os.path.basename(f))



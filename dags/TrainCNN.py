import argparse
import configparser

import boto3
import pandas as pd
import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from matplotlib import pyplot as plt
import CPR_utils as util



def prepare():
    # Create dictionary for alphabets and related numbers
    alphabets_dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
                 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29:'3',
                 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}

    alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    dataset_classes = []
    for cls in alphabets:
        dataset_classes.append([cls])

    label_list = []
    for l in labels:
        label_list.append([l])

    # One hot encoding format for output
    ohe = OneHotEncoder(handle_unknown='ignore', categorical_features=None)
    ohe.fit(dataset_classes)
    labels_ohe = ohe.transform(label_list).toarray()

    return labels_ohe


def build_model():
    # CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(36, activation='softmax'))

    print(model.summary())

    return model

def visualization():
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide amazon credentials')
    parser.add_argument('--conf', default="../conf/config.cfg", help='the path of config.cfg')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.conf)

    aws_access_key_id = config['AWS_access_credentials']['aws_access_key_id']
    aws_secret_access_key = config['AWS_access_credentials']['aws_secret_access_key']

    client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)


    #load data
    data_path = 'data.pickle'
    labels_path = 'labels.pickle'
    with open('../tmp/data.pickle', 'wb') as f:
        client.download_file(config['buckets']['chardata'], data_path, f)

    with open('./tmp/labels.pickle', 'wb') as f:
        client.download_file(config['buckets']['chardata'], labels_path, f)

    d = open("../tmp/data.pickle", "rb")
    l = open("../tmp/labels.pickle", "rb")
    data = pickle.load(d)
    labels = pickle.load(l)

    labels_ohe = prepare()

    data = np.array(data)
    labels = np.array(labels)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(data, labels_ohe, test_size=0.20, random_state=42)

    X_train = X_train.reshape(29260, 28, 28, 1)
    X_test = X_test.reshape(7316, 28, 28, 1)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    model = build_model()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64)

    model_save_path = '../conf/cnn_classifier.h5'
    model.save(model_save_path)
    client.upload_file(model_save_path, config['buckets']['models'], os.path.basename(model_save_path))

    print("model has been saved and uploaded!")





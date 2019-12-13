import os
import fnmatch
import cv2
import numpy as np
import string
import time
import boto3
import configparser

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse




def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            #print(char)
            dig_lst.append(char_list.index(' '))
    return dig_lst


def download_dir(prefix, local, bucket, client):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket':bucket,
        'Prefix':prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)


def train_test_split(path, valid_split_perc = 10):
    training_img = []
    training_txt = []
    train_input_length = []
    train_label_length = []
    orig_txt = []

    # lists for validation dataset
    valid_img = []
    valid_txt = []
    valid_input_length = []
    valid_label_length = []
    valid_orig_txt = []

    max_label_len = 0

    i = 1
    flag = 0

    for root, dirnames, filenames in os.walk(path):
        # print(filenames)

        for f_name in fnmatch.filter(filenames, '*.png'):
            # print(f_name)
            # read input image and convert into gray scale image
            # print(os.path.join(root, f_name))
            img = cv2.cvtColor(cv2.imread(os.path.join(root, f_name)), cv2.COLOR_BGR2GRAY)

            # convert each image of shape (32, 128, 1)
            w, h = img.shape
            # print(w,h)
            if h > 128 or w > 32:
                img = cv2.resize(img, (128, 32), interpolation=cv2.INTER_AREA)
            if w < 32:
                add_zeros = np.ones((32 - w, h)) * 255
                img = np.concatenate((img, add_zeros))

            if h < 128:
                add_zeros = np.ones((32, 128 - h)) * 255
                img = np.concatenate((img, add_zeros), axis=1)
            img = np.expand_dims(img, axis=2)
            # print(img.shape)

            # Normalize each image
            img = img / 255.

            # get the text from the image
            txt = f_name.split('_')[0]
            # print(txt)
            # compute maximum length of the text
            if len(txt) > max_label_len:
                max_label_len = len(txt)

            # split the 150000 data into validation and training dataset as 10% and 90% respectively
            if i % valid_split_perc == 0:
                valid_orig_txt.append(txt)
                valid_label_length.append(len(txt))
                valid_input_length.append(31)
                valid_img.append(img)
                valid_txt.append(encode_to_labels(txt))
            else:
                orig_txt.append(txt)
                train_label_length.append(len(txt))
                train_input_length.append(31)
                training_img.append(img)
                training_txt.append(encode_to_labels(txt))

                # break the loop if total data is 150000
            if i == len(os.listdir(path+'/'+os.listdir(path)[0])):
                flag = 1
                break
            i += 1
        if flag == 1:
            break

    train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=len(char_list))
    valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value=len(char_list))

    training_img = np.array(training_img)
    train_input_length = np.array(train_input_length)
    train_label_length = np.array(train_label_length)

    valid_img = np.array(valid_img)
    valid_input_length = np.array(valid_input_length)
    valid_label_length = np.array(valid_label_length)

    return train_padded_txt,valid_padded_txt,training_img,train_input_length,train_label_length, valid_img, valid_input_length, valid_label_length, max_label_len

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_crnn_model(train_padded_txt,valid_padded_txt,training_img,train_input_length,train_label_length, valid_img, valid_input_length, valid_label_length, max_label_len,batch_size = 64, epochs =10 ):
    # input with shape of height=32 and width=128
    inputs = Input(shape=(32, 128, 1))

    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

    outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

    # model to be used at test time
    act_model = Model(inputs, outputs)

    print(act_model.summary())
    labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

    # model to be used at training time
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    filepath = "best_crnn_model.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], y=np.zeros(len(training_img)),
              batch_size=batch_size, epochs=epochs, validation_data=(
        [valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]), verbose=1,
              callbacks=callbacks_list)

    return act_model, model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Please provide amazon credentials')
    parser.add_argument('--conf', default="../conf/config.cfg", help='the path of config.cfg')
    parser.add_argument('--local', default="../data/crnn_train", help='train storage location')
    parser.add_argument('--prefix', default="License-plate", help='storage location in the bucket')
    parser.add_argument('--valid_split', default=10,help = 'validation split percentage')
    parser.add_argument('--epochs', default=10 ,help = 'Number of epochs')
    parser.add_argument('--batch_size', default=64, help='train batch size')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.conf)
    aws_access_key_id = config['AWS_access_credentials']['aws_access_key_id']
    aws_secret_access_key = config['AWS_access_credentials']['aws_secret_access_key']
    bucket_name = config['Plate_Dataset_Bucket']['bucket_name']
    object_name = config['Plate_Dataset_Bucket']['object_name']
    s3_client = boto3.client('s3',
                             aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key)
    tf.logging.set_verbosity(tf.logging.ERROR)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # ignore warnings in the output
    # total number of our output classes: len(char_list)
    char_list = string.ascii_letters + string.digits + ' '
    download_dir(args.prefix, args.local, bucket_name, client=s3_client)
    train_padded_txt,valid_padded_txt,training_img,train_input_length,train_label_length, valid_img, valid_input_length, valid_label_length, max_label_len = train_test_split(args.local,args.valid_split)
    act_model, model = build_crnn_model(train_padded_txt,valid_padded_txt,training_img,train_input_length,train_label_length, valid_img, valid_input_length, valid_label_length, max_label_len,batch_size = args.batch_size, epochs = args.epochs )





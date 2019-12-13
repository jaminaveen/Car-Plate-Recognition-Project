import argparse
import configparser
import os
import pickle
import CPR_utils as util

import boto3
import cv2 as cv

def prepare(dataset_input_path):
    data = []  # List of images
    labels = []  # List of labels

    # Load all directory
    for root, dirs, files in os.walk(dataset_input_path):

        # Filter every folder
        for dir in dirs:
            print(" Class : \t \t " + dir)
            # Filter all files in the directory
            for filename in os.listdir(dataset_input_path + "/" + dir):
                # Make sure that our file is text
                if filename.endswith('.jpg'):
                    img = cv.imread(dataset_input_path + "/" + dir + "/" + filename)
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    data.append(gray)
                    labels.append(dir)

    # Save test data and labels
    data_path = "../tmp/data.pickle"
    labels_path = "../tmp/labels.pickle"
    pickle.dump(data, open(data_path, "wb"))
    pickle.dump(labels, open(labels_path, "wb"))

    print('Length data : ' + str(len(data)))
    print('Length labels : ' + str(len(labels)))
    print('Processs finished !')

    return [data_path, labels_path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide amazon credentials')
    parser.add_argument('--conf', default="../conf/config.cfg", help='the path of config.cfg')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.conf)

    aws_access_key_id = config['AWS_access_credentials']['aws_access_key_id']
    aws_secret_access_key = config['AWS_access_credentials']['aws_secret_access_key']

    client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    dataset_input_path = '../dataset'
    if not os.path.exists(dataset_input_path):
        os.makedirs(dataset_input_path)

    paths = prepare(dataset_input_path)

    for f in paths:
        client.upload_file(f, config['buckets']['chardata'], os.path.basename(f))

    print("All character files in local have been prepared and uploaded to S3.")
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
from IPython.display import Image
from matplotlib import pyplot as plt
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

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
    S3.upload_file(SOURCE_FILENAME, 'info7374', SOURCE_FILENAME)


confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 416  # 608     #Width of network's input image
inpHeight = 416  # 608     #Height of network's input image

# input_path = 'Pa140025.jpg'
is_video = False
classesFile = "./conf/classes.names";

classes = None


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, frame):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
    # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    cropped = None
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4] > confThreshold:
                print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        # calculate bottom and right
        bottom = top + height
        right = left + width

        # crop the plate out
        cropped = frame[top:bottom, left:right].copy()
        # drawPred
        print('type', type(cropped))
        drawPred(classIds[i], confidences[i], left, top, right, bottom, frame)
    print('type', type(cropped))
    return cropped


def yolo_bounding_box(input_path):
    # Process inputs
    # Open the input file
    cap = cv.VideoCapture(input_path)

    if is_video:
        outputFile = input_path + '_yolo_out_py.avi'
    else:
        outputFile = input_path

    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "./conf/darknet-yolov3.cfg";
    modelWeights = "./conf/lapi.weights";

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    if is_video:
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    while cv.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()  # frame: an image object from cv2

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        cropped = postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes

        cv.imwrite('cropped_plate_final.jpg', cropped.astype(np.uint8))
        # save
        cv.imwrite(outputFile, frame.astype(np.uint8))

    return outputFile

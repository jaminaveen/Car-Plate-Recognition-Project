# dependencies
import configparser
import cv2 as cv
import argparse
import numpy as np
import os
import boto3
import CPR_utils as util


### Initialize the parameters

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 416  # 608     #Width of network's input image
inpHeight = 416  # 608     #Height of network's input image

# Load names of classes
classesFile = "../conf/classes.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "../conf/darknet-yolov3.cfg";
modelWeights = "../conf/lapi.weights";

### Restore the nural network
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

### Define the post processing functions

# Get the names of the output layers
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
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (0, 0, 255), cv.FILLED)
    # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
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

    cropped = None
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
        drawPred(classIds[i], confidences[i], left, top, right, bottom, frame)

    # quick fix
    if cropped is None:
        print("cropped is None")
        raise TypeError

    return cropped

def detect_one(input_path, local_tmp, bounded_local_tmp):
    # Open the input file
    cap = cv.VideoCapture(input_path)

    # if is_video:
    #     outputFile = input_path + '_yolo_out_py.avi'
    # else:
    #     outputFile = input_path + '_yolo_out_py.jpg'

    fname = input_path.split(".com/")[1].split('.')[0]

    detected_plate_out_filename = fname + '_yolo_out_plate.jpg'
    bounding_boxed_out_filename = fname + '_yolo_out_boxed.jpg'


    # # Get the video writer initialized to save the output video
    # if is_video:
    #     vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
    #                                 (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    while cv.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()  # frame: an image object from cv2

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is temporarily stored as: ")
            print(f"detected_plate_out_filename: {detected_plate_out_filename}")
            print(f"bounding_boxed_out_filename: {bounding_boxed_out_filename}")
            cv.waitKey(3000)
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        try:
            cropped = postprocess(frame, outs)
            # save cropped
            cv.imwrite(os.path.join(local_tmp, detected_plate_out_filename), cropped.astype(np.uint8))

            # Put efficiency information.
            # The function getPerfProfile returns the overall time for inference(t)
            # and the timings for each of the layers(in layersTimes)
            # t, _ = net.getPerfProfile()
            # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # Write the frame with the detection boxes
            # if is_video:
            #     vid_writer.write(frame.astype(np.uint8))
            # else:
            cv.imwrite(os.path.join(bounded_local_tmp, bounding_boxed_out_filename), frame.astype(np.uint8))

        except TypeError as e:
            print("No plate detected!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide amazon credentials')
    parser.add_argument('--conf', default="../conf/config.cfg", help='the path of config.cfg')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.conf)

    aws_access_key_id = config['AWS_access_credentials']['aws_access_key_id']
    aws_secret_access_key = config['AWS_access_credentials']['aws_secret_access_key']

    client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    img_urls = util.get_objecturls_from_bucket(client,config['buckets']['scraped_car_pic'])

    local_tmp = '../detected_tmp'
    if not os.path.exists(local_tmp):
        os.makedirs(local_tmp)

    bounded_local_tmp = '../detected_bounded_tmp'
    if not os.path.exists(bounded_local_tmp):
        os.makedirs(bounded_local_tmp)

    for img_url in img_urls:
        detect_one(img_url, local_tmp, bounded_local_tmp)

    files_to_upload = os.listdir(local_tmp)
    for f in files_to_upload:
        client.upload_file(os.path.join(local_tmp,f),config['buckets']['detected'], os.path.basename(f))

    print("All files in car_pic have been detected.")
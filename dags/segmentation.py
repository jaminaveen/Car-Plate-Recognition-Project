import argparse
import configparser
import os
import pickle

import boto3
import numpy as np
import cv2 as cv
import CPR_utils as util


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def square(img):
    """
    This function resize non square image to square one (height == width)
    :param img: input image as numpy array
    :return: numpy array
    """

    # image after making height equal to width
    squared_image = img

    # Get image height and width
    h = img.shape[0]
    w = img.shape[1]

    # In case height superior than width
    if h > w:
        diff = h-w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = np.zeros(shape=(h, (diff//2)+1))

        squared_image = np.concatenate((x1, img, x2), axis=1)

    # In case height inferior than width
    if h < w:
        diff = w-h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = np.zeros(shape=((diff//2)+1, w))

        squared_image = np.concatenate((x1, img, x2), axis=0)

    return squared_image

def sort(vector):
    sort = True
    while (sort == True):

        sort = False
        for i in range(len(vector) - 1):
            x_1 = vector[i][0]
            y_1 = vector[i][1]

            for j in range(i + 1, len(vector)):

                x_2 = vector[j][0]
                y_2 = vector[j][1]

                if (x_1 >= x_2 and y_2 >= y_1):
                    tmp = vector[i]
                    vector[i] = vector[j]
                    vector[j] = tmp
                    sort = True

                elif (x_1 < x_2 and y_2 > y_1):
                    tmp = vector[i]
                    vector[i] = vector[j]
                    vector[j] = tmp
                    sort = True
    return vector

def segment_one(img_file_path):
    cap = cv.VideoCapture(img_file_path)
    _, img = cap.read()
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    height = img.shape[0]
    width = img.shape[1]
    area = height * width

    # scale1 = 0.001
    scale1 = 0.01
    scale2 = 0.1
    area_condition1 = area * scale1
    area_condition2 = area * scale2
    # global thresholding
    ret1, th1 = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)

    # Otsu's thresholding
    ret2, th2 = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(imgray, (5, 5), 0)
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # sort contours
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    cropped = dict()
    # cropped = []
    for cnt in contours:
        (x, y, w, h) = cv.boundingRect(cnt)
        distance_center = (2 * x + w) / 2

        if distance_center in cropped:
            pass
        else:
            if area_condition1 < w * h < area_condition2 and w / h > 0.3 and h / w > 1:
                cv.drawContours(img, [cnt], 0, (0, 255, 0), 1)
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                c = th2[y:y + h, x:x + w]
                c = np.array(c)
                c = cv.bitwise_not(c)
                c = square(c)
                c = cv.resize(c, (28, 28), interpolation=cv.INTER_AREA)
                cropped[distance_center] = c
                # cropped.append(c)

    sorted_cropped = []
    for x_center in sorted(cropped):
        sorted_cropped.append(cropped[x_center])

    # cv.imwrite('detection.png', img)
    return img, sorted_cropped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide amazon credentials')
    parser.add_argument('--conf', default="../conf/config.cfg", help='the path of config.cfg')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.conf)

    aws_access_key_id = config['AWS_access_credentials']['aws_access_key_id']
    aws_secret_access_key = config['AWS_access_credentials']['aws_secret_access_key']

    client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    img_urls = util.get_objecturls_from_bucket(client,config['buckets']['detected'])

    for img_url in img_urls:
        corped_bounded_img, digits = segment_one(img_url)

        # save corped_bounded_img locally first
        fname = img_url.split(".com/")[1].split('.')[0]
        corped_bounded_filename = fname + '_corped_bounded.jpg'


        segmented_corped_bounded_tmp = '../segmented_corped_bounded_tmp'
        if not os.path.exists(segmented_corped_bounded_tmp):
            os.makedirs(segmented_corped_bounded_tmp)

        segmented_digits_tmp = '../segmented_digits_tmp'
        if not os.path.exists(segmented_digits_tmp):
            os.makedirs(segmented_digits_tmp)

        digits_file_name = os.path.basename(img_url).split('.')[0] + '.pickle'
        with open(os.path.join(segmented_digits_tmp, digits_file_name),'wb') as f:
            pickle.dump(digits,f)

        digits_to_upload = os.listdir(segmented_digits_tmp)
        # upload digits to s3
        for f in digits_to_upload:
            upload_path = os.path.join(segmented_digits_tmp, f)
            client.upload_file(upload_path, config['buckets']['segmented_digits'], os.path.basename(f))

        cv.imwrite(os.path.join(segmented_corped_bounded_tmp, corped_bounded_filename), corped_bounded_img.astype(np.uint8))
        #
        # files_to_upload = os.listdir(segmented_corped_bounded_tmp)
        # for f in files_to_upload:
        #     print(f)
        #     client.upload_file(f, config['buckets']['segmented_archive'], os.path.basename(f))


# plt.imshow(img)
#
# digits
#
# fig, axs = plt.subplots(len(digits))
# fig.set_size_inches(20,50)
# for ax in range(len(axs)):
#     axs[ax].imshow(digits[ax])

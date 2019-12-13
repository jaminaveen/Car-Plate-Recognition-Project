import argparse
import os

from bs4 import BeautifulSoup
import requests
from urllib.request import Request, urlopen, URLError
import boto3
import configparser


class MyException(Exception):
    pass

def scraping_job(client, config):
    car_images_list = []
    car_plate_images_list = []
    car_plate_images_text_list = []

    headers = requests.utils.default_headers()
    headers.update({'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'})

    for i in range(1):
        url = 'http://platesmania.com/us/gallery-' + str(i)
        req = requests.get(url, headers=headers)
        soup = BeautifulSoup(req.content, 'html.parser')
        car_images = soup.findAll('img', attrs={'class': 'lazyOwl'})
        for car in car_images:
            car_images_list.append(car['data-src'])

        car_plate_images = soup.findAll('img',
                                        attrs={'class': 'img-responsive center-block margin-bottom-10'})

        for car in car_plate_images:
            car_plate_images_list.append(car['src'])
            car_plate_images_text_list.append(car['alt'])
        print('======================== Scraping for page', i, ' has finished ========================')

    # creating unique filename for each image in the car number plates
    file_names = []
    for img in car_plate_images_list:
        file_names.append(img.split('/')[-1])

    plate_local_tmp = '../scraped_plate_tmp/'
    if not os.path.exists(plate_local_tmp):
        os.makedirs(plate_local_tmp)
    file_names = [plate_local_tmp + m + '_' + str(n) for m, n in
                  zip(car_plate_images_text_list, file_names)]

    # downloading the images locally
    for i in range(len(file_names)):
        req = Request(car_plate_images_list[i], headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})

        try:
            response = urlopen(req, None, 10)
        except URLError:
            raise MyException("urlopen timeout")
            continue


        data = response.read() #TODO time out issue
        response.close()
        file_name = car_plate_images_text_list[i]
        output_file = open(file_names[i], 'wb')
        output_file.write(data)
        output_file.close()

    for f in file_names:
        # print(f)
        f_basename = os.path.basename(f)
        client.upload_file(f, config['buckets']['scraped_plate_pic'], f'{f_basename}')

    car_images_list = list(set(car_images_list))
    # car_images_list.remove('http://img03.platesmania.com/191209/s/13915182.jpg') TODO: this link has issue



    carpic_local_tmp = '../scraped_carpic_tmp/'
    if not os.path.exists(carpic_local_tmp):
        os.makedirs(carpic_local_tmp)
    file_name = []
    for i in car_images_list:
        # print(i.split('/')[-1])
        file_name.append(carpic_local_tmp + i.split('/')[-1])


    # downloading the images locally
    for i in range(len(car_images_list)):  # len(car_plate_images_list)):
        # print(car_images_list[i])
        try:
            req = Request(car_images_list[i], headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})

            response = urlopen(req, None, 10)
            data = response.read()
            response.close()
            # file_names = car_plate_images_text_list[i]
            output_file = open(file_name[i], 'wb')
            output_file.write(data)
            output_file.close()
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            print(e)

    for f in file_name:
        # print(f)
        f_basename = os.path.basename(f)
        # print(img_name_in_s3)
        # print(img_name_in_s3)
        # client.upload_file(f, '/data-pipeline-license-plate',img_name_in_s3)
        client.upload_file(f, config['buckets']['scraped_car_pic'], f'{f_basename}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide amazon credentials')
    parser.add_argument('--conf', default="../conf/config.cfg", help='the path of config.cfg')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.conf)

    aws_access_key_id = config['AWS_access_credentials']['aws_access_key_id']
    aws_secret_access_key = config['AWS_access_credentials']['aws_secret_access_key']

    client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    scraping_job(client, config)
import argparse
import boto3
import configparser
from bs4 import BeautifulSoup
import csv
import time
import glob
import pandas as pd
import CPR_utils as util



region_name = 'us-east-1'
endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
# Uncomment this line to use in production
# endpoint = 'https://mturk-requester.us-east-1.amazonaws.com'


def publish_hits(urls, create_hits_in_live = False):
    #uncomment for production setting
    #if len(urls) = 100: 
    if True:
        environments = {
                "live": {
                    "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
                    "preview": "https://www.mturk.com/mturk/preview",
                    "manage": "https://requester.mturk.com/mturk/manageHITs",
                    "reward": "0.00"
                },
                "sandbox": {
                    "endpoint": "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
                    "preview": "https://workersandbox.mturk.com/mturk/preview",
                    "manage": "https://requestersandbox.mturk.com/mturk/manageHITs",
                    "reward": "0.11"
                },
        }
        mturk_environment = environments["live"] if create_hits_in_live else environments["sandbox"]
        
        try:
            mturk_valid_data = pd.read_csv("..\data\mturk_valid.csv")
            hit_already_created = mturk_valid_data['image_s3_url'].tolist()
        except FileNotFoundError:
            with open("..\data\mturk_valid.csv",'w') as f:
                f.close()
            hit_already_created = []
            
        
        hit_ids = {}
            
        for url in urls:
            if url not in hit_already_created:
                response = mturk_client.create_hit(
                    MaxAssignments=1,
                    LifetimeInSeconds=604800,
                    AssignmentDurationInSeconds=900,
                    Reward=mturk_environment['reward'],
                    Title='Answer a simple question',
                    Keywords='question, answer, research',
                    Description='Answer a simple question. Created from mturk-code-samples.',
                    #Question = question_sample,
                    HITLayoutId='36MUXCXBEGS6MVQRB991571NIS8073',
                    HITLayoutParameters= [
                    {
                            'Name': 'image_url',
                            'Value': url
                    },

                    {
                            'Name': 'character_predicted',
                            'Value': url.split('/')[3]

                    },
                    ]
            )
                hit_ids[url]= [response['HIT']['HITId'],url.split('/')[3]]
                #print(response['HIT']['HITId'])
    with open('..\data\hits_meta'+str(time.time())+'.csv', 'w') as f:
        for key in hit_ids.keys():
            f.write("%s,%s,%s\n"%(key,hit_ids[key][0],hit_ids[key][1]))
        f.close()
        
    return hit_ids


def get_all_hit_ids(prefix = '..\data\hits_meta'):
    
    all_hit_ids = []
    for meta_file in glob.glob('..\data\hits_meta*'):
        file = meta_file
        with open(file,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                #print(row[0]," ",row[1])
                all_hit_ids.append([row[0],row[1],row[2]])
            
    return all_hit_ids


def parse_hit_response(hit_id):
    list_assignments = mturk_client.list_assignments_for_hit(HITId = hit_id)
    try:
        soup = BeautifulSoup(list_assignments['Assignments'][0]['Answer'], 'html.parser')
        selected_label = soup.find_all('freetext')[0].text
        label_written = soup.find_all('freetext')[0].text
    except:
        selected_label = ' '
        label_written = ' '
    
    return selected_label, label_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Please provide amazon credentials')
    parser.add_argument('--conf', default="../conf/config.cfg", help='the path of config.cfg')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.conf)

    aws_access_key_id = config['AWS_access_credentials']['aws_access_key_id']
    aws_secret_access_key = config['AWS_access_credentials']['aws_secret_access_key']

    mturk_client = boto3.client(
        'mturk',
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


    urls = util.get_objecturls_from_bucket(s3_client,'segmentedchars')

    published_hit_ids = publish_hits(urls)

    all_hit_ids = get_all_hit_ids()



    with open("..\data\mturk_valid.csv",'w') as f:
        f.write("hitid,image_s3_url,predictedlabel,selected_label,labelwritten\n")
        for hit in all_hit_ids:
            image_s3_url = hit[0]
            hit_id = hit[1]
            predicted_label = hit[2]
            selected_label, label_written = parse_hit_response(hit[1])
            #print(selected_label, label_written)
            f.write(f"{hit_id},{image_s3_url},{predicted_label},{selected_label},{label_written}\n")







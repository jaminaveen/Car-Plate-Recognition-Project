#!/usr/bin/env python
# coding: utf-8

# ## -------- Connecting MTurk with python -------------------

# In[3]:


import boto3
import configparser
from bs4 import BeautifulSoup
import csv
import time
import glob
import pandas as pd

region_name = 'us-east-1'


# Uncomment this line to use in production
# endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
config = configparser.ConfigParser()
config.read('../conf/config.cfg')

aws_access_key_id = config['AWS_access_credentials']['aws_access_key_id']
aws_secret_access_key = config['AWS_access_credentials']['aws_secret_access_key']

endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

# Uncomment this line to use in production
# endpoint = 'https://mturk-requester.us-east-1.amazonaws.com'


# In[4]:


mturk_client = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,
    region_name=region_name,
    aws_access_key_id= aws_access_key_id,
    aws_secret_access_key= aws_secret_access_key,
)

# This will return $10,000.00 in the MTurk Developer Sandbox
#print(mturk_client.get_account_balance()['AvailableBalance'])


# In[5]:


s3_client = boto3.client(
    's3',
    aws_access_key_id= aws_access_key_id,
    aws_secret_access_key= aws_secret_access_key,
)
def upload_segmented_char_to_bucket(s3_client, file,OBJECT_NAME):
    s3_client.upload_file(file, 'segmentedchars', OBJECT_NAME)

upload_segmented_char_to_bucket(s3_client,r'C:\Users\navee\Desktop\My_Masters_NEU\Fall2019\ProductGradeDataPipelines\NumberPlateDetection\ANPR\Licence_plate_recognition\USA_plates\dataset\2\2_6.jpg', r'2/2_6.jpg')


# ## pull all urls

# In[6]:


def get_objecturls_from_bucket(client, bucket):
    """
    params:
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    next_token = ''
    urls = []
    base_kwargs = {
        'Bucket':bucket
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        #print(contents)
        for i in contents:
            k = i.get('Key')
            url = "https://%s.s3.amazonaws.com/%s" % (bucket, k)
            urls.append(url)
        next_token = results.get('NextContinuationToken')
    return urls


# ## create hit

# In[17]:


def publish_hits(urls, hits_meta_file, create_hits_in_live = False):

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


# In[18]:


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
        
    


# In[19]:


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
    


# In[23]:


urls = get_objecturls_from_bucket(s3_client,'segmentedchars')
published_hit_ids = publish_hits(urls, '..\data\hits_meta.csv')
all_hit_ids = get_all_hit_ids()
with open("..\data\mturk_valid.csv",'w') as f:
    f.write("hitid,image_s3_url,predictedlabel,selected_label,labelwritten\n")
    for hit in all_hit_ids:
        image_s3_url = hit[0]
        hit_id = hit[1]
        predicted_label = hit[2]
        selected_label, label_written = parse_hit_response(hit[1])
        #print(selected_label, label_written)
        f.write("%s,%s,%s,%s,%s\n"%(hit_id,image_s3_url,predicted_label,selected_label,label_written))
    f.close()


# In[ ]:





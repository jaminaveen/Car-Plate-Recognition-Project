#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import boto3
import configparser

class aws_util:
    
    def __init__(self, resource):
        config = configparser.ConfigParser()
        config.read('../conf/config.cfg')
        #config.sections()
        self.aws_access_key_id = config['AWS_access_credentials']['aws_access_key_id']
        self.aws_secret_access_key = config['AWS_access_credentials']['aws_secret_access_key']
        self.resource = resource
        self.s3_client = boto3.client(resource,
                    aws_access_key_id = self.aws_access_key_id,
                    aws_secret_access_key = self.aws_secret_access_key)
        
    def download_dir(self, prefix, local, bucket):
        """
        params:
        - prefix: pattern to match in s3
        - local: local path to folder in which to place files
        - bucket: s3 bucket with target contents
        - client: initialized s3 client object
        """
        client = self.s3_client
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


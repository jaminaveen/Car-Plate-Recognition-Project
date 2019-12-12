
def upload_file_to_bucket(client, file, bucket):
    client.upload_file(file, bucket)


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



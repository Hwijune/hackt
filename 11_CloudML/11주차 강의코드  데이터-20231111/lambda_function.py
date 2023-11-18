import sys
sys.path.append('./libs')
import os
import json
import boto3
import base64
import requests
client_id = os.environ['client_id']
client_secret = os.environ['client_secret']

def get_headers(client_id, client_secret):
    encoded = base64.b64encode(('{}:{}'.format(client_id, client_secret)).encode('utf-8')).decode('ascii')
    headers = {'Authorization' : 'Basic {}'.format(encoded)}
    body_param = {'grant_type':'client_credentials'}
    endpoint = 'https://accounts.spotify.com/api/token'
    r = requests.post(endpoint, data=body_param, headers=headers)
    access_token = '{} {}'.format(json.loads(r.text)['token_type'], json.loads(r.text)['access_token'])
    headers = {'Authorization':access_token}
    return headers

def lambda_handler(event, context):
    try:
        dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2',
                                  endpoint_url='http://dynamodb.ap-northeast-2.amazonaws.com',
                                  aws_access_key_id=os.environ['aws_id'],
                                  aws_secret_access_key=os.environ['aws_key'])
    except:
        print('DynamoDB접속오류!')
        sys.exit(1)
    headers = get_headers(client_id, client_secret)
    table = dynamodb.Table('moondynamo')
    #artist_id = '1dfeR4HaWDbWqFHLkxsg1d'
    artist_id = event['artist_id']
    params = {'market':'KR'}
    url = 'https://api.spotify.com/v1/artists/{a_id}/top-tracks'.format(a_id=artist_id)
    res = requests.get(url, params=params, headers=headers)
    raw = json.loads(res.text)
    for t in raw['tracks']:
        data = {'artist_id':artist_id, 'track_id':t.get('id')}
        data.update(t.get('album'))
        table.put_item(Item=data)
    return 'SUCESS'


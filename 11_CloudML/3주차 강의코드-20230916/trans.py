#!/usr/bin/env python
# coding: utf-8

# In[2]:


from urllib import parse
from urllib import request
import json


# In[3]:


def korean(txt):
    my_id = 'TClQh6ZS7wtfcy1mexDR'
    my_key = '_cVKquf3bH'
    endpoint = 'https://openapi.naver.com/v1/papago/n2mt'
    s = 'ko'
    t = 'en'
    txt = parse.quote(txt)
    param = 'source='+s+'&target='+t+'&text='+txt
    res = request.Request(endpoint)
    res.add_header('X-Naver-Client-Id',my_id)
    res.add_header('X-Naver-Client-Secret',my_key)
    response = request.urlopen(res, data=param.encode('utf-8'))
    results = json.load(response)
    return results.get('message').get('result').get('translatedText')


# In[9]:


if __name__=='__main__':
    raw_txt = input('번역할 문장을 한글로 입력하세요: ')
    print(korean(raw_txt))


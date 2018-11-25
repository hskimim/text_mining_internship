
# coding: utf-8

# In[1]:


from selenium import webdriver
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import time
from scrapy.http import TextResponse


# In[14]:


get_ipython().run_cell_magic('time', '', 'title_ls = []\nclicked_ls = []\ntime_ls = []\nlink_ls = []\n\nfor page in range(1,31+1):\n    if page % 100 ==0 : print(\'{} page is over.\'.format(page))\n    url = \'http://www.chaintalk.io/archive/talkbox/p{}\'.format(page)\n    req = requests.get(url)\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    for content in range(1,20+1):\n        try : \n            if content % 10 == 0 : time.sleep(1)\n            testing_ls = response.xpath(\'//*[@id="fboardlist"]/div/table/tbody/tr[{}]/td[2]/a/text()\'.format(content)).extract()[0]\n            testing_ls[1:].replace(\' \',\'\')\n            title_ls.append(testing_ls)\n            clicked_ls.append(response.xpath(\'//*[@id="fboardlist"]/div/table/tbody/tr[{}]/td[5]/text()\'.format(content)).extract()[0])\n            time_ls.append(response.xpath(\'//*[@id="fboardlist"]/div/table/tbody/tr[{}]/td[4]/text()\'.format(content)).extract()[0])\n            link_ls.append(response.xpath(\'//*[@id="fboardlist"]/div/table/tbody/tr[{}]/td[2]/a/@href\'.format(content)).extract()[0])\n        except : print(\'pass\')')


# In[15]:


df = pd.DataFrame()

df['title'] = title_ls
df['time'] = time_ls
df['click'] = clicked_ls
df['link'] = link_ls


# In[16]:


df.tail()


# In[27]:


get_ipython().run_cell_magic('time', '', 'content_ls = []\nfor idx in range(len(df)):\n    if idx % 10 ==0 : print(idx)\n    req = requests.get(df[\'link\'].values[idx])\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    process_ls = \',\'.join(response.xpath(\'//*[@id="bo_v_con"]/div/p/text()\').extract()).replace(\'\\xa0\',\'\').replace(\',\',\'\')\n    content_ls.append(process_ls)')


# In[28]:


df1 = pd.DataFrame()

df1['title'] = title_ls
df1['time'] = time_ls
df1['click'] = clicked_ls
df1['content'] = content_ls


# In[29]:


df1.tail()


# In[31]:


df1.to_csv('chaintalk_data.csv',index=False)


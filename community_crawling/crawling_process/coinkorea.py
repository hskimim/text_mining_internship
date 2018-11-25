
# coding: utf-8

# In[1]:


from selenium import webdriver
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import time
from scrapy.http import TextResponse


# In[29]:


get_ipython().run_cell_magic('time', '', 'title_ls = []\nclicked_ls = []\ntime_ls = []\nlink_ls = []\n\nfor page in range(1,500+1):\n    if page % 100 ==0 : print(\'{} page is over.\'.format(page))\n    url = \'https://coinkorea.info/index.php?mid=community&page={}\'.format(page)\n    req = requests.get(url)\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    for content in range(6,25+1):\n        try :\n            try : \n                title_ls.append(response.xpath(\'//*[@id="main"]/div/div/div/div[1]/div/div/table/tbody/tr[{}]/td[2]/a/span[1]/text()\'.format(content)).extract()[0])\n            except : \n                title_ls.append(response.xpath(\'//*[@id="main"]/div/div/div/div[1]/div/div/table/tbody/tr[{}]/td[2]/a/span[1]/span/text()\'.format(content)).extract()[0])\n            clicked_ls.append(response.xpath(\'//*[@id="main"]/div/div/div/div[1]/div/div/table/tbody/tr[{}]/td[6]/text()\'.format(content)).extract()[0])\n            time_ls.append(response.xpath(\'//*[@id="main"]/div/div/div/div[1]/div/div/table/tbody/tr[{}]/td[5]/text()\'.format(content)).extract()[0])\n            link_ls.append(response.xpath(\'///*[@id="main"]/div/div/div/div[1]/div/div/table/tbody/tr[{}]/td[2]/a/@href\'.format(content)).extract()[0])\n        except : print(page,content)')


# In[30]:


df = pd.DataFrame()

df['title'] = title_ls
df['time'] = time_ls
df['click'] = clicked_ls
df['link'] = link_ls


# In[32]:


df.tail()


# not yet worked from the below

# In[43]:


get_ipython().run_cell_magic('time', '', 'content_ls = []\nfor idx in range(len(df)):\n    if idx % 1000 ==0 : print(idx)\n    req = requests.get(\'https://coinkorea.info\'+df[\'link\'].values[idx])\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    process_ls = \',\'.join(response.xpath(\'//*[@id="main"]/div/div/div/div[1]/div/div[1]/div[1]/div[3]/div[1]/p/text()\').extract()).replace(\'\\xa0\',\'\').replace(\',\',\'\')\n    content_ls.append(process_ls)')


# In[44]:


df1 = pd.DataFrame()

df1['title'] = title_ls
df1['time'] = time_ls
df1['click'] = clicked_ls
df1['content'] = content_ls


# In[48]:


df1.tail()


# In[46]:


df1.to_csv('coinkorea_data.csv')


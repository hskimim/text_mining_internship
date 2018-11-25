
# coding: utf-8

# In[80]:


from selenium import webdriver
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import time
from scrapy.http import TextResponse


# In[84]:


get_ipython().run_cell_magic('time', '', 'title_ls = []\nclicked_ls = []\ntime_ls = []\nlink_ls = []\n\nfor page in range(1,147+1):\n    if page % 100 ==0 : print(\'{} page is over.\'.format(page))\n    url = \'http://hozaebox.com/bbs/board.php?bo_table=freeboard&page={}\'.format(1)\n    req = requests.get(url)\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    for content in range(1,15+1):\n        try : \n            testing_ls = response.xpath(\'//*[@id="fboardlist"]/table/tbody/tr[{}]/td[2]/a/text()\'.format(content)).extract()[1]\n            testing_ls = testing_ls[1:].replace(\' \',\'\')\n            title_ls.append(testing_ls)\n            clicked_ls.append(response.xpath(\'//*[@id="fboardlist"]/table/tbody/tr[{}]/td[5]/text()\'.format(content)).extract()[0])\n            time_ls.append(response.xpath(\'//*[@id="fboardlist"]/table/tbody/tr[{}]/td[4]/text()\'.format(content)).extract()[0])\n            link_ls.append(response.xpath(\'//*[@id="fboardlist"]/table/tbody/tr[{}]/td[2]/a/@href\'.format(content)).extract()[0])\n        except : print(\'pass\')')


# In[85]:


df3 = pd.DataFrame()

df3['title'] = title_ls
df3['time'] = time_ls
df3['click'] = clicked_ls
df3['link'] = link_ls
df3.tail()


# In[86]:


get_ipython().run_cell_magic('time', '', 'content_ls = []\nfor idx in range(len(df3)):\n    if idx % 10 ==0 : print(idx)\n    req = requests.get(df3[\'link\'].values[idx])\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    process_ls = \',\'.join(response.xpath(\'//*[@id="bo_v_con"]/p/text()\').extract()).replace(\',\',\'\')\n    content_ls.append(process_ls)')


# In[88]:


df4 = pd.DataFrame()

df4['title'] = title_ls
df4['time'] = time_ls
df4['click'] = clicked_ls
df4['content'] = content_ls


# In[90]:


df4.to_csv('coinpan_data.csv',index=False)


# In[91]:


df4


# ### 없는 데이터

# In[92]:


len([i for i in df4['content'].values if i == ''])


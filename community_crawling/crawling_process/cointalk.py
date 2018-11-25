
# coding: utf-8

# In[1]:


from selenium import webdriver
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import time
from scrapy.http import TextResponse


# In[2]:


get_ipython().run_cell_magic('time', '', 'title_ls = []\nclicked_ls = []\ntime_ls = []\nlink_ls = []\n\nfor page in range(1,182+1):\n    if page % 100 ==0 : print(\'{} page is over.\'.format(page))\n    url = \'http://cointalk.co.kr/bbs/board.php?bo_table=coinnews&page={}\'.format(page)\n    req = requests.get(url)\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    for content in range(1,35+1):\n        if content % 10 == 0 : time.sleep(1)\n        try : \n            title_ls.append(response.xpath(\'//*[@id="fboardlist"]/table/tbody/tr[{}]/td[2]/a/text()\'.format(content)).extract()[0])\n            click_process = \',\'.join(re.findall(\'\\d\',response.xpath(\'//*[@id="fboardlist"]/table/tbody/tr[{}]/td[4]/text()\'.format(content)).extract()[0])).replace(\',\',\'\')\n            clicked_ls.append(click_process)\n            time_ls.append(response.xpath(\'//*[@id="fboardlist"]/table/tbody/tr[{}]/td[7]/text()\'.format(content)).extract()[0])\n            link_ls.append(response.xpath(\'//*[@id="fboardlist"]/table/tbody/tr[{}]/td[2]/a/@href\'.format(content)).extract()[0])\n        except : print(\'pass\')')


# In[3]:


df = pd.DataFrame()

df['title'] = title_ls
df['time'] = time_ls
df['click'] = clicked_ls
df['link'] = link_ls


# In[4]:


df.tail()


# In[19]:


req = requests.get(df['link'].values[20])
response = TextResponse(req.url, body=req.text, encoding='utf-8')
len(','.join(response.xpath('//*[@id="st-view"]/section[1]/article/p/text()').extract()))


# In[ ]:


(/*[@id="st-view"]/section[1]/article/p[4])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'content_ls = []\nfor idx in range(len(df)):\n    if idx % 10 ==0 : print(idx)\n    req = requests.get(df[\'link\'].values[idx])\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    testing_ls = \',\'.join(response.xpath(\'//*[@id="st-view"]/section[1]/article/p/span/text()\').extract() + \\\n    response.xpath(\'//*[@id="st-view"]/section[1]/article/p/text()\').extract()).replace(\'\\xa0\',\'\').replace(\',\',\'\')\n    process_ls = testing_ls\n    content_ls.append(process_ls)')


# In[ ]:


df1 = pd.DataFrame()

df1['title'] = title_ls
df1['time'] = time_ls
df1['click'] = clicked_ls
df1['content'] = content_ls


# In[ ]:


df1.head()


# In[ ]:


df1.tail()


# In[ ]:


df1.to_csv('cointalk_data.csv',index=False)


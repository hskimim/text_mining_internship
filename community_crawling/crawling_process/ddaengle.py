
# coding: utf-8

# In[2]:


from selenium import webdriver
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import time
from scrapy.http import TextResponse


# # scrapy + selenium ( not optimal)

# ## scrapy 를 통해서 link , title , click , time 을 가져오는 코드

# In[ ]:


title_ls = []
clicked_ls = []
time_ls = []
link_ls = []

for page in range(1,100+1):
    if page % 10 == 0 : time.sleep(1)
    url = 'https://www.ddengle.com/index.php?mid=board_all&page={}'.format(page)
    driver.set_window_size(5000, 5000)
    driver.get(url)
    for i in range(1,20+1):
        try : 
            driver.execute_script('window.scrollTo(100,{})'.format(100))
            title_ls.append(                driver.find_element_by_css_selector(            '#bd_1598876_0 > div.bd_lst_wrp > table > tbody > tr:nth-child({}) > td.title > a'.format(i)).text)

            clicked_ls.append(            driver.find_element_by_css_selector(            '#bd_1598876_0 > div.bd_lst_wrp > table > tbody > tr:nth-child({}) > td:nth-child(4)'.format(i)).text)

            time_ls.append(                          driver.find_element_by_css_selector(            '#bd_1598876_0 > div.bd_lst_wrp > table > tbody > tr:nth-child({}) > td.time'.format(i)).text)

            link_ls.append(            driver.find_element_by_css_selector(            '#bd_1598876_0 > div.bd_lst_wrp > table > tbody > tr:nth-child({}) > td.title > a'.format(i)).get_attribute('href')        )
        except : print('pass')


# ### Link 만 가져오는 코드

# In[169]:


get_ipython().run_cell_magic('time', '', 'content_ls = []\nfor idx in range(len(df1)):\n    if idx % 500 ==0 : print(idx)\n    req = requests.get(df1[\'link\'].values[idx])\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    process_ls = \',\'.join(response.xpath(\'//*[@id="zema9_body"]/article/div/p/text()\').extract()).replace(\'\\xa0\',\'\').replace(\',\',\'\')\n    try : start_idx = re.search(\'-{30,}\',process_ls).start()\n    except : start_idx = len(process_ls) + 1\n    content_ls.append(process_ls[:start_idx])')


# In[173]:


df2 = pd.DataFrame()

df2['title'] = title_ls
df2['time'] = time_ls
df2['click'] = clicked_ls
df2['link'] = content_ls


# In[174]:


df1.to_csv('link.csv',index=False)
df2.to_csv('content_1999.csv',index=False)


# In[ ]:


df2


# # lastest vers of function ( Maybe it is optimal)

# In[1]:


from selenium import webdriver
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import time
from scrapy.http import TextResponse


# In[ ]:


get_ipython().run_cell_magic('time', '', 'title_ls = []\nclicked_ls = []\ntime_ls = []\nlink_ls = []\n\nfor page in range(1,5490+1):\n    if page % 100 ==0 : print(\'{} page is over.\'.format(page))\n    url = \'https://www.ddengle.com/index.php?mid=board_all&page={}\'.format(page)\n    req = requests.get(url)\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    for content in range(1,20+1):\n        try : \n            title_ls.append(response.xpath(\'//*[@id="bd_1598876_0"]/div[2]/table/tbody/tr[{}]/td[2]/a/text()\'.format(content)).extract()[0])\n            clicked_ls.append(response.xpath(\'//*[@id="bd_1598876_0"]/div[2]/table/tbody/tr[{}]/td[4]/text()\'.format(content)).extract()[0])\n            time_ls.append(response.xpath(\'//*[@id="bd_1598876_0"]/div[2]/table/tbody/tr[{}]/td[6]/text()\'.format(content)).extract()[0])\n            link_ls.append(response.xpath(\'//*[@id="bd_1598876_0"]/div[2]/table/tbody/tr[{}]/td[2]/a/@href\'.format(content)).extract()[0])\n        except : print(\'pass\')')


# In[ ]:


df3 = pd.DataFrame()

df3['title'] = title_ls
df3['time'] = time_ls
df3['click'] = clicked_ls
df3['link'] = link_ls
df3.tail()


# In[2]:


df3.to_csv('link.csv',index=False)


# In[3]:


df3 = pd.read_csv('../link.csv')
df3.tail()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'content_ls = []\nfor idx in range(len(df3)):\n    if idx % 1000 ==0 : print(idx) ; time.sleep(1)\n    req = requests.get(df3[\'link\'].values[idx])\n    response = TextResponse(req.url, body=req.text, encoding=\'utf-8\')\n    process_ls = \',\'.join(response.xpath(\'//*[@id="zema9_body"]/article/div/p/text()\').extract()).replace(\'\\xa0\',\'\').replace(\',\',\'\')\n    try : start_idx = re.search(\'-{30,}\',process_ls).start()\n    except : start_idx = len(process_ls) + 1\n    content_ls.append(process_ls[:start_idx])')


# In[ ]:


df4 = pd.DataFrame()

df4['title'] = df3['title'].values
df4['time'] = df3['time'].values
df4['click'] = df3['click'].values
df4['content'] = content_ls


# In[ ]:


df4.to_csv('ddaengle_data.csv',index=False)



# coding: utf-8

# In[1]:


import pandas as pd
import os
import re
import string
import time


# In[2]:


file =  [file for file in os.listdir() if file[-3:] == 'csv']


# In[3]:


file


# In[4]:


df = pd.DataFrame()
for csv_ in file : 
    process_df = pd.read_csv(csv_)
    df = pd.concat([df,process_df],axis=0)
df = df.iloc[:,1:]
df.reset_index(drop=True,inplace=True)
print(df.shape)


# In[5]:


df.tail()


# # 특수문자 제거해주는 프로세스(title,content)

# In[192]:


exception_ls = string.printable[62:]
exception_ls= exception_ls.replace(' ','')


# In[193]:


new_title_ls = []
for val in df['title'].values:
    try : 
        new_title_ls.append(','.join([i for i in val if i not in exception_ls]).replace(',',''))
    except : new_title_ls.append('NaN')
len(new_title_ls)


# In[194]:


df['title'] = new_title_ls


# In[195]:


new_content_ls = []
for val in df['content'].values:
    try : 
        new_content_ls.append(','.join([i for i in val if i not in exception_ls]).replace(',',''))
    except : new_content_ls.append('NaN')
len(new_content_ls)


# In[196]:


df['content'] = new_content_ls


# In[197]:


df


# # NaN 값을 제거해주는 프로세스

# In[198]:


not_null_index = [idx for idx,val in enumerate(df['content']) if val != 'NaN']
null_index = [idx for idx,val in enumerate(df['content']) if val == 'NaN']


# In[199]:


df.iloc[null_index]


# In[200]:


df = df.iloc[not_null_index]


# In[201]:


df.reset_index(drop=True,inplace=True)


# In[202]:


df


# ## 몇 시간 전은 11월 6일로 만들어 준다
# ## 몇 일전은 11월 6일 - a 로 만들어 준다.

# In[203]:


new_time_ls = []

for time in df['time'].values : 
    if re.findall('시간 전',time) != [] :
        new_time_ls.append('2018.11.6')
    else : new_time_ls.append(time)
len(new_time_ls)


# In[204]:


new_time_ls2 = []

for time in new_time_ls : 
    if re.findall('일 전',time) != [] :
        new_time_ls2.append((pd.to_datetime('20181106', format = '%Y%m%d') -          datetime.timedelta(days = int(time[0]))).strftime('%Y.%m.%d'))
    else : new_time_ls2.append(time)
len(new_time_ls2)


# In[207]:


df['time'] = new_time_ls2


# In[208]:


df


# In[209]:


df.to_csv('concat_data.csv',index=False)


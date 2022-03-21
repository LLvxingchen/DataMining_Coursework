#!/usr/bin/env python
# coding: utf-8

# ### 前言

# 本文档为针对Trending YouTube Video Statistics数据集探索性分析与预处理的报告。
# 报告整体分为四个模块介绍：  
# 
# 1. 数据集基本情况 
#   
# 2. 数据可视化和摘要
#   
# 3. 数据预处理  
# 
# 4. 探索性分析
#   
# 以下将逐步展开介绍各部分内容以及操作过程。
# 
# ---
# 
# 

# ### 1. 数据集基本情况

# In[108]:


#导入数据集
import pandas as pd

CA = pd.read_csv("./CAvideos.csv",encoding="latin-1") #加拿大
DE = pd.read_csv("./DEvideos.csv",encoding="latin-1") #英国
FR = pd.read_csv("./FRvideos.csv",encoding="latin-1") #法国
GB = pd.read_csv("./GBvideos.csv",encoding="latin-1") #德国
IN = pd.read_csv("./INvideos.csv",encoding="latin-1") #印度
JP = pd.read_csv("./JPvideos.csv",encoding="latin-1") #日本
KR = pd.read_csv("./KRvideos.csv",encoding="latin-1") #韩国
MX = pd.read_csv("./MXvideos.csv",encoding="latin-1") #墨西哥
RU = pd.read_csv("./RUvideos.csv",encoding="latin-1") #俄罗斯
US = pd.read_csv("./USvideos.csv",encoding="latin-1") #美国

data_all = pd.concat([CA,DE,FR,GB,IN,JP,KR,MX,RU,US], axis=0)
data_all.info()


# In[22]:


#观察数据集中的数据
data_all.head()


# 通过初步了解可知，原始数据来包含自加拿大、英、法、德、印度、日、韩、墨西哥、俄、美十个地区的热榜视频信息，共计375942条  
# 原始数据中属性种类共计16种：
# 
# |  | 属性名 | 中文名  | 数据类型 |
# |:---:|:---:|:---:|:---:|
# | 1 | video_id | 视频ID  | 字符串 |
# | 2 | trending_date | 上榜日期  | 字符串 |
# | 3 | title | 标题  | 字符串 |
# | 4 | channel_title | 频道名  | 字符串 |
# | 5 | category_id | 类别ID  | 整型 |
# | 6 | publish_time | 发行时间  | 字符串 |
# | 7 | tags | 标签  | 字符串 |
# | 8 | views | 观看数量  | 整型 |
# | 9 | likes | 点赞数量 | 整型 |
# | 10 | dislikes | 点踩数量 | 整型 |
# | 11 | comment_count | 评论数量 | 整型 |
# | 12 | thumbnail_link | 缩略图链接  | 字符串 |
# | 13 | comments_disabled | 允许评论  | 布尔型 |
# | 14 | ratings_disabled | 允许打分  | 布尔型 |
# | 15 | video_error_or_removed | 视频出错  | 布尔型 |
# | 16 | description | 视频简介 | 字符串 |
# 
# 
# 

# ### 2. 数据摘要和可视化

# #### 2.1数据摘要
# 

# 首先观察数据集中的数据，区分有分析价值的标称属性和数值属性。  
# **根据数据类型，同时考虑属性的实际含义可知：**
# 
# 
# 

# |  |  |  |  |  |  |  |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | **标称属性** | channel_title  | category_id  | comments_disabled  | ratings_disabled  | video_error_or_removed |  |
# | **数值属性** | publish_time  | comment_count | views  | likes  | dislikes  |
# 

# In[31]:


#找出各标称属性的可能取值频数：
for _ in (['channel_title','category_id','comments_disabled','ratings_disabled','video_error_or_removed']):
    print(_,'属性取值频数：')
    print(data_all[_].value_counts())
    print()


# In[119]:


#各数值属性的5数概括
data_all['publish_time']=data_all['publish_time'].astype('datetime64[ns]')
pd.set_option('display.float_format',lambda x : '%.2f' % x)
for _ in (['publish_time','views','likes','dislikes','comment_count']):
    print(_,'属性5数概括：')
    print(data_all[_].describe(datetime_is_numeric=True))
    print()


# In[104]:


#检查各数值属性的缺失率
for _ in (['publish_time','views','likes','dislikes','comment_count']):
    print(_,'属性缺失率：')
    print(data_all[_].isnull().sum()/data_all[_].shape[0])
    print()


# #### 2.2数据可视化
# 首先使用直方图观察标称数据的分布情况：

# In[109]:


import matplotlib.pyplot as plt
import numpy as np
data_all1 = data_all.copy()

#绘制category_id直方图
data_all['category_id'].hist(bins=np.arange(0.5,43.5),alpha=0.6,figsize=(20,5))  

my_x_ticks = np.arange(0, 45, 1)
plt.xticks(my_x_ticks)
plt.grid(axis='x')
plt.title('the hist of category_id')
plt.show()

fig = plt.figure(figsize=(20,5))
#绘制comments_disabled直方图
ax1 = fig.add_subplot(1,3,1)
data_all1['comments_disabled'] = data_all1['comments_disabled'].astype(str)
data_all1['comments_disabled'].hist(bins=np.arange(-0.5,10.5),alpha=0.6) 
plt.grid(axis='x')
plt.title('the hist of comments_disabled')


#绘制ratings_disabled直方图
ax2 = fig.add_subplot(1,3,2)
data_all1['ratings_disabled'] = data_all1['ratings_disabled'].astype(str)
data_all1['ratings_disabled'].hist(bins=np.arange(-0.5,10.5),alpha=0.6) 
plt.grid(axis='x')
plt.title('the hist of ratings_disabled')


#绘制video_error_or_removed直方图
ax3 = fig.add_subplot(1,3,3)
data_all1['video_error_or_removed'] = data_all1['video_error_or_removed'].astype(str)
data_all1['video_error_or_removed'].hist(bins=np.arange(-0.5,10.5),alpha=0.6) 
plt.grid(axis='x')
plt.title('the hist of video_error_or_removed')
plt.show()


# 通过可视化方法观察到，对于标称属性而言：
# - **排名前五的视频类别是24、22、10、25、23，分别对应：  
# Entertainment（娱乐）、People & Blogs（人物博客）、Music（音乐）、News & Politics（新闻时政）、Comedy（喜剧）**
# - **上榜视频绝大部分允许评价、打分**
# - **上榜视频绝大部分仍可以正常观看**

# 使用折线图和箱图观察数值数据的分布和离群点情况：

# In[208]:


data_all2 = data_all.copy()
fig = plt.figure(figsize=(20,5))
#views的数量折线图
ax1 = fig.add_subplot(1,2,1)
data_all2['views'].plot(alpha=0.6)
plt.grid(axis='y')
plt.title('views Count')
#views的箱图
ax2 = fig.add_subplot(1,2,2)
data_all2['views'].plot.box()
plt.title('boxplot of views')
plt.show()

fig = plt.figure(figsize=(20,5))
#likes的数量折线图
ax1 = fig.add_subplot(1,2,1)
data_all2['likes'].plot(alpha=0.6)
plt.grid(axis='y')
plt.title('likes Count')
#likes的箱图
ax2 = fig.add_subplot(1,2,2)
data_all2['likes'].plot.box()
plt.title('boxplot of likes')
plt.show()

fig = plt.figure(figsize=(20,5))
#dislikes的数量折线图
ax1 = fig.add_subplot(1,2,1)
data_all2['dislikes'].plot(alpha=0.6)
plt.grid(axis='y')
plt.title('dislikes Count')
#dislikes的箱图
ax2 = fig.add_subplot(1,2,2)
data_all2['dislikes'].plot.box()
plt.title('boxplot of dislikes')
plt.show()

fig = plt.figure(figsize=(20,5))
#comment_count的数量折线图
ax1 = fig.add_subplot(1,2,1)
data_all2['comment_count'].plot(alpha=0.6)
plt.grid(axis='y')
plt.title('comment_count Count')
#comment_count的箱图
ax2 = fig.add_subplot(1,2,2)
data_all2['comment_count'].plot.box()
plt.title('boxplot of comment_count')
plt.show()


# 通过可视化方法观察到，对于数值属性而言：
# - **点赞量、喜欢、不喜欢、评论的数目箱图几乎都看不到“箱子”**
# - **通过下图分析原因，可能是由于它们的极大值都比第3个分位数大到100-100倍，即大部分的数据量级远小于极大值**

# In[237]:


data_all3=data_all.copy()

fig = plt.figure(figsize=(20,10))
#views的describe折线图
fig.add_subplot(2,2,1)
data_all3['views'].describe().plot(title='views')

#likes的describe折线图
fig.add_subplot(2,2,2)
data_all3['likes'].describe().plot(title='likes')

#dislikes的describe折线图
fig.add_subplot(2,2,3)
ax1 = data_all3['dislikes'].describe().plot(title='dislikes')

#comment_count的describe折线图
fig.add_subplot(2,2,4)
ax1 = data_all3['comment_count'].describe().plot(title='comment_count')


# 因此，对于该数据集内的数据属性值可以更多的从5数描述中获取有用概况信息：
# - 观看量均值为1326568次，最小值117，最大值424538912，中位数177370
# - 点赞量均值为37884次，最小值0，最大值5613827，中位数3446
# - 点踩量均值为2126次，最小值0，最大值5613827，中位数179
# - 评论量均值为4253次，最小值0，最大值1626501，中位数511
# 
# 

# ### 3.数据预处理

# In[247]:


#检查所有属性的缺失情况
data_all.isnull().sum()/data_all.shape[0]


# 绝大部分属性无缺失值，具有5%缺失的description属性也并不是需要分析的属性，因此无需处理。

# ### 4. 探索性分析

# ##### 0.采用剔数据集展开探索性分析：
# 为便于分析，我们首先将category_id属性值转为对应的分类名称。

# In[187]:


#将category_id转为实际类别
import json

with open("./US_category_id.json",'r') as js:
    catagory_id = json.load(js)

id_map={}
for _ in catagory_id['items']:
    id_map[int(_['id'])] = _['snippet']['title']
for _ in [CA,DE,FR,GB,IN,JP,KR,MX,RU,US]:
    _['category_id']=_['category_id'].map(id_map)
    
data_all = pd.concat([CA,DE,FR,GB,IN,JP,KR,MX,RU,US], axis=0)
data_all['category_id'].describe()


# ##### 1.观察上榜视频的发布时间统计情况：
# - 早上6点-下午16点是视频发布量增长的主流时间段，早上6点发布量最低，下午16点视频发布量达到最高
# - 下午16点-凌晨24点视频发布量逐渐减少

# In[184]:


data_all5=data_all.copy()
data_all5['publish_hour']=data_all['publish_time'].dt.hour
group=data_all5.groupby('publish_hour')
group.size().plot(kind='line',figsize=(20,5),linestyle='-.',linewidth=3,alpha=0.8)
plt.xticks(range(0,24))
plt.title('the publish_hour of videos')
plt.show()


# ##### 2.观察各类视频的上榜数量：  
# 排名前五为Entertainment（娱乐）、People & Blogs（人物博客）、Music（音乐）、News & Politics（新闻时政）、Comedy（喜剧）  
# 排名后五为Nonprofits & Activism（非营利组织与行动主义）、Travel & Events（旅行活动）、Movies（电影）、Shows（演出）、Trailers（预告片）

# In[384]:


#按视频分类排序可视化各类视频数量
import seaborn as sns
data_all['category_id'].value_counts().sort_values(ascending=False).plot.barh(alpha=0.6,figsize=(20,8))


# ##### 3.观察各类视频的平均播放量、点赞、点踩和评论量：  
# - **Music类节目的平均观看、点赞、评论数量排名第1且远超其他类别，被踩量排名第2**
#     - 推断原因是观看量的巨大基数使得其各类评价量都要更高一些  
#     
# - **nonprofits & activism（非营利组织与行动主义）平均观看量排名第12**
#     - 这类视频的关注量不高，但它的平均点赞量排名第3，评论量第2，点踩量第1
#     - 可能是这类视频的受众两极分化比较严重，喜欢的很喜欢，讨厌的会很讨厌  
#     
# - **Trailers（预告片）平均各项排名倒数第1**
#     - 上榜数量本就最少，又因为预告片本身简短预告的属性导致

# In[107]:


#按视频分类排序可视化各类视频的平均播放量、点赞、点踩和评论量
groupby_id=data_all.groupby('category_id')
views_ave=groupby_id.views.mean().sort_values(ascending=False)
likes_ave=groupby_id.likes.mean().sort_values(ascending=False)
dislikes_ave=groupby_id.dislikes.mean().sort_values(ascending=False)
comment_count_ave=groupby_id.comment_count.mean().sort_values(ascending=False)

fig = plt.figure(figsize=(20,12))
fig.add_subplot(2,2,1)
views_ave.plot(kind='barh',alpha=0.6)
plt.title('the average views of different type videos')
fig.add_subplot(2,2,2)
likes_ave.plot(kind='barh',alpha=0.6)
plt.title('the average likes of different type videos')
fig.add_subplot(2,2,3)
dislikes_ave.plot(kind='barh',alpha=0.6)
plt.title('the average dislikes of different type videos')
fig.add_subplot(2,2,4)
comment_count_ave.plot(kind='barh',alpha=0.6)
plt.title('the average comment_count of different type videos')
plt.show()


# ##### 4.观察各类视频播放量、点赞、点踩和评论量的关系：  
# - **观看量与点赞量的相关程度更高，表明视频的大部分用户表态正向居多**
# - **点赞量、点踩量与评论量的相关程度也比较高，表明有点赞或点踩的态度表达时评论也会更多**
# 

# In[344]:


#点赞量和评论量之间的比例关系
import numpy as np
import decimal
from pandas.plotting import scatter_matrix
views_likes=(data_all['likes']+data_all['dislikes'])/data_all['views']
se=data_all[['views','likes','dislikes','comment_count']]

corr=se.corr()
sns.heatmap(corr,annot=True,cmap='Blues', cbar=True)
plt.show()


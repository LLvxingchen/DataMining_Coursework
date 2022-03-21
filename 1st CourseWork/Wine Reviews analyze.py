#!/usr/bin/env python
# coding: utf-8

# ### 前言

# 本文档为针对Wine Reviews数据集探索性分析与预处理的报告。
# 报告整体分为三个模块介绍：  
# 
# 1. 数据集基本情况 
#   
# 2. 数据可视化和摘要
#   
# 3. 数据预处理  
# 
#   
# 以下将逐步展开介绍各部分内容以及操作过程。
# 
# ---
# 
# 

# ### 1. 数据集基本情况

# In[10]:


#导入数据集
import pandas as pd

data1 = pd.read_csv("./winemag-data_first150k.csv",encoding="utf-8")
data2 = pd.read_csv("./winemag-data-130k-v2.csv",encoding="utf-8")
# label = pd.read_csv("./winemag-data-130k-v2.json",encoding="utf-8")

print(data1.info())
print(data2.info())


# In[3]:


#观察子数据集1中的数据
data1.head()


# In[4]:


#观察子数据集2中的数据
data2.head()


# 通过初步了解可知，原始的红酒评论数据主要来自两个子数据集，分别含有150930和129971条数据  
# 数据集1中属性种类共计11种：
# 
# |    | 属性名          | 中文名   | 数据类型 |
# |----|--------------|-------|------|
# | 1  | Unnamed: 0   | 序号    | 整型   |
# | 2  | country      | 国家    | 字符串  |
# | 3  | description  | 口味描述  | 字符串  |
# | 4  | designation  | 葡萄酒名称 | 字符串  |
# | 5  | points       | 打分    | 整型   |
# | 6  | price        | 价格    | 浮点数  |
# | 7  | province     | 省份    | 字符串  |
# | 8  | region_1     | 产地1   | 字符串  |
# | 9  | region_2     | 产地2   | 字符串  |
# | 10 | variety      | 品种    | 字符串  |
# | 11 | winery       | 酒厂    | 字符串  |
# 
# 数据集1中属性种类共计14种：
# 
# |    | 属性名                    | 中文名   | 数据类型 |
# |----|------------------------|-------|------|
# | 1  | Unnamed: 0             | 序号    | 整型   |
# | 2  | country                | 国家    | 字符串  |
# | 3  | description            | 口味描述  | 字符串  |
# | 4  | designation            | 葡萄酒名称 | 字符串  |
# | 5  | points                 | 打分    | 整型   |
# | 6  | price                  | 价格    | 浮点数  |
# | 7  | province               | 省份    | 字符串  |
# | 8  | region_1               | 产地1   | 字符串  |
# | 9  | region_2               | 产地2   | 字符串  |
# | 10 | taster_name            | 品鉴师姓名 | 字符串  |
# | 11 | taster_twitter_handle  | 品鉴师推特 | 字符串  |
# | 12 | title                  | 标签    | 字符串  |
# | 13 | variety                | 品种    | 字符串  |
# | 14 | winery                 | 酒厂    | 字符串  |
# 
# 

# ### 2. 数据摘要和可视化

# ### 2.1数据摘要
# 

# 首先观察数据集中的数据，区分有分析价值的标称属性和数值属性。  
# **根据数据类型，同时考虑属性的实际含义可知：**
# 
# 
# 

# | 数值数据 | points  | price       |          |          |          |             |                       |       |         |        |
# |------|---------|-------------|----------|----------|----------|-------------|-----------------------|-------|---------|--------|
# | 标称数据 | country | description|designation | province | region_1 | region_2 | taster_name | taster_twitter_handle | title | variety | winery |
# 
# 
# 

# **将2个数据子集合一，将公共属性和数据子集2独有的属性分别分析**

# In[43]:


#2个数据子集合一,删除子集2独有属性和无用的序号属性
data=pd.concat([data1,data2], axis=0)
data=data.drop(['Unnamed: 0','taster_name','taster_twitter_handle','title'],1)
data.columns


# In[44]:


#找出各公共标称属性的可能取值频数：
data_Nominal=data[['country','description', 'designation', 'province','region_1', 'region_2', 'variety', 'winery']]
for _ in data_Nominal.columns:
    print(_,'属性取值频数：')
    print(data_Nominal[_].value_counts())
    print()


# In[14]:


#找出子集2标称属性的可能取值频数：
for _ in ['taster_name', 'taster_twitter_handle', 'title']:
    print(_,'属性取值频数：')
    print(data2[_].value_counts())
    print()


# In[153]:


#各数值属性的5数概括
data_Numerical=data[['points','price']]
pd.set_option('display.float_format',lambda x : '%.2f' % x)
for _ in data_Numerical.columns:
    print(_,'属性5数概括：')
    print(data_Numerical[_].describe())
    print()


# In[15]:


#检查各公共标称属性的缺失率
for _ in data_Nominal.columns:
    print(_,'属性缺失率：')
    print(data_Nominal[_].isnull().sum()/data_Nominal[_].shape[0])
    print()


# In[25]:


#检查子集2标称属性的缺失率
for _ in ['taster_name', 'taster_twitter_handle', 'title']:
    print(_,'属性缺失率：')
    print(data2[_].isnull().sum()/data2[_].shape[0])
    print()


# In[23]:


#检查各数值属性的缺失率
for _ in data_Numerical.columns:
    print(_,'属性缺失率：')
    print(data_Numerical[_].isnull().sum()/data_Numerical[_].shape[0])
    print()


# ####  2.2数据可视化
# 

# ##### (1)使用图表观察标称数据的分布情况：
# - country(国家)分布前5：US(美国)、France（法国）、Italy（意大利）、Spain（西班牙）、Portugal（葡萄牙）
# - designation(名称)分布前5：Reserve、Reserva、Estate、Barrel sample、Riserva
# - 省份(名称)分布前5：California、Washington、Tuscany、Bordeaux、Oregon、
# - region_1(区域1)分布前5：Napa Valley、Columbia Valley (WA)、Russian River Valley、California、Mendoza
# - region_2(名称)分布前5：Central Coast、Sonoma、Columbia Valley、Napa、Willamette Valley、
# - variety(品种)分布前5：Pinot Noir、Chardonnay、Cabernet Sauvignon、Red Blend、Bordeaux-style Red Blend
# - winery(酒厂)分布前5：Williams Selyem、Testarossa、DFJ Vinhos、Chateau Ste. Michelle、Wines & Winemakers
# 

# In[111]:


import matplotlib.pyplot as plt
fig=plt.figure(figsize=(20,20))
fig.add_subplot(2,2,1)
data.country.value_counts().plot(kind='barh',title='the distrbution of country frequency')
fig.add_subplot(2,2,2)
data.designation.value_counts().head(30).plot(kind='barh',title='the distrbution of designation frequency')

fig=plt.figure(figsize=(20,20))
fig.add_subplot(2,2,1)
data.province.value_counts().head(30).plot(kind='barh',title='the distrbution of province frequency')
fig.add_subplot(2,2,2)
data.region_1.value_counts().head(30).plot(kind='barh',title='the distrbution of region_1 frequency')

fig=plt.figure(figsize=(20,20))
fig.add_subplot(2,2,1)
data.region_2.value_counts().head(30).plot(kind='barh',title='the distrbution of region_2 frequency')
fig.add_subplot(2,2,2)
data.variety.value_counts().head(30).plot(kind='barh',title='the distrbution of variety frequency')
plt.show()

data.winery.value_counts().head(30).plot(kind='barh',figsize=(20,10),title='the distrbution of winery frequency')
plt.show()


# - taster_name(品鉴师)分布前5：Roger Voss、Michael Schachner、Kerin O’Keefe、Virginie Boone、Paul Gregutt
# - taster_twitter_handle(品鉴师twittere)分布前5：@vossroger、@wineschach、@kerinokeefe、@vboone、@paulgwine 
# - title(标签)分布前5：Gloria Ferrer NV Sonoma Brut Sparkling (Sonoma County)、Korbel NV Brut Sparkling (California)、Segura Viudas NV Extra Dry Sparkling (Cava)、Gloria Ferrer NV Blanc de Noirs Sparkling (Carneros)、Segura Viudas NV Aria Estate Extra Dry Sparkling (Cava)

# In[114]:


data2.taster_name.value_counts().head(30).plot(kind='barh',figsize=(20,10),title='the distrbution of taster_name frequency')
plt.show()
data2.taster_twitter_handle.value_counts().head(30).plot(kind='barh',figsize=(20,10),title='the distrbution of taster_twitter_handle frequency')
plt.show()
data2.title.value_counts().head(50).plot(kind='barh',figsize=(20,10),title='the distrbution of title frequency')
plt.show()


# ##### (2)使用直方图观察数值数据的分布情况：
# - 打分均值88，分数分布情况接近正太分布
# - 价格情况绝大部分集中在0-300元的区间内

# In[34]:


#直方图观察数值数据的分布
import matplotlib.pyplot as plt
import numpy as np
data_Numerical.hist(figsize=(20,5),alpha=0.7)
plt.show()


# ##### (3)使用箱图观察数值数据的分布和离群点情况：
# - 打分情况的离群点较少，主要是少数的分数超过96左右
# - 价格情况的离群点较多
#     - 结合5数概括分析其均值为34美元，第3四分位数为40美元
#     - 极大值达到3300美元，可能是少数具有收藏价值的红酒

# In[48]:


#箱图观察数值数据的离群点
fig=plt.figure(figsize=(20,5))
fig.add_subplot(1,2,1)
data_Numerical.points.plot(kind='box')
fig.add_subplot(1,2,2)
data_Numerical.price.plot(kind='box')
plt.show()


# 因此，对于该数据集内的数据属性值可以更多的从5数描述中获取有用概况信息：
# - 观看量均值为1326568次，最小值117，最大值424538912，中位数177370
# - 点赞量均值为37884次，最小值0，最大值5613827，中位数3446
# - 点踩量均值为2126次，最小值0，最大值5613827，中位数179
# - 评论量均值为4253次，最小值0，最大值1626501，中位数511
# 
# 

# ### 3.数据预处理

# ##### (0)首先检查要处理的缺失属性和缺失比例
# | 缺失属性 | price | region_1 | region_2 | designation | taster_name | taster_twitter_handle |
# |------|-------|----------|----------|-------------|-------------|-----------------------|
# | 缺失比例 | 8%    | 16%      | 60%      | 30%         | 20%         | 24%                   |
# 
# 

# In[156]:


#检查所有公有属性和子集2独有属性的缺失情况
for _ in ['taster_name', 'taster_twitter_handle', 'title']:
    print(_,end='\t')
    print('{:0.2}'.format(data2[_].isnull().sum()/data2[_].shape[0]))
print(data.isnull().sum()/data.shape[0])


# ##### (0)首先检查要处理的缺失属性和缺失比例
# 
# | 缺失属性                  | 缺失比例 |  原因分析             |
# |-----------------------|------|-------------------|
# | price                 | 8%   | 忘记写价格（无意）             |
# | region_1              | 16%  | 忘记写省份内的具体产地（无意）       |
# | region_2              | 60%  | 产地1已经足够具体，无需产地2补充（有意） |
# | designation           | 30%  | 忘记标注酒名（无意）            |
# | taster_name           | 20%  | 品鉴师没有留名（无意or有意）           |
# | taster_twitter_handle | 24%  | 品鉴师没有留twitter（无意or有意）     |
# 

# ##### (1)方法一：直接删除缺失部分

# In[37]:


#删除缺失属性的数据后打印缺失率
data_1=data.dropna()
data_2=data2.dropna()
for _ in ['taster_name', 'taster_twitter_handle', 'title']:
    print(_,end='\t')
    print('{:0.2}'.format(data_2[_].isnull().sum()/data_2[_].shape[0]))
print(data_1.isnull().sum()/data_1.shape[0])


# ##### 0)直接删除法后，数据集数量大幅减少
# - 数据子集1数量由150930变为733250
# - 数据子集2数量由313418变为129971

# In[112]:


print(data_1.size)
print(data_2.size)


# ##### 1)可视化展示直接删除缺失后，标称数据分布变化情况：
# - country(国家)分布前5：US(美国)、France（法国）、Italy（意大利）、Spain（西班牙）、Portugal（葡萄牙）
#     - **变为只剩下US(美国)**
# - designation(名称)分布前5：Reserve、Reserva、Estate、Barrel sample、Riserva
#     - **变为Reserve、Estate、Estate Grown、Dry、Estate Bottled**
# - 省份(名称)分布前5：California、Washington、Tuscany、Bordeaux、Oregon
#     - **变为California、Washington、Oregon、New York（只剩下4个省份）**
# - region_1(区域1)分布前5：Napa Valley、Columbia Valley (WA)、Russian River Valley、California、Mendoza
#     - **变为Napa Valley、Columbia Valley (WA)、Russian River Valley、Paso Robles、California**
# - region_2(名称)分布前5：Central Coast、Sonoma、Columbia Valley、Napa、Willamette Valley、
#     - **无改变**
# - variety(品种)分布前5：Pinot Noir、Chardonnay、Cabernet Sauvignon、Red Blend、Bordeaux-style Red Blend
#     - **变为Pinot Noir、Cabernet Sauvignon、Chardonnay、Red Blend、Syrah**
# - winery(酒厂)分布前5：Williams Selyem、Testarossa、DFJ Vinhos、Chateau Ste. Michelle、Wines & Winemakers
#     - **变为Williams Selyem、Testarossa、Columbia Crest、Kendall-Jackson、Chateau Ste. Michelle**
# - taster_name(品鉴师)分布前5：Roger Voss、Michael Schachner、Kerin O’Keefe、Virginie Boone、Paul Gregutt
#     - **变为Virginie Boone、Paul Gregutt、Matt Kettmann、Sean P. Sullivan、Jim Gordon**
# - taster_twitter_handle(品鉴师twittere)分布前5：@vossroger、@wineschach、@kerinokeefe、@vboone、@paulgwine 
#     - **变为@vboone、@paulgwine 、@mattkettmann、@wawinereport、@gordone_cellars**
# - title(标签)分布前5：Gloria Ferrer NV Sonoma Brut Sparkling (Sonoma County)、Korbel NV Brut Sparkling (California)、Segura Viudas NV Extra Dry Sparkling (Cava)、Gloria Ferrer NV Blanc de Noirs Sparkling (Carneros)、Segura Viudas NV Aria Estate Extra Dry Sparkling (Cava)
#     - **变为Gloria Ferrer NV Sonoma Brut Sparkling (Sonoma County)、Breathless NV Brut Sparkling (North Coast)、Callaway 2013 Winemaker's Reserve Roussanne (Temecula Valley)、Korbel NV Brut Sparkling (California)、Woodward Canyon 2013 Estate Cabernet Sauvignon (Walla Walla Valley (WA))**
# 

# In[51]:


#可视化对比缺失
fig=plt.figure(figsize=(20,20))
fig.add_subplot(2,2,1)
data_1.country.value_counts().plot(kind='barh',title='the distrbution of country frequency')
fig.add_subplot(2,2,2)
data_1.designation.value_counts().head(30).plot(kind='barh',title='the distrbution of designation frequency')

fig=plt.figure(figsize=(20,20))
fig.add_subplot(2,2,1)
data_1.province.value_counts().plot(kind='barh',title='the distrbution of province frequency')
fig.add_subplot(2,2,2)
data_1.region_1.value_counts().head(30).plot(kind='barh',title='the distrbution of region_1 frequency')

fig=plt.figure(figsize=(20,20))
fig.add_subplot(2,2,1)
data_1.region_2.value_counts().head(30).plot(kind='barh',title='the distrbution of region_2 frequency')
fig.add_subplot(2,2,2)
data_1.variety.value_counts().head(30).plot(kind='barh',title='the distrbution of variety frequency')
plt.show()

data_1.winery.value_counts().head(30).plot(kind='barh',figsize=(10,5),title='the distrbution of winery frequency')
plt.show()


# In[60]:


data_2.taster_name.value_counts().head(30).plot(kind='barh',figsize=(20,10),title='the distrbution of taster_name frequency')
plt.show()
data_2.taster_twitter_handle.value_counts().head(30).plot(kind='barh',figsize=(20,10),title='the distrbution of taster_twitter_handle frequency')
plt.show()
data_2.title.value_counts().head(50).plot(kind='barh',figsize=(20,10),title='the distrbution of title frequency')
plt.show()


# ##### 2)使用删除法前后，数值属性分布情况变化：
# - 得分情况的均值和四分位数都有一定提高，极高分数情况减少
# - 价格情况均值和前三个四分位数少量提高，极大值数量减少

# In[124]:


#原points属性删除前后的直方图变化
import matplotlib.pyplot as plt
import numpy as np
data_Numerical_1=data_1[['points','price']]
fig=plt.figure(figsize=(20,10))
fig.add_subplot(1,2,1)
data_Numerical.points.hist(figsize=(20,5),alpha=0.7)
fig.add_subplot(1,2,2)
data_Numerical_1.points.hist(figsize=(20,5),alpha=0.7)
plt.show()


# In[125]:


#原points属性删除前后的箱图变化
fig=plt.figure(figsize=(20,5))
fig.add_subplot(1,2,1)
data_Numerical.points.plot(kind='box')
fig.add_subplot(1,2,2)
data_Numerical_1.points.plot(kind='box')
plt.show()


# In[126]:


#原Price属性删除前后的直方图变化
fig=plt.figure(figsize=(20,10))
fig.add_subplot(1,2,1)
data_Numerical.price.hist(figsize=(20,5),alpha=0.7)
fig.add_subplot(1,2,2)
data_Numerical_1.price.hist(figsize=(20,5),alpha=0.7)
plt.show()


# In[127]:


#原Price属性删除前后的箱图变化
fig=plt.figure(figsize=(20,5))
fig.add_subplot(1,2,1)
data_Numerical.price.plot(kind='box')
fig.add_subplot(1,2,2)
data_Numerical_1.price.plot(kind='box')
plt.show()


# In[136]:


#price删除前后五数概括情况
print(data_Numerical.price.describe())
print(data_Numerical_1.price.describe())


# ##### (2)方法二：最高频值填充法

# In[62]:


#使用众数填补缺失属性值
na=['price','region_1','region_2','designation']
na2=['taster_name','taster_twitter_handle']
data_3=data.copy()
data_4=data2.copy()
for _ in na:
    data_3[_]=data_3[_].fillna(data_3[_].mode().values[0])
for _ in na2:
    data_4[_]=data_4[_].fillna(data_4[_].mode().values[0])
for _ in ['taster_name', 'taster_twitter_handle', 'title']:
    print(_,end='\t')
    print('{:0.2}'.format(data_4[_].isnull().sum()/data_4[_].shape[0]))
print(data_3.isnull().sum()/data_3.shape[0])


# ##### 1)可视化展示最大值填充后，发生变化的公共标称数据分布情况：
# - designation(名称)分布前5：Reserve、Reserva、Estate、Barrel sample、Riserva
#     - **Reserve增加了20多倍**
# - region_1(区域1)分布前5：Napa Valley、Columbia Valley (WA)、Russian River Valley、California、Mendoza
#     - **Napa Valley增加5倍左右**
# - region_2(名称)分布前5：Central Coast、Sonoma、Columbia Valley、Napa、Willamette Valley、
#     - **Central Coast增加10倍左右**
# - taster_name(品鉴师)分布前5：Roger Voss、Michael Schachner、Kerin O’Keefe、Virginie Boone、Paul Gregutt
#     - **Roger Voss增加2倍左右**
# - taster_twitter_handle(品鉴师twittere)分布前5：@vossroger、@wineschach、@kerinokeefe、@vboone、@paulgwine 
#     - **@vossroger增加2倍左右**

# In[120]:


#可视化对比缺失
fig=plt.figure(figsize=(20,10))
fig.add_subplot(1,2,1)
data.designation.value_counts().head(30).plot(kind='barh',title='the distrbution of designation frequency')
fig.add_subplot(1,2,2)
data_3.designation.value_counts().head(30).plot(kind='barh',title='the distrbution of designation frequency')
plt.show()

fig=plt.figure(figsize=(20,10))
fig.add_subplot(1,2,1)
data.region_1.value_counts().head(30).plot(kind='barh',title='the distrbution of region_1 frequency')
fig.add_subplot(1,2,2)
data_3.region_1.value_counts().head(30).plot(kind='barh',title='the distrbution of region_1 frequency')
plt.show()

fig=plt.figure(figsize=(20,10))
fig.add_subplot(1,2,1)
data.region_2.value_counts().head(30).plot(kind='barh',title='the distrbution of region_2 frequency')
fig.add_subplot(1,2,2)
data_3.region_2.value_counts().head(30).plot(kind='barh',title='the distrbution of region_2 frequency')
plt.show()


# In[122]:


fig=plt.figure(figsize=(20,10))
fig.add_subplot(1,2,1)
data2.taster_name.value_counts().head(30).plot(kind='barh',title='the distrbution of taster_name frequency')
fig.add_subplot(1,2,2)
data_4.taster_name.value_counts().head(30).plot(kind='barh',title='the distrbution of taster_name frequency')
plt.show()

fig=plt.figure(figsize=(20,10))
fig.add_subplot(1,2,1)
data2.taster_twitter_handle.value_counts().head(30).plot(kind='barh',figsize=(20,10),title='the distrbution of taster_twitter_handle frequency')
fig.add_subplot(1,2,2)
data_4.taster_twitter_handle.value_counts().head(30).plot(kind='barh',figsize=(20,10),title='the distrbution of taster_twitter_handle frequency')
plt.show()


# ##### 2)使用众数填充法前后，数值属性price的极大值提高

# In[130]:


#原Price属性填充前后的直方图变化
data_Numerical_2=data_3[['points','price']]
fig=plt.figure(figsize=(20,10))
fig.add_subplot(1,2,1)
data_Numerical.price.hist(figsize=(20,5),alpha=0.7)
fig.add_subplot(1,2,2)
data_Numerical_2.price.hist(figsize=(20,5),alpha=0.7)
plt.show()


# In[131]:


#原Price属性填充前后的箱图变化
fig=plt.figure(figsize=(20,5))
fig.add_subplot(1,2,1)
data_Numerical.price.plot(kind='box')
fig.add_subplot(1,2,2)
data_Numerical_2.price.plot(kind='box')
plt.show()


# ##### (3)方法三：属性相关关系补充法

# 对缺失的数值属性price来讲，由于该数据集中数据属性只有points和price，经计算得知两者相关性并不强（0.44），因此不适用属性相关关系补充法。

# In[33]:


#计算属性之间的相关性
import seaborn as sns
corr=data.corr()
sns.heatmap(corr,annot=True,cmap='Blues', cbar=True)


# ##### (4)方法四：数据对象相似性补充法

# 检查数值属性points和price之间的拟合直线，发现其并不是明显的线性关系，不适用回归拟合，因此采用KNN方法寻找相似的对象进行插值补充。

# In[34]:


sns.regplot(x=data_Numerical['points'], y=data_Numerical['price'])


# In[201]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
data_Numerical_fit=pd.DataFrame(imputer.fit_transform(data_Numerical))
data_Numerical_fit=data_Numerical_fit.rename(columns={0:'points',1:'price'})
data_Numerical_3=data_Numerical.copy()
data_Numerical_3=data_Numerical_fit
print(data_Numerical_3['price'].isnull().sum()/data_Numerical_3['price'].shape[0])


# ##### 使用KNN填充前后，数值属性price的均值变化不大（34.18->33.84），另外的四分位数也都不变

# In[203]:


#原Price属性填充前后的直方图变化
fig=plt.figure(figsize=(20,10))
fig.add_subplot(1,2,1)
data_Numerical.price.hist(figsize=(20,5),alpha=0.7)
fig.add_subplot(1,2,2)
data_Numerical_3.price.hist(figsize=(20,5),alpha=0.7)
plt.show()


# In[204]:


#原Price属性填充前后的箱图变化
fig=plt.figure(figsize=(20,5))
fig.add_subplot(1,2,1)
data_Numerical.price.plot(kind='box')
fig.add_subplot(1,2,2)
data_Numerical_3.price.plot(kind='box')
plt.show()


# In[205]:


#price删除前后五数概括情况
print(data_Numerical.price.describe())
print(data_Numerical_3.price.describe())


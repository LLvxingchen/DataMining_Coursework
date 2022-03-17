#!/usr/bin/env python
# coding: utf-8

# ### 前言

# 本文档为针对Oakland Crime Statistics 2011 to 2016数据集探索性分析与预处理的报告。
# 报告整体分为三个模块介绍：  
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

#导入基本数据
import pandas as pd
data2011 = pd.read_csv("./records-for-2011.csv",encoding="utf-8")
data2012 = pd.read_csv("./records-for-2012.csv",encoding="utf-8")
data2013 = pd.read_csv("./records-for-2013.csv",encoding="utf-8")
data2014 = pd.read_csv("./records-for-2014.csv",encoding="utf-8")
data2015 = pd.read_csv("./records-for-2015.csv",encoding="utf-8")
data2016 = pd.read_csv("./records-for-2016.csv",encoding="utf-8")

data_all = pd.concat([data2011, data2012, data2013, data2014, data2015, data2016], axis=0)
data_all.info()


#观察数据集中的数据
data_all.head()

#找出各标称属性的可能取值频数：
for _ in ('Agency','Location','Area Id','Beat','Priority','Incident Type Id','Incident Type Description','Event Number','Location 1','Zip Codes','Location'):
    print(_,'属性取值频数：')
    print(data_all[_].value_counts())
    print()


#检查各属性的缺失率
data_all.isnull().sum()/data_all.shape[0]


#删除属性列Agency、Zip Codes、3个Location、Incident Type Description、Event Number、Create Time和Closed Time
#删除Priority为0的极少量数据
data_all = data_all.drop('Agency',1) 
data_all = data_all.drop('Zip Codes',1)  
data_all = data_all.drop('Location',1)  
data_all = data_all.drop('Location 1',1)  
data_all = data_all.drop('Location ',1) 
data_all = data_all.drop('Incident Type Description',1)
data_all = data_all.drop('Event Number',1)
data_all = data_all[~data_all['Priority'].isin([0])]
data_all = data_all.drop('Create Time',1)
data_all = data_all.drop('Closed Time',1)
print('接下来仍需要处理和分析的属性：')
data_all.columns


# #### 2.2数据可视化
# 由于本数据集中无有实际意义的数值属性，对标称数据的箱型图可视化意义不大，故**可视化部分主要对标称属性取值的分布情况展开**。
import matplotlib.pyplot as plt
import numpy as np
data_all1 = data_all.copy()

for _ in data_all1.columns:
    data_all1[_] = data_all1[_].astype(str)

fig,axes = plt.subplots(2,2,figsize=(20,10))

ax2 = axes[0,0]
ax2.set_title('Area Id')
ax2.hist(data_all1['Area Id'],bins=13,color='green',alpha=0.5)

ax3 = axes[0,1]
ax3.set_title('Beat')
ax3.hist(data_all1['Beat'],bins=58,color='green',alpha=0.5)

ax4 = axes[1,0]
ax4.set_title('Priority')
ax4.hist(data_all1['Priority'],bins=2,color='green',alpha=0.5)

ax5 = axes[1,1]
ax5.set_title('Incident Type Id')
ax5.hist(data_all1['Incident Type Id'],bins=286,color='green',alpha=0.5)

plt.show()



# ### 3.缺失值处理
# 对缺失值选用4种处理策略：  
# 1. 剔除缺失部分
# 2. 最高频率值填补
# 3. 属性的相关关系填补（无数值属性，不可用）
# 4. 数据对象相似性填补（无数值属性，不可用）

# #### 3.1 剔除缺失部分
#剔除含缺失数据
data_all2 = data_all.copy()
data_all2 = data_all2.dropna()
data_all2.isnull().sum()


#可视化剔除后数据分布与原可视化结果对比
for _ in data_all2.columns:
    data_all2[_] = data_all2[_].astype(str)

fig,axes = plt.subplots(2,2,figsize=(20,10))

ax2 = axes[0,0]
ax2.set_title('Area Id')
ax2.hist(data_all2['Area Id'],bins=12,color='green',alpha=0.5)

ax3 = axes[0,1]
ax3.set_title('Beat')
ax3.hist(data_all2['Beat'],bins=57,color='green',alpha=0.5)

ax4 = axes[1,0]
ax4.set_title('Priority')
ax4.hist(data_all2['Priority'],bins=1,color='green',alpha=0.5)

ax5 = axes[1,1]
ax5.set_title('Incident Type Id')
ax5.hist(data_all2['Incident Type Id'],bins=280,color='green',alpha=0.5)

plt.show()


# #### 3.2 最高频率值填补
#最高频率值填补
data_all3 = data_all.copy()
data_all3['Area Id'] = data_all3['Area Id'].fillna(data_all3['Area Id'].value_counts().index[0])
data_all3['Beat'] = data_all3['Beat'].fillna(data_all3['Beat'].value_counts().index[0])
data_all3['Priority'] = data_all3['Priority'].fillna(data_all3['Priority'].value_counts().index[0])
data_all3['Incident Type Id'] = data_all3['Incident Type Id'].fillna(data_all3['Incident Type Id'].value_counts().index[0])
data_all3.isnull().sum()


#可视化剔除后数据分布与原可视化结果对比
for _ in data_all3.columns:
    data_all3[_] = data_all3[_].astype(str)

fig,axes = plt.subplots(2,2,figsize=(20,10))

ax2 = axes[0,0]
ax2.set_title('Area Id')
ax2.hist(data_all3['Area Id'],bins=12,color='green',alpha=0.5)

ax3 = axes[0,1]
ax3.set_title('Beat')
ax3.hist(data_all3['Beat'],bins=57,color='green',alpha=0.5)

ax4 = axes[1,0]
ax4.set_title('Priority')
ax4.hist(data_all3['Priority'],bins=1,color='green',alpha=0.5)

ax5 = axes[1,1]
ax5.set_title('Incident Type Id')
ax5.hist(data_all3['Incident Type Id'],bins=285,color='green',alpha=0.5)

plt.xticks(rotation=270)
plt.show()


# ### 4. 探索性分析
# **采用剔除缺失数据后的数据集，展开探索性分析：**

# 1. 按照警局事件数据量绘制Area Id的条形统计图，可知警局事件量排名前五的区域为  **1.0、2.0、P3、P1、P2**
#警局事件数据量排序
data_all2['Area Id'].value_counts().plot(kind='bar',alpha=0.8)
plt.show()


# 2. 按照警局事件紧急程度绘制Area Id的条形统计图，可知整体来看**各地的2级事件普遍多于1级事件（即大事件较少）**，符合日常认知
#给各地事件按紧急程度排序，1级比2级更紧急
group = data_all2.groupby('Area Id')
group['Priority'].value_counts().unstack().plot(kind='barh', figsize=(10, 5),stacked=True)


# 3. 按照警局事件类型条形统计图，可知排名前五的为：  **933R(警铃响)、SECCK(安全检查)、415（酗酒或扰乱和平）、911H（拨打911挂断）、10851（车辆被盗）**  
#前50种警局事件类型排序
data_all2['Incident Type Id'].value_counts()[0:50].plot(kind='bar',figsize=(20,5),alpha=0.8)
plt.show()


# 4. 按照警察巡逻地点绘制条形统计图，可知排名前五的为： **04X、08X、30Y、26Y、19X**  
#警局警察巡逻地点排序
data_all2['Beat'].value_counts().plot(kind='bar',alpha=0.8,figsize=(20,5))
plt.show()


#!/usr/bin/env python
# coding: utf-8

# ### 前言

# 本文档为针对Wine Review数据集进行频繁模式与关联规则挖掘的报告。
# 报告整体分为四个模块介绍：  
# 
# 1. 数据集简介
#   
# 2. 任务分析
#   
# 3. 具体流程 
# 
# 4. 可视化展示与验证
# 
#   
# 以下将逐步展开介绍各部分内容以及操作过程。
# 
# ---

# ### 一. 数据集简介
# 
# 1. Wine Review是包含了两个数据子集的红酒评论数据集
# 2. 两个子数据集，分别含有150930和129971条数据
# 3. 数据集1、2公有属性种类共计11种：
# 
# | 属性名  | Unnamed: 0 | country | description | designation | points | price | province | region_1 | region_2 | variety | winery |
# |:----:|:----------:|:-------:|:-----------:|:-----------:|:------:|:-----:|:--------:|:--------:|:--------:|:-------:|:------:|
# | 中文名  | 序号         | 国家      | 口味描述        | 葡萄酒名称       | 打分     | 价格    | 省份       | 产地1      | 产地2      | 葡萄品种      | 酒厂     |
# | 数据类型 | 整型         | 字符串     | 字符串         | 字符串         | 整型     | 浮点数   | 字符串      | 字符串      | 字符串      | 字符串     | 字符串    |
# 
# 
# 
# 4. 数据集2独有属性3种：
# 
# | 属性名   | taster_name | taster_twitter_handle | title |
# |:-----:|:-----------:|:---------------------:|:-----:|
# | 中文名   | 品鉴师姓名       | 品鉴师推特                 | 标签    |
# | 数据类型  | 字符串         | 字符串                   | 字符串   |
# 
# ---

# ### 二. 任务分析
# 根据数据集情况，我们选取数据子集2，并将**｛points（打分）、price（价格）、region_1（主产地）、variety（葡萄品种）、taster_twitter_handle(品鉴师twitter)、winery（酒厂）｝**作为所有项集来挖掘其中的频繁模式，寻找并分析其中的关联规则。
# 
# ---

# ### 三. 具体流程
# ##### 0.导入数据集并进行预处理

# In[2]:


import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import pylab
get_ipython().run_line_magic('matplotlib', 'inline')

#导入数据集
data = pd.read_csv("./winemag-data-130k-v2.csv",encoding="utf-8")

#删除无需项集和所需项集值为空的数据
A=data.drop(['Unnamed: 0','country','description','designation','province','region_2','taster_name','title'],1).dropna()
display(A.columns)
A.describe()


# ##### 1.将数据转为适合进行关联规则挖掘的形式，构造事务集
# - points(打分)和price(价格)为数值数据，根据平均值将其转为标称数据
# - 剩余属性在原字符串后加上属性名作为标识，方便之后分析

# In[3]:


#处理points和price数据
A['points'] = A['points'].apply(lambda x:'high_points' if x > 89.0 else 'low_points' )
A['price'] = A['price'].apply(lambda x:'expensive' if x > 28.0 else 'cheap' )
A['region_1'] = A['region_1'].apply(lambda x:x+'_region')
A['taster_twitter_handle'] = A['taster_twitter_handle'].apply(lambda x:x+'_twitter')
A['variety'] = A['variety'].apply(lambda x:x+'_variety')
A['winery'] = A['winery'].apply(lambda x:x+'_winery')

#构造事务集
transactions = []
for index, row in A.iterrows():
    transactions += [(row['points'],row['price'],row['region_1'],row['taster_twitter_handle'],row['variety'],row['winery'])]

display(transactions[:5])


# ##### 2. 寻找频繁子项
# 
# 由于需要分析的属性数据量较大，而apriori搜索效率较低，我们参考使用改进后的[efficient-apriori](https://github.com/tommyod/Efficient-Apriori)算法，具体如下：

# In[1]:


import itertools
import numbers
import typing
import collections
from dataclasses import field, dataclass
import collections.abc

@dataclass
class ItemsetCount:
    itemset_count: int = 0
    members: set = field(default_factory=set)


class TransactionManager:
    def __init__(self, transactions: typing.Iterable[typing.Iterable[typing.Hashable]]):
        self._indices_by_item = collections.defaultdict(set)
        i = -1
        for i, transaction in enumerate(transactions):
            for item in transaction:
                self._indices_by_item[item].add(i)

        # Total number of transactions
        self._transactions = i + 1

    @property
    def items(self):
        return set(self._indices_by_item.keys())

    def __len__(self):
        return self._transactions

    def transaction_indices(self, transaction: typing.Iterable[typing.Hashable]):

        transaction = set(transaction)
        item = transaction.pop()
        indices = self._indices_by_item[item]
        while transaction:
            item = transaction.pop()
            indices = indices.intersection(self._indices_by_item[item])
        return indices

    def transaction_indices_sc(self, transaction: typing.Iterable[typing.Hashable], min_support: float = 0):
        transaction = sorted(transaction, key=lambda item: len(self._indices_by_item[item]), reverse=True)
        item = transaction.pop()
        indices = self._indices_by_item[item]
        support = len(indices) / len(self)
        if support < min_support:
            return False, None
        while transaction:
            item = transaction.pop()
            indices = indices.intersection(self._indices_by_item[item])
            support = len(indices) / len(self)
            if support < min_support:
                return False, None
        return True, indices


def join_step(itemsets: typing.List[tuple]):
    i = 0
    while i < len(itemsets):
        skip = 1
        *itemset_first, itemset_last = itemsets[i]
        tail_items = [itemset_last]
        tail_items_append = tail_items.append  # Micro-optimization
        for j in range(i + 1, len(itemsets)):
            *itemset_n_first, itemset_n_last = itemsets[j]
            if itemset_first == itemset_n_first:
                tail_items_append(itemset_n_last)
                skip += 1
            else:
                break
        itemset_first_tuple = tuple(itemset_first)
        for a, b in sorted(itertools.combinations(tail_items, 2)):
            yield itemset_first_tuple + (a,) + (b,)
        i += skip


def prune_step(itemsets: typing.Iterable[tuple], possible_itemsets: typing.List[tuple]):
    itemsets = set(itemsets)

    for possible_itemset in possible_itemsets:
        for i in range(len(possible_itemset) - 2):
            removed = possible_itemset[:i] + possible_itemset[i + 1 :]
            if removed not in itemsets:
                break
        else:
            yield possible_itemset


def apriori_gen(itemsets: typing.List[tuple]):
    possible_extensions = join_step(itemsets)
    yield from prune_step(itemsets, possible_extensions)


def itemsets_from_transactions(
    transactions: typing.Iterable[typing.Union[set, tuple, list]],
    min_support: float,
    max_length: int = 8,
    verbosity: int = 0,
    output_transaction_ids: bool = False,
):
    if not (isinstance(min_support, numbers.Number) and (0 <= min_support <= 1)):
        raise ValueError("`min_support` must be a number between 0 and 1.")

    manager = TransactionManager(transactions)

    transaction_count = len(manager)
    if transaction_count == 0:
        return dict(), 0  # large_itemsets, num_transactions
    if verbosity > 0:
        print("Generating itemsets.")
        print(" Counting itemsets of length 1.")

    candidates: typing.Dict[tuple, int] = {(item,): len(indices) for item, indices in manager._indices_by_item.items()}
    large_itemsets: typing.Dict[int, typing.Dict[tuple, int]] = {
        1: {item: count for (item, count) in candidates.items() if (count / len(manager)) >= min_support}
    }

    if verbosity > 0:
        print("  Found {} candidate itemsets of length 1.".format(len(manager.items)))
        print("  Found {} large itemsets of length 1.".format(len(large_itemsets.get(1, dict()))))
    if verbosity > 1:
        print("    {}".format(list(item for item in large_itemsets.get(1, dict()).keys())))

    if not large_itemsets.get(1, dict()):
        return dict(), 0  # large_itemsets, num_transactions
    k = 2
    while large_itemsets[k - 1] and (max_length != 1):
        if verbosity > 0:
            print(" Counting itemsets of length {}.".format(k))
        itemsets_list = sorted(item for item in large_itemsets[k - 1].keys())
        C_k: typing.List[tuple] = list(apriori_gen(itemsets_list))

        if verbosity > 0:
            print("  Found {} candidate itemsets of length {}.".format(len(C_k), k))
        if verbosity > 1:
            print("   {}".format(C_k))
        if not C_k:
            break
        if verbosity > 1:
            print("    Iterating over transactions.")

        found_itemsets: typing.Dict[tuple, int] = dict()
        for candidate in C_k:
            over_min_support, indices = manager.transaction_indices_sc(candidate, min_support=min_support)
            if over_min_support:
                found_itemsets[candidate] = len(indices)
        if not found_itemsets:
            break
        large_itemsets[k] = {i: counts for (i, counts) in found_itemsets.items()}

        if verbosity > 0:
            num_found = len(large_itemsets[k])
            print("  Found {} large itemsets of length {}.".format(num_found, k))
        if verbosity > 1:
            print("   {}".format(list(large_itemsets[k].keys())))
        k += 1
        if k > max_length:
            break

    if verbosity > 0:
        print("Itemset generation terminated.\n")

    if output_transaction_ids:
        itemsets_out = {
            length: {
                item: ItemsetCount(itemset_count=count, members=manager.transaction_indices(set(item)))
                for (item, count) in itemsets.items()
            }
            for (length, itemsets) in large_itemsets.items()
        }
        return itemsets_out, len(manager)
    return large_itemsets, len(manager)

class Rule(object):
    # Number of decimals used for printing
    _decimals = 3

    def __init__(
        self,
        lhs: tuple,
        rhs: tuple,
        count_full: int = 0,
        count_lhs: int = 0,
        count_rhs: int = 0,
        num_transactions: int = 0,
    ):
        self.lhs = lhs  # antecedent
        self.rhs = rhs  # consequent
        self.count_full = count_full
        self.count_lhs = count_lhs
        self.count_rhs = count_rhs
        self.num_transactions = num_transactions

    @property
    def confidence(self):
        try:
            return self.count_full / self.count_lhs
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def support(self):
        try:
            return self.count_full / self.num_transactions
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def lift(self):
        try:
            observed_support = self.count_full / self.num_transactions
            prod_counts = self.count_lhs * self.count_rhs
            expected_support = prod_counts / self.num_transactions**2
            return observed_support / expected_support
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def conviction(self):
        try:
            eps = 10e-10  # Avoid zero division
            prob_not_rhs = 1 - self.count_rhs / self.num_transactions
            prob_not_rhs_given_lhs = 1 - self.confidence
            return prob_not_rhs / (prob_not_rhs_given_lhs + eps)
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def rpf(self):
        try:
            return self.confidence * self.support
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @staticmethod
    def _pf(s):
        return "{" + ", ".join(str(k) for k in s) + "}"
    def __repr__(self):
        return "{} -> {}".format(self._pf(self.lhs), self._pf(self.rhs))
    def __str__(self):
        conf = "conf: {0:.3f}".format(self.confidence)
        supp = "supp: {0:.3f}".format(self.support)
        lift = "lift: {0:.3f}".format(self.lift)
        conv = "conv: {0:.3f}".format(self.conviction)
        return "{} -> {} ({}, {}, {}, {})".format(self._pf(self.lhs), self._pf(self.rhs), conf, supp, lift, conv)
    def __eq__(self, other):
        return (set(self.lhs) == set(other.lhs)) and (set(self.rhs) == set(other.rhs))
    def __hash__(self):
        return hash(frozenset(self.lhs + self.rhs))
    def __len__(self):
        return len(self.lhs + self.rhs)


def generate_rules_simple(itemsets: typing.Dict[int, typing.Dict],min_confidence: float,num_transactions: int,):
    for size in itemsets.keys():
        if size < 2:
            continue
        yielded: set = set()
        yielded_add = yielded.add
        for itemset in itemsets[size].keys():

            # Generate rules
            for result in _genrules(itemset, itemset, itemsets, min_confidence, num_transactions):
                if result in yielded:
                    continue
                else:
                    yielded_add(result)
                    yield result


def _genrules(l_k, a_m, itemsets, min_conf, num_transactions):
    def count(itemset):
        return itemsets[len(itemset)][itemset]
    for a_m in itertools.combinations(a_m, len(a_m) - 1):
        confidence = count(l_k) / count(a_m)
        if confidence < min_conf:
            continue
        rhs = set(l_k).difference(set(a_m))
        rhs = tuple(sorted(rhs))
        yield Rule(a_m, rhs, count(l_k), count(a_m), count(rhs), num_transactions)
        if len(a_m) <= 1:
            continue
        yield from _genrules(l_k, a_m, itemsets, min_conf, num_transactions)


def generate_rules_apriori(itemsets: typing.Dict[int, typing.Dict[tuple, int]],min_confidence: float,num_transactions: int,verbosity: int = 0,):
    if not ((0 <= min_confidence <= 1) and isinstance(min_confidence, numbers.Number)):
        raise ValueError("`min_confidence` must be a number between 0 and 1.")

    if not ((num_transactions >= 0) and isinstance(num_transactions, numbers.Number)):
        raise ValueError("`num_transactions` must be a number greater than 0.")

    def count(itemset):
        return itemsets[len(itemset)][itemset]

    if verbosity > 0:
        print("Generating rules from itemsets.")
    for size in itemsets.keys():
        if size < 2:
            continue

        if verbosity > 0:
            print(" Generating rules of size {}.".format(size))

        for itemset in itemsets[size].keys():
            for removed in itertools.combinations(itemset, 1):
                remaining = set(itemset).difference(set(removed))
                lhs = tuple(sorted(remaining))
                conf = count(itemset) / count(lhs)
                if conf >= min_confidence:
                    yield Rule(
                        lhs,
                        removed,
                        count(itemset),
                        count(lhs),
                        count(removed),
                        num_transactions,
                    )
            H_1 = list(itertools.combinations(itemset, 1))
            yield from _ap_genrules(itemset, H_1, itemsets, min_confidence, num_transactions)

    if verbosity > 0:
        print("Rule generation terminated.\n")
        
def _ap_genrules(itemset: tuple,H_m: typing.List[tuple],itemsets: typing.Dict[int, typing.Dict[tuple, int]],min_conf: float,num_transactions: int,):
    def count(itemset):
        return itemsets[len(itemset)][itemset]
    if len(itemset) <= (len(H_m[0]) + 1):
        return

    H_m = list(apriori_gen(H_m))
    H_m_copy = H_m.copy()

    for h_m in H_m:
        lhs = tuple(sorted(set(itemset).difference(set(h_m))))
        if (count(itemset) / count(lhs)) >= min_conf:
            yield Rule(
                lhs,
                h_m,
                count(itemset),
                count(lhs),
                count(h_m),
                num_transactions,
            )
        else:
            H_m_copy.remove(h_m)
    if H_m_copy:
        yield from _ap_genrules(itemset, H_m_copy, itemsets, min_conf, num_transactions)

def apriori(
    transactions: typing.Iterable[typing.Union[set, tuple, list]],
    min_support: float = 0.5,
    min_confidence: float = 0.5,
    max_length: int = 8,
    verbosity: int = 0,
    output_transaction_ids: bool = False,
):
    itemsets, num_trans = itemsets_from_transactions(
        transactions,
        min_support,
        max_length,
        verbosity,
        output_transaction_ids=True,
    )

    itemsets_raw = {
        length: {item: counter.itemset_count for (item, counter) in itemsets.items()}
        for (length, itemsets) in itemsets.items()
    }
    rules = generate_rules_apriori(itemsets_raw, min_confidence, num_trans, verbosity)

    if output_transaction_ids:
        return itemsets, list(rules)
    else:
        return itemsets_raw, list(rules)


# 设置support（支持度）和confidence（置信度）的最小阈值为0.03和0.7，找出满足条件的频繁项集与关联规则：
# - **由于数据集数量较多，故设置min_support ≥ 0.03**
# - **为保证获得置信度较高的关联规则，设置min_confidence ≥ 0.7**

# In[16]:


itemsets, rules = apriori(transactions, min_support=0.03,  min_confidence=0.7)
print('频繁项集：')
display(itemsets)


# ##### 3. 导出关联规则，计算其支持度和置信度，并使用Lift（提升度）和Conviction（确信度）来评价
# - Conf(置信度)：包含X和Y的事务数与所有包含X的事务数之比，也就是当项集X出现时，项集Y同时出现的概率
# - Sup(支持度)：事务集中同时包含X和Y的事务数与所有事务数之比
# - Lift(提升度)：越表明X和Y的相关性，lift<1负相关，lift=1独立，lift>1正相关
# - Conv(确信度)：表示X出现而Y不出现的概率，也就是规则预测错误的概率,它的值越大，表明X、Y的独立性越小

# In[71]:


import csv
pd.set_option('display.float_format',lambda x : '%.6f' % x)
pd.set_option('max_colwidth',100)
print('导出的关联规则与评价指标：')
rules_rhs = filter(lambda rule: (len(rule.lhs) >= 1), rules)
with open('result.csv', 'wt',encoding="utf-8") as f:
    f_csv = csv.writer(f, delimiter=',')
    f_csv.writerow(['rule','conf','sup','lift','conv'])
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift,reverse = True):
        f_csv.writerow([str(rule).split('(')[0],rule.confidence,rule.support,rule.lift,rule.conviction])
        
pd.read_csv('result.csv')


# ##### 4. 对挖掘出的27条关联规则进行分析

# **首先对于X、Y项集中项数都为1的情况:**

# 1. 葡萄产地为Mendoza的红酒 => 品鉴师@wineschach的关联置信度为1，葡萄品种为Bordeaux-style Red Blend_variety => 品鉴师@vossroger置信度0.7，即**该产地和葡萄品种的红酒大部分由这两位品鉴师品尝**
# 2. 品鉴师@wineschach => 价格cheap、打分low_points的置信度均大于0.7，lift1.3值左右，说明**他所品尝的红酒都较便宜，且打分较低**
# 3. 葡萄品种Rosé_variety => 价格cheap置信度0.9，说明**以该品种葡萄为原料的红酒价格普遍较低**
# 4. 葡萄品种Pinot Noir_variety => 价格expensive的置信度0.78，说明**以该品种葡萄为原料的红酒价格较高**
# 5. 打分high_points => 价格expensive的置信度0.75，**高分酒价格较高，符合常理**
# 6. 打分low_points => 价格cheap置信度0.71，价格cheap => 打分low_points置信度0.8,**低分酒便宜，便宜酒低分，也符合常理**

# In[70]:


rules_rhs = filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules)
with open('result_1-1.csv', 'wt',encoding="utf-8") as f:
    f_csv = csv.writer(f, delimiter=',')
    f_csv.writerow(['rule','conf','sup','lift','conv'])
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift,reverse = True):
        f_csv.writerow([str(rule).split('(')[0],rule.confidence,rule.support,rule.lift,rule.conviction])
        
pd.read_csv('result_1-1.csv')


# **对于X、Y项集中项数大于1的情况:**

# 1. {葡萄种类，高价(低价)} => ｛高分(低分)｝之类
# 2. {品鉴师，高价(低价)} => ｛高分(低分)｝、{品鉴师，高分(低分)} => ｛高价(低价)｝之类

# In[72]:


rules_rhs = filter(lambda rule: len(rule.lhs) > 1, rules)
with open('result_2-1.csv', 'wt',encoding="utf-8") as f:
    f_csv = csv.writer(f, delimiter=',')
    f_csv.writerow(['rule','conf','sup','lift','conv'])
    for rule in sorted(rules_rhs, key=lambda rule: rule.lift,reverse = True):
        f_csv.writerow([str(rule).split('(')[0],rule.confidence,rule.support,rule.lift,rule.conviction])
        
pd.read_csv('result_2-1.csv')


# ### 四.可视化展示与验证

# **1 以品鉴师@wineschach => 价格cheap、打分low_points为例，可视化查看具体情况进行检验，符合挖掘结果**

# In[102]:


fig = plt.figure(figsize=(20,5))
fig.add_subplot(1,2,1)
A[A['taster_twitter_handle']=='@wineschach_twitter']['price'].value_counts().plot(kind='bar',alpha=0.8)
plt.xticks(rotation=0)
plt.title('@wineschach - price')
fig.add_subplot(1,2,2)
A[A['taster_twitter_handle']=='@wineschach_twitter']['points'].value_counts().plot(kind='bar',alpha=0.8)
plt.xticks(rotation=0)
plt.title('@wineschach - points')
plt.show()


# **2 可视化检验两种葡萄品种(variety)与价格(cheap,expensive)之间的关系，符合挖掘结果**

# In[137]:


fig = plt.figure(figsize=(20,5))
fig.add_subplot(1,2,1)
A.groupby('variety').get_group('Rosé_variety').price.value_counts().plot(kind='barh',alpha=0.8)
plt.xticks(rotation=0)
plt.title('Rosé_variety - price')
fig.add_subplot(1,2,2)
A.groupby('variety').get_group('Pinot Noir_variety').price.value_counts().plot(kind='barh',alpha=0.8)
plt.xticks(rotation=0)
plt.title('Pinot Noir_variety - price')
plt.show()


# **3 可视化检验价格(cheap,expensive)与打分(low_points,high_points)之间的关系，符合挖掘结果**

# In[109]:


A.groupby('price').points.value_counts().plot(kind='line',figsize=(15,6),linestyle='-.',linewidth=3,alpha=0.8)


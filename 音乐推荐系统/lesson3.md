## 使用协同过滤基于movielens数据集构建模型并进行预测


```python
from surprise import KNNWithMeans
from surprise import Dataset
from surprise.model_selection import cross_validate

# 默认载入movielens数据集
data = Dataset.load_builtin('ml-100k')

algo = KNNWithMeans()
result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
```


```python
data.raw_ratings[1]
```


```python
"""
以下的程序段告诉大家如何在协同过滤算法建模以后，根据一个item取回相似度最高的item，主要是用到algo.get_neighbors()这个函数
"""
import os

from surprise import KNNBaseline
from surprise import Dataset


def read_item_names():
    """
    获取电影名到电影id 和 电影id到电影名的映射
    """

    file_name = (os.path.expanduser('~') + '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


# 首先，用算法计算相互间的相似度
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)
```


```python
# 获取电影名到电影id 和 电影id到电影名的映射
rid_to_name, name_to_rid = read_item_names()
```


```python
# 拿出来Toy Story这部电影对应的item id
toy_story_raw_id = name_to_rid['Toy Story (1995)']
toy_story_raw_id
```


```python
toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
toy_story_inner_id
```


```python
# 找到最近的10个邻居
toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)
toy_story_neighbors
```


```python
# 从近邻的id映射回电影名称
toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors)
toy_story_neighbors = (rid_to_name[rid] for rid in toy_story_neighbors)

print('The 10 nearest neighbors of Toy Story are:')
for movie in toy_story_neighbors:
    print(movie)
```


```python
# 拿出来Toy Story这部电影对应的item id
toy_story_raw_id = name_to_rid['Toy Story (1995)']
toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

# 找到最近的10个邻居
toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)

# 从近邻的id映射回电影名称
toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors)            
toy_story_neighbors = (rid_to_name[rid] for rid in toy_story_neighbors)
                       
print('The 10 nearest neighbors of Toy Story are:')
for movie in toy_story_neighbors:
    print(movie)
```


```python

```

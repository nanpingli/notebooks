## 使用协同过滤基于网易云音乐数据构建模型并进行预测


```python
import os
import pickle
from surprise import KNNBaseline, Reader
from surprise import Dataset

path = "./data/output/popular/"

# 重建歌单id到歌单名的映射字典
id_name_dic = pickle.load(open( path+"popular_playlist.pkl","rb"))
print("加载歌单id到歌单名的映射字典完成...")
# 重建歌单名到歌单id的映射字典
name_id_dic = {}
for playlist_id in id_name_dic:
    name_id_dic[id_name_dic[playlist_id]] = playlist_id
print("加载歌单名到歌单id的映射字典完成...")

file_path = os.path.expanduser(path+"popular_music_suprise_format.txt")
# 指定文件格式
reader = Reader(line_format='user item rating timestamp', sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path, reader=reader)
# 计算歌曲和歌曲之间的相似度
print("构建数据集...")
trainset = music_data.build_full_trainset()
#sim_options = {'name': 'pearson_baseline', 'user_based': False}
```


```python
i = list(id_name_dic.keys())[2]
i
```


```python
print(id_name_dic[i])
```


```python
trainset.n_items
```


```python
trainset.n_users
```

## 基于用户的协同过滤

主要思想：找出和当前用户兴趣相近的用户，针对网易云音乐歌单数据而言，这里的用户就是歌单


```python
print("开始训练模型...")
#sim_options = {'user_based': False}
#algo = KNNBaseline(sim_options=sim_options)
algo = KNNBaseline()

algo.fit(trainset)

current_playlist = list(name_id_dic.keys())[39]
print("歌单名称", current_playlist)

# 取出近邻
# 映射名字到id
playlist_id = name_id_dic[current_playlist]
print("歌单id", playlist_id)
# 取出来对应的内部user id => to_inner_uid
playlist_inner_id = algo.trainset.to_inner_uid(playlist_id)
print("内部id", playlist_inner_id)

playlist_neighbors = algo.get_neighbors(playlist_inner_id, k=10)

# 把歌曲id转成歌曲名字
# to_raw_uid映射回去
playlist_neighbors = (algo.trainset.to_raw_uid(inner_id)
                       for inner_id in playlist_neighbors)
playlist_neighbors = (id_name_dic[playlist_id]
                       for playlist_id in playlist_neighbors)

print()
print("和歌单 《", current_playlist, "》 最接近的10个歌单为：\n")
for playlist in playlist_neighbors:
    print(playlist, algo.trainset.to_inner_uid(name_id_dic[playlist]))
```

## 基于协同过滤的用户评分预测


```python
import pickle
# 重建歌曲id到歌曲名的映射字典
song_id_name_dic = pickle.load(open(path+"popular_song.pkl","rb"))
print("加载歌曲id到歌曲名的映射字典完成...")
# 重建歌曲名到歌曲id的映射字典
song_name_id_dic = {}
for song_id in song_id_name_dic:
    song_name_id_dic[song_id_name_dic[song_id]] = song_id
print("加载歌曲名到歌曲id的映射字典完成...")
```


```python
#内部编码的4号用户
user_inner_id = 4
user_rating = trainset.ur[user_inner_id]
items = map(lambda x:x[0], user_rating)
for song in items:
    print(algo.predict(user_inner_id, song, r_ui=1), song_id_name_dic[algo.trainset.to_raw_iid(song)])
```

## 基于矩阵分解的用户评分预测


```python
### 使用NMF
from surprise import NMF
from surprise import Dataset

file_path = os.path.expanduser(path+'./popular_music_suprise_format.txt')
# 指定文件格式
reader = Reader(line_format='user item rating timestamp', sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path, reader=reader)
# 构建数据集和建模
algo = NMF()
trainset = music_data.build_full_trainset()
algo.fit(trainset)
```


```python
user_inner_id = 4
user_rating = trainset.ur[user_inner_id]
items = map(lambda x:x[0], user_rating)
for song in items:
    print(algo.predict(algo.trainset.to_raw_uid(user_inner_id), algo.trainset.to_raw_iid(song), r_ui=1), song_id_name_dic[algo.trainset.to_raw_iid(song)])
```

## 模型保存与加载


```python
import surprise
surprise.dump.dump('./model/recommendation.model', algo=algo)
# 可以用下面的方式载入
algo = surprise.dump.load('./model/recommendation.model')
```

## 不同的推荐系统算法评估


```python
import os
from surprise import Reader, Dataset
# 指定文件路径
file_path = os.path.expanduser(path+'./popular_music_suprise_format.txt')
# 指定文件格式
reader = Reader(line_format='user item rating timestamp', sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path, reader=reader)
```


```python
from surprise.model_selection import cross_validate
```

### 使用BaselineOnly


```python
from surprise import BaselineOnly
algo = BaselineOnly()
result = cross_validate(algo, music_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

### 使用基础版协同过滤


```python
from surprise import KNNBasic
algo = KNNBasic()
result = cross_validate(algo, music_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

### 使用均值协同过滤


```python
from surprise import KNNWithMeans
algo = KNNWithMeans()
result = cross_validate(algo, music_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

### 使用协同过滤baseline


```python
from surprise import KNNBaseline
algo = KNNBaseline()
result = cross_validate(algo, music_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

### 使用SVD


```python
from surprise import SVD
algo = SVD()
result = cross_validate(algo, music_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

由于云平台资源有限，下面的代码没有继续运行演示，同学们要看效果的话可以重启一下，然后只运行部分算法

### 使用SVD++


```python
from surprise import SVDpp
algo = SVDpp()
result = cross_validate(algo, music_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

### 使用NMF


```python
from surprise import NMF
algo = NMF()
result = cross_validate(algo, music_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```


```python

```

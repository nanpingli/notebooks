## 歌曲序列建模

![](./pic/word2vec.png)

### 从word2vec到song2vec

我们把歌曲的id序列取出来，类比于分完词后的句子，送到word2vec中去学习一下，看看会有什么效果。


```python
import multiprocessing
import gensim
import sys
from random import shuffle

path = "./data/output/popular/"

def parse_playlist_get_sequence(in_line, playlist_sequence):
    song_sequence = []
    contents = in_line.strip().split("\t")
    # 解析歌单序列
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split(":::")
            song_sequence.append(song_id)
        except:
            print("song format error")
            print(song+"\n")
    for i in range(len(song_sequence)):
        shuffle(song_sequence)
        playlist_sequence.append(song_sequence)


def train_song2vec(in_file, out_file):
    #所有歌单序列
    playlist_sequence = []
    #遍历所有歌单
    for line in open(in_file,encoding='utf-8'):
        parse_playlist_get_sequence(line, playlist_sequence)
    #使用word2vec训练
    cores = multiprocessing.cpu_count()
    print("using all "+str(cores)+" cores")
    print("Training word2vec model...")
    model = gensim.models.Word2Vec(sentences=playlist_sequence, size=150, min_count=3, window=7, workers=cores)
    print("Saving model...")
    model.save(out_file)
```


```python
song_sequence_file = "./data/popular.playlist"
model_file = "./model/song2vec.model"
train_song2vec(song_sequence_file, model_file)
```

模型已经训练完了，咱们来试一把预测，看看效果。这个预测的过程，实际上就是对某首歌曲，查找“最近”的歌曲（向量距离最近的歌曲）


```python
import pickle
song_dic = pickle.load(open(path+"popular_song.pkl","rb"))
model = gensim.models.Word2Vec.load(model_file)
for song in list(song_dic.keys())[:10]:
    print(song, song_dic[song])

    
```


```python
song_id_list = list(song_dic.keys())[1000:1500:50]
for song_id in song_id_list:
    result_song_list = model.most_similar(song_id)
    print(song_id, song_dic[song_id])
    print("\n相似歌曲 和 相似度 分别为:")
    for song in result_song_list:
        print("\t", song_dic[song[0]], song[1])
    print("\n")
```

所以我们用word2vec学会了哪些歌曲和哪些歌曲最接近。


```python
song_id_list = list(song_dic.keys())[0:25000]
for song_id in song_id_list:
    if "我可以" in song_dic[song_id]:
        result_song_list = model.most_similar(song_id)
        print(song_id, song_dic[song_id])
        print("\n相似歌曲 和 相似度 分别为:")
        for song in result_song_list:
            print("\t", song_dic[song[0]], song[1])
        print("\n")
```


```python

```

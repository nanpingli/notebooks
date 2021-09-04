# Word2Vec (Word Embedding)

Implement Word2Vec algorithm to compute vector representations of words, with TensorFlow 2.0. This example is using a small chunk of Wikipedia articles to train from.

More info: [Mikolov, Tomas et al. "Efficient Estimation of Word Representations in Vector Space.", 2013](https://arxiv.org/pdf/1301.3781.pdf)

- Author: Aymeric Damien
- Project: https://github.com/aymericdamien/TensorFlow-Examples/


```python
from __future__ import division, print_function, absolute_import

import collections
import os
import random
import urllib
import zipfile

import numpy as np
import tensorflow as tf
```


```python
# Training Parameters.
learning_rate = 0.1
batch_size = 128
num_steps = 3000000
display_step = 10000
eval_step = 200000

# Evaluation Parameters.
eval_words = ['five', 'of', 'going', 'hardware', 'american', 'britain']

# Word2Vec Parameters.
embedding_size = 200 # Dimension of the embedding vector.
max_vocabulary_size = 50000 # Total number of different words in the vocabulary.
min_occurrence = 10 # Remove all words that does not appears at least n times.
skip_window = 3 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
num_sampled = 64 # Number of negative examples to sample.
```


```python
# Download a small chunk of Wikipedia articles collection.
url = 'http://mattmahoney.net/dc/text8.zip'
data_path = '../../../data/text8/text8.zip'
if not os.path.exists(data_path):
    print("Downloading the dataset... (It may take some time)")
    filename, _ = urllib.request.urlretrieve(url, data_path)
    print("Done!")
# Unzip the dataset file. Text has already been processed.
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()
```


```python
# Build the dictionary and replace rare words with UNK token.
count = [('UNK', -1)]
# Retrieve the most common words.
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))
# Remove samples with less than 'min_occurrence' occurrences.
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        # The collection is ordered, so stop when 'min_occurrence' is reached.
        break
# Compute the vocabulary size.
vocabulary_size = len(count)
# Assign an id to each word.
word2id = dict()
for i, (word, _)in enumerate(count):
    word2id[word] = i

data = list()
unk_count = 0
for word in text_words:
    # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary.
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count)
id2word = dict(zip(word2id.values(), word2id.keys()))

print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocabulary_size)
print("Most common words:", count[:10])
```

    Words count: 17005207
    Unique words: 253854
    Vocabulary size: 47135
    Most common words: [('UNK', 444176), (b'the', 1061396), (b'of', 593677), (b'and', 416629), (b'one', 411764), (b'in', 372201), (b'a', 325873), (b'to', 316376), (b'zero', 264975), (b'nine', 250430)]



```python
data_index = 0
# Generate training batch for the skip-gram model.
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # get window size (words left and right + current one).
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch.
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels
```


```python
# Ensure the following ops & var are assigned on CPU
# (some ops are not compatible on GPU).
with tf.device('/cpu:0'):
    # Create the embedding variable (each row represent a word embedding vector).
    embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
    # Construct the variables for the NCE loss.
    nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

def get_embedding(x):
    with tf.device('/cpu:0'):
        # Lookup the corresponding embedding vectors for each sample in X.
        x_embed = tf.nn.embedding_lookup(embedding, x)
        return x_embed

def nce_loss(x_embed, y):
    with tf.device('/cpu:0'):
        # Compute the average NCE loss for the batch.
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=y,
                           inputs=x_embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))
        return loss

# Evaluation.
def evaluate(x_embed):
    with tf.device('/cpu:0'):
        # Compute the cosine similarity between input data embedding and every embedding vectors
        x_embed = tf.cast(x_embed, tf.float32)
        x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))
        embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True), tf.float32)
        cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)
        return cosine_sim_op

# Define the optimizer.
optimizer = tf.optimizers.SGD(learning_rate)
```


```python
# Optimization process. 
def run_optimization(x, y):
    with tf.device('/cpu:0'):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            emb = get_embedding(x)
            loss = nce_loss(emb, y)

        # Compute gradients.
        gradients = g.gradient(loss, [embedding, nce_weights, nce_biases])

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, [embedding, nce_weights, nce_biases]))
```


```python
# Words for testing.
x_test = np.array([word2id[bytes(w,'utf8')] for w in eval_words])

# Run training for the given number of steps.
for step in range(1, num_steps + 1):
    batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0 or step == 1:
        loss = nce_loss(get_embedding(batch_x), batch_y)
        print("step: %i, loss: %f" % (step, loss))
        
    # Evaluation.
    if step % eval_step == 0 or step == 1:
        print("Evaluation...")
        sim = evaluate(get_embedding(x_test)).numpy()
        for i in range(len(eval_words)):
            top_k = 8  # number of nearest neighbors.
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = '"%s" nearest neighbors:' % eval_words[i]
            for k in range(top_k):
                log_str = '%s %s,' % (log_str, id2word[nearest[k]])
            print(log_str)
```

    step: 1, loss: 516.164490
    Evaluation...
    "five" nearest neighbors: b'embraces', b'domestic', b'banzer', b'convertible', b'vertigo', b'thoroughly', b'extracted', b'heartbreak',
    "of" nearest neighbors: b'imaginative', b'avant', b'ifr', b'depictions', b'utopia', b'avert', b'hussite', b'rockne',
    "going" nearest neighbors: b'scanning', b'talk', b'corse', b'muriel', b'grayson', b'fictions', b'haymarket', b'rubber',
    "hardware" nearest neighbors: b'ennis', b'kepler', b'silverstein', b'karin', b'affine', b'cytosol', b'beggars', b'lisa',
    "american" nearest neighbors: b'subproblems', b'fab', b'bixby', b'conditions', b'amplifying', b'slowing', b'supremely', b'dashed',
    "britain" nearest neighbors: b'baekje', b'temporarily', b'chew', b'espoused', b'nord', b'spooner', b'tricked', b'herbivore',
    step: 10000, loss: 122.647903
    step: 20000, loss: 92.400269
    step: 30000, loss: 43.487522
    step: 40000, loss: 23.921398
    step: 50000, loss: 48.567932
    step: 60000, loss: 47.483021
    step: 70000, loss: 20.345394
    step: 80000, loss: 23.524035
    step: 90000, loss: 46.382580
    step: 100000, loss: 33.049805
    step: 110000, loss: 29.985252
    step: 120000, loss: 19.932423
    step: 130000, loss: 30.541470
    step: 140000, loss: 27.805435
    step: 150000, loss: 11.529972
    step: 160000, loss: 19.734922
    step: 170000, loss: 12.101885
    step: 180000, loss: 19.633785
    step: 190000, loss: 8.505646
    step: 200000, loss: 14.760680
    Evaluation...
    "five" nearest neighbors: b'three', b'four', b'seven', b'two', b'eight', b'six', b'nine', b'zero',
    "of" nearest neighbors: b'a', b'the', b'from', b'first', b'with', b'his', b'and', b'was',
    "going" nearest neighbors: b'him', b'also', b'be', b'are', b'the', b'such', b'known', b'from',
    "hardware" nearest neighbors: b'towards', b'different', b'deep', b'some', b'are', b'individuals', b'others', b'have',
    "american" nearest neighbors: b'and', UNK, b's', b'by', b'when', b'over', b'in', b'or',
    "britain" nearest neighbors: b'became', b'its', b'who', b'when', b'see', b'both', b'most', b'like',
    step: 210000, loss: 17.457645
    step: 220000, loss: 16.506571
    step: 230000, loss: 11.811950
    step: 240000, loss: 6.618119
    step: 250000, loss: 9.542974
    step: 260000, loss: 13.374646
    step: 270000, loss: 9.489355
    step: 280000, loss: 7.066238
    step: 290000, loss: 8.168268
    step: 300000, loss: 15.673563
    step: 310000, loss: 13.514465
    step: 320000, loss: 10.461176
    step: 330000, loss: 8.491606
    step: 340000, loss: 12.054713
    step: 350000, loss: 12.160612
    step: 360000, loss: 7.302654
    step: 370000, loss: 8.413340
    step: 380000, loss: 8.953992
    step: 390000, loss: 6.772161
    step: 400000, loss: 9.735182
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'two', b'eight', b'seven', b'one', b'zero',
    "of" nearest neighbors: b'for', b'and', b'the', b'or', b'a', b'its', b'is', b'with',
    "going" nearest neighbors: b'being', b'be', b'the', b'are', b'which', b'his', b'because', b'known',
    "hardware" nearest neighbors: UNK, b'major', b'well', b'different', b'known', b'example', b'work', b'which',
    "american" nearest neighbors: b's', b'in', UNK, b'since', b'and', b'history', b'see', b'including',
    "britain" nearest neighbors: b'became', b'city', b'work', b'about', b'like', b'who', b'second', b'were',
    step: 410000, loss: 6.904413
    step: 420000, loss: 16.762432
    step: 430000, loss: 9.381282
    step: 440000, loss: 7.094061
    step: 450000, loss: 6.454022
    step: 460000, loss: 11.206313
    step: 470000, loss: 7.173163
    step: 480000, loss: 21.510647
    step: 490000, loss: 5.730925
    step: 500000, loss: 12.008631
    step: 510000, loss: 10.346094
    step: 520000, loss: 9.248569
    step: 530000, loss: 8.753950
    step: 540000, loss: 6.530839
    step: 550000, loss: 13.247885
    step: 560000, loss: 9.460489
    step: 570000, loss: 7.047210
    step: 580000, loss: 7.244588
    step: 590000, loss: 6.304128
    step: 600000, loss: 6.249343
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'seven', b'two', b'eight', b'one', b'zero',
    "of" nearest neighbors: b'and', b'the', b'including', b'with', b'while', b'in', b'by', b'which',
    "going" nearest neighbors: b'more', b'information', b'so', b'example', b'very', b'called', UNK, b'but',
    "hardware" nearest neighbors: b'others', b'high', b'free', UNK, b'well', b'include', b'such', b'large',
    "american" nearest neighbors: b'german', b'english', b'french', b's', b'since', b'after', b'see', b'during',
    "britain" nearest neighbors: b'at', b'but', b'than', b'when', b'the', b'well', b'another', b'city',
    step: 610000, loss: 3.984053
    step: 620000, loss: 6.084139
    step: 630000, loss: 6.194900
    step: 640000, loss: 8.752662
    step: 650000, loss: 8.276511
    step: 660000, loss: 8.329190
    step: 670000, loss: 6.659948
    step: 680000, loss: 4.843768
    step: 690000, loss: 7.457244
    step: 700000, loss: 6.985305
    step: 710000, loss: 12.266106
    step: 720000, loss: 5.908364
    step: 730000, loss: 8.085674
    step: 740000, loss: 7.493288
    step: 750000, loss: 12.553300
    step: 760000, loss: 8.693295
    step: 770000, loss: 12.455524
    step: 780000, loss: 7.711662
    step: 790000, loss: 9.400787
    step: 800000, loss: 6.427617
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'seven', b'two', b'eight', b'one', b'nine',
    "of" nearest neighbors: b'and', b'including', b'in', b'the', b'a', b'from', b'with', b'first',
    "going" nearest neighbors: b'information', b'out', b'so', b'them', b'own', b'which', b'general', b'then',
    "hardware" nearest neighbors: b'free', b'system', b'computer', b'information', b'data', b'high', b'other', b'general',
    "american" nearest neighbors: b'born', b'english', b'b', b'french', b'british', b'john', b'd', b'german',
    "britain" nearest neighbors: b'with', b'about', b'through', b'become', b'among', b'however', b'were', b'by',
    step: 810000, loss: 7.180316
    step: 820000, loss: 6.320290
    step: 830000, loss: 6.519843
    step: 840000, loss: 5.782719
    step: 850000, loss: 7.749493
    step: 860000, loss: 6.412918
    step: 870000, loss: 5.848679
    step: 880000, loss: 9.161039
    step: 890000, loss: 6.869849
    step: 900000, loss: 5.358594
    step: 910000, loss: 6.846594
    step: 920000, loss: 6.906522
    step: 930000, loss: 7.354561
    step: 940000, loss: 6.768304
    step: 950000, loss: 7.871628
    step: 960000, loss: 6.229767
    step: 970000, loss: 6.195165
    step: 980000, loss: 4.684243
    step: 990000, loss: 8.111429
    step: 1000000, loss: 5.277769
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'seven', b'eight', b'two', b'nine', b'zero',
    "of" nearest neighbors: b'the', b'including', b'and', b'first', b'became', b'from', b'under', b'history',
    "going" nearest neighbors: b'information', b'out', b'within', b'being', b'example', b'off', b'then', b'control',
    "hardware" nearest neighbors: b'computer', b'information', b'free', b'public', b'general', b'data', b'systems', b'space',
    "american" nearest neighbors: b'b', b'born', b'd', UNK, b'nine', b'john', b'seven', b'english',
    "britain" nearest neighbors: b'second', b'following', b'when', b'their', b'its', b'were', b'among', b'through',
    step: 1010000, loss: 5.688076
    step: 1020000, loss: 5.119394
    step: 1030000, loss: 9.941516
    step: 1040000, loss: 6.079704
    step: 1050000, loss: 6.933391
    step: 1060000, loss: 5.388718
    step: 1070000, loss: 7.375499
    step: 1080000, loss: 7.331379
    step: 1090000, loss: 5.897586
    step: 1100000, loss: 5.659865
    step: 1110000, loss: 7.238754
    step: 1120000, loss: 7.182052
    step: 1130000, loss: 6.380013
    step: 1140000, loss: 5.728670
    step: 1150000, loss: 5.818570
    step: 1160000, loss: 5.967332
    step: 1170000, loss: 6.886037
    step: 1180000, loss: 6.662858
    step: 1190000, loss: 6.884555
    step: 1200000, loss: 6.895592
    Evaluation...
    "five" nearest neighbors: b'four', b'six', b'three', b'seven', b'two', b'eight', b'zero', b'nine',
    "of" nearest neighbors: b'and', b'the', b'including', b'in', b'from', b'following', b'first', b'against',
    "going" nearest neighbors: b'then', b'off', b'out', b'through', b'so', b'up', b'information', b'played',
    "hardware" nearest neighbors: b'computer', b'public', b'information', b'data', b'free', b'space', b'general', b'a',
    "american" nearest neighbors: b'born', b'john', b'b', b'german', b'd', UNK, b'british', b's',
    "britain" nearest neighbors: b'following', b'the', b'within', b'its', b'of', b'although', b'on', b'second',
    step: 1210000, loss: 6.539179
    step: 1220000, loss: 7.087770
    step: 1230000, loss: 6.244791
    step: 1240000, loss: 6.414522
    step: 1250000, loss: 6.396931
    step: 1260000, loss: 6.948792
    step: 1270000, loss: 6.488492
    step: 1280000, loss: 9.906003
    step: 1290000, loss: 5.692191
    step: 1300000, loss: 7.403669
    step: 1310000, loss: 5.365675
    step: 1320000, loss: 5.934374
    step: 1330000, loss: 5.771346
    step: 1340000, loss: 5.618988
    step: 1350000, loss: 6.495852
    step: 1360000, loss: 10.497143
    step: 1370000, loss: 5.602166
    step: 1380000, loss: 6.320607
    step: 1390000, loss: 6.004068
    step: 1400000, loss: 7.443360
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'seven', b'two', b'eight', b'one', b'zero',
    "of" nearest neighbors: b'the', b'and', b'in', b'its', b'following', b'include', b'with', b'including',
    "going" nearest neighbors: b'off', b'played', b'so', b'then', b'right', b'group', b'led', b'all',
    "hardware" nearest neighbors: b'data', b'computer', b'information', b'high', b'free', b'space', b'towards', b'related',
    "american" nearest neighbors: b'english', b'french', b'german', b'born', b'film', b'david', b'references', b'john',
    "britain" nearest neighbors: b'between', b'europe', b'following', b'within', b'countries', b'people', b'modern', b'from',
    step: 1410000, loss: 7.202952
    step: 1420000, loss: 4.393835
    step: 1430000, loss: 5.566625
    step: 1440000, loss: 6.487858
    step: 1450000, loss: 5.184804
    step: 1460000, loss: 5.913851
    step: 1470000, loss: 6.087892
    step: 1480000, loss: 5.803067
    step: 1490000, loss: 5.005461
    step: 1500000, loss: 4.819779
    step: 1510000, loss: 6.694599
    step: 1520000, loss: 6.495092
    step: 1530000, loss: 5.132710
    step: 1540000, loss: 6.394836
    step: 1550000, loss: 6.431632
    step: 1560000, loss: 5.317185
    step: 1570000, loss: 5.523492
    step: 1580000, loss: 4.315282
    step: 1590000, loss: 6.238313
    step: 1600000, loss: 6.204445
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'two', b'seven', b'eight', b'one', b'zero',
    "of" nearest neighbors: b'the', b'and', b'including', b'in', b'its', b'original', b'following', b'within',
    "going" nearest neighbors: b'off', b'then', b'another', b'against', b'eventually', b'played', b'where', b'called',
    "hardware" nearest neighbors: b'data', b'computer', b'technology', b'towards', b'design', b'free', b'information', b'high',
    "american" nearest neighbors: b'born', b'english', b'british', b'french', b'german', b'john', b'b', b'actor',
    "britain" nearest neighbors: b'of', b'following', b'the', b'between', b'part', b'europe', b'in', b'through',
    step: 1610000, loss: 5.872983
    step: 1620000, loss: 5.151544
    step: 1630000, loss: 6.484306
    step: 1640000, loss: 4.828469
    step: 1650000, loss: 8.183352
    step: 1660000, loss: 6.722273
    step: 1670000, loss: 4.970653
    step: 1680000, loss: 5.678249
    step: 1690000, loss: 5.501106
    step: 1700000, loss: 6.204937
    step: 1710000, loss: 5.338465
    step: 1720000, loss: 5.728054
    step: 1730000, loss: 5.166333
    step: 1740000, loss: 6.394368
    step: 1750000, loss: 5.370402
    step: 1760000, loss: 12.485665
    step: 1770000, loss: 5.633874
    step: 1780000, loss: 6.004461
    step: 1790000, loss: 4.903961
    step: 1800000, loss: 7.071621
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'seven', b'eight', b'two', b'nine', b'zero',
    "of" nearest neighbors: b'the', b'and', b'including', b'in', b'from', b'became', b'first', b'history',
    "going" nearest neighbors: b'off', b'then', b'another', b'eventually', b'within', b'play', b'physical', b'while',
    "hardware" nearest neighbors: b'data', b'design', b'technology', b'towards', b'free', b'computer', b'software', b'information',
    "american" nearest neighbors: b'actor', b'b', b'd', b'born', UNK, b'robert', b'john', b'english',
    "britain" nearest neighbors: b'europe', b'within', b'following', b'second', b'during', b'france', b'ancient', b'world',
    step: 1810000, loss: 5.429294
    step: 1820000, loss: 5.504661
    step: 1830000, loss: 7.792231
    step: 1840000, loss: 5.692595
    step: 1850000, loss: 5.357494
    step: 1860000, loss: 7.885692
    step: 1870000, loss: 5.554457
    step: 1880000, loss: 5.539616
    step: 1890000, loss: 5.439644
    step: 1900000, loss: 5.122875
    step: 1910000, loss: 6.028441
    step: 1920000, loss: 4.976595
    step: 1930000, loss: 6.535644
    step: 1940000, loss: 5.815711
    step: 1950000, loss: 5.054225
    step: 1960000, loss: 6.398353
    step: 1970000, loss: 5.259978
    step: 1980000, loss: 5.717337
    step: 1990000, loss: 7.203055
    step: 2000000, loss: 7.163271
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'seven', b'eight', b'two', b'nine', b'zero',
    "of" nearest neighbors: b'the', b'and', b'including', b'following', b'within', b'from', b'its', b'in',
    "going" nearest neighbors: b'off', b'eventually', b'with', b'mostly', b'how', b'play', b'physical', b'then',
    "hardware" nearest neighbors: b'design', b'technology', b'computer', b'data', b'free', b'production', b'towards', b'information',
    "american" nearest neighbors: b'born', b'actor', b'd', b'b', b'robert', b'john', b'author', b'german',
    "britain" nearest neighbors: b'following', b'europe', b'france', b'ancient', b'england', b'established', b'country', b'within',
    step: 2010000, loss: 5.467506
    step: 2020000, loss: 5.562696
    step: 2030000, loss: 5.714738
    step: 2040000, loss: 6.572971
    step: 2050000, loss: 5.369168
    step: 2060000, loss: 5.154427
    step: 2070000, loss: 5.177800
    step: 2080000, loss: 4.594750
    step: 2090000, loss: 4.883307
    step: 2100000, loss: 5.332426
    step: 2110000, loss: 6.021369
    step: 2120000, loss: 7.020082
    step: 2130000, loss: 5.235673
    step: 2140000, loss: 6.557518
    step: 2150000, loss: 5.213410
    step: 2160000, loss: 5.090568
    step: 2170000, loss: 5.433657
    step: 2180000, loss: 5.603325
    step: 2190000, loss: 5.028516
    step: 2200000, loss: 6.143916
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'seven', b'eight', b'two', b'one', b'zero',
    "of" nearest neighbors: b'including', b'within', b'the', b'for', b'in', b'its', b'and', b'both',
    "going" nearest neighbors: b'off', b'eventually', b'physical', b'them', b'complete', b'mostly', b'according', b'once',
    "hardware" nearest neighbors: b'technology', b'design', b'computer', b'free', b'data', b'software', b'systems', b'special',
    "american" nearest neighbors: b'english', b'french', b'author', b'german', b'russian', b'born', b'canadian', b'film',
    "britain" nearest neighbors: b'following', b'europe', b'france', b'china', b'part', b'england', b'became', b'australia',
    step: 2210000, loss: 4.438079
    step: 2220000, loss: 6.097802
    step: 2230000, loss: 4.890207
    step: 2240000, loss: 7.806995
    step: 2250000, loss: 5.942187
    step: 2260000, loss: 5.043049
    step: 2270000, loss: 9.259454
    step: 2280000, loss: 6.049246
    step: 2290000, loss: 5.194667
    step: 2300000, loss: 5.619394
    step: 2310000, loss: 5.970448
    step: 2320000, loss: 5.038931
    step: 2330000, loss: 6.211797
    step: 2340000, loss: 5.389241
    step: 2350000, loss: 6.414670
    step: 2360000, loss: 5.520501
    step: 2370000, loss: 6.354387
    step: 2380000, loss: 5.730098
    step: 2390000, loss: 5.345685
    step: 2400000, loss: 6.483643
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'seven', b'two', b'eight', b'zero', b'nine',
    "of" nearest neighbors: b'the', b'in', b'and', b'including', b'from', b'following', b'within', b'original',
    "going" nearest neighbors: b'off', b'then', b'complete', b'take', b'go', b'another', b'physical', b'away',
    "hardware" nearest neighbors: b'technology', b'free', b'design', b'data', b'software', b'computer', b'systems', b'special',
    "american" nearest neighbors: b'actor', b'author', b'born', b'singer', b'b', b'french', b'british', b'english',
    "britain" nearest neighbors: b'china', b'europe', b'following', b'france', b'germany', b'country', b'australia', b'during',
    step: 2410000, loss: 5.938927
    step: 2420000, loss: 5.128527
    step: 2430000, loss: 5.187475
    step: 2440000, loss: 5.134459
    step: 2450000, loss: 5.292656
    step: 2460000, loss: 4.940915
    step: 2470000, loss: 8.910482
    step: 2480000, loss: 5.537578
    step: 2490000, loss: 5.235968
    step: 2500000, loss: 6.015111
    step: 2510000, loss: 5.231968
    step: 2520000, loss: 5.392875
    step: 2530000, loss: 6.788073
    step: 2540000, loss: 5.236230
    step: 2550000, loss: 5.038559
    step: 2560000, loss: 5.158301
    step: 2570000, loss: 5.557582
    step: 2580000, loss: 5.830987
    step: 2590000, loss: 6.187993
    step: 2600000, loss: 5.489066
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'seven', b'eight', b'nine', b'two', b'zero',
    "of" nearest neighbors: b'the', b'and', b'from', b'first', b'became', b'following', b'including', b'included',
    "going" nearest neighbors: b'off', b'out', b'take', b'away', b'eventually', b'go', b'another', b'play',
    "hardware" nearest neighbors: b'computer', b'technology', b'design', b'free', b'data', b'software', b'systems', b'special',
    "american" nearest neighbors: b'actor', b'author', b'singer', b'born', b'b', b'd', b'writer', b'robert',
    "britain" nearest neighbors: b'europe', b'france', b'western', b'china', b'following', b'germany', b'country', b'great',
    step: 2610000, loss: 8.115990
    step: 2620000, loss: 5.634456
    step: 2630000, loss: 5.228924
    step: 2640000, loss: 5.156111
    step: 2650000, loss: 5.086111
    step: 2660000, loss: 7.033471
    step: 2670000, loss: 6.301710
    step: 2680000, loss: 5.509211
    step: 2690000, loss: 6.186076
    step: 2700000, loss: 6.021070
    step: 2710000, loss: 5.280078
    step: 2720000, loss: 5.046353
    step: 2730000, loss: 6.267231
    step: 2740000, loss: 5.408620
    step: 2750000, loss: 4.743390
    step: 2760000, loss: 6.128206
    step: 2770000, loss: 5.871664
    step: 2780000, loss: 6.202782
    step: 2790000, loss: 5.145587
    step: 2800000, loss: 5.448274
    Evaluation...
    "five" nearest neighbors: b'four', b'six', b'three', b'seven', b'eight', b'two', b'nine', b'one',
    "of" nearest neighbors: b'in', b'the', b'and', b'following', b'including', b'from', b'under', b'along',
    "going" nearest neighbors: b'off', b'eventually', b'go', b'away', b'take', b'complete', b'another', b'play',
    "hardware" nearest neighbors: b'design', b'technology', b'computer', b'software', b'free', b'data', b'programs', b'systems',
    "american" nearest neighbors: b'actor', b'singer', b'born', b'author', b'writer', b'canadian', b'b', b'italian',
    "britain" nearest neighbors: b'france', b'europe', b'china', b'country', b'germany', b'western', b'england', b'australia',
    step: 2810000, loss: 5.841339
    step: 2820000, loss: 7.170264
    step: 2830000, loss: 8.719088
    step: 2840000, loss: 7.794249
    step: 2850000, loss: 5.473113
    step: 2860000, loss: 5.287600
    step: 2870000, loss: 6.463801
    step: 2880000, loss: 5.625960
    step: 2890000, loss: 5.873847
    step: 2900000, loss: 5.041293
    step: 2910000, loss: 5.148143
    step: 2920000, loss: 6.192270
    step: 2930000, loss: 4.839157
    step: 2940000, loss: 6.784483
    step: 2950000, loss: 4.633334
    step: 2960000, loss: 6.358610
    step: 2970000, loss: 5.581231
    step: 2980000, loss: 5.649710
    step: 2990000, loss: 6.275765
    step: 3000000, loss: 5.753077
    Evaluation...
    "five" nearest neighbors: b'four', b'three', b'six', b'seven', b'two', b'eight', b'zero', b'one',
    "of" nearest neighbors: b'the', b'and', b'in', b'within', b'including', b'from', b'with', b'original',
    "going" nearest neighbors: b'off', b'go', b'away', b'out', b'take', b'continued', b'play', b'eventually',
    "hardware" nearest neighbors: b'design', b'software', b'technology', b'computer', b'programs', b'data', b'free', b'systems',
    "american" nearest neighbors: b'canadian', b'irish', b'author', b'english', b'australian', b'french', b'singer', b'russian',
    "britain" nearest neighbors: b'china', b'france', b'england', b'europe', b'germany', b'established', b'great', b'western',



```python

```

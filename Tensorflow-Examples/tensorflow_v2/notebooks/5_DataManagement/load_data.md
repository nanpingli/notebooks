# Load and parse data with TensorFlow 2.0 (tf.data)

A TensorFlow 2.0 example to build input pipelines for loading data efficiently.


- Numpy Arrays
- Images
- CSV file
- Custom data from a Generator

For more information about creating and loading TensorFlow's `TFRecords` data format, see: [tfrecords.ipynb](tfrecords.ipynb)

- Author: Aymeric Damien
- Project: https://github.com/aymericdamien/TensorFlow-Examples/


```python
from __future__ import absolute_import, division, print_function

import numpy as np
import random
import requests
import string
import tarfile
import tensorflow as tf
```

### Load Numpy Arrays

Build a data pipeline over numpy arrays.


```python
# Create a toy dataset (even and odd numbers, with respective labels of 0 and 1).
evens = np.arange(0, 100, step=2, dtype=np.int32)
evens_label = np.zeros(50, dtype=np.int32)
odds = np.arange(1, 100, step=2, dtype=np.int32)
odds_label = np.ones(50, dtype=np.int32)
# Concatenate arrays
features = np.concatenate([evens, odds])
labels = np.concatenate([evens_label, odds_label])

# Load a numpy array using tf data api with `from_tensor_slices`.
data = tf.data.Dataset.from_tensor_slices((features, labels))
# Refill data indefinitely.  
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=100)
# Batch data (aggregate records together).
data = data.batch(batch_size=4)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)
```


```python
for batch_x, batch_y in data.take(5):
    print(batch_x, batch_y)
```

    tf.Tensor([ 9 94 29 85], shape=(4,), dtype=int32) tf.Tensor([1 0 1 1], shape=(4,), dtype=int32)
    tf.Tensor([68 57 88 41], shape=(4,), dtype=int32) tf.Tensor([0 1 0 1], shape=(4,), dtype=int32)
    tf.Tensor([51 19 18 56], shape=(4,), dtype=int32) tf.Tensor([1 1 0 0], shape=(4,), dtype=int32)
    tf.Tensor([70 84 99 32], shape=(4,), dtype=int32) tf.Tensor([0 0 1 0], shape=(4,), dtype=int32)
    tf.Tensor([40  0 25 28], shape=(4,), dtype=int32) tf.Tensor([0 0 1 0], shape=(4,), dtype=int32)



```python
# Note: If you are planning on calling multiple time,
# you can user the iterator way:
ite_data = iter(data)
for i in range(5):
    batch_x, batch_y = next(ite_data)
    print(batch_x, batch_y)

for i in range(5):
    batch_x, batch_y = next(ite_data)
    print(batch_x, batch_y)
```

    tf.Tensor([ 9 94 29 85], shape=(4,), dtype=int32) tf.Tensor([1 0 1 1], shape=(4,), dtype=int32)
    tf.Tensor([68 57 88 41], shape=(4,), dtype=int32) tf.Tensor([0 1 0 1], shape=(4,), dtype=int32)
    tf.Tensor([51 19 18 56], shape=(4,), dtype=int32) tf.Tensor([1 1 0 0], shape=(4,), dtype=int32)
    tf.Tensor([70 84 99 32], shape=(4,), dtype=int32) tf.Tensor([0 0 1 0], shape=(4,), dtype=int32)
    tf.Tensor([40  0 25 28], shape=(4,), dtype=int32) tf.Tensor([0 0 1 0], shape=(4,), dtype=int32)
    tf.Tensor([20 38 22 79], shape=(4,), dtype=int32) tf.Tensor([0 0 0 1], shape=(4,), dtype=int32)
    tf.Tensor([20 22 96 27], shape=(4,), dtype=int32) tf.Tensor([0 0 0 1], shape=(4,), dtype=int32)
    tf.Tensor([34 58 86 67], shape=(4,), dtype=int32) tf.Tensor([0 0 0 1], shape=(4,), dtype=int32)
    tf.Tensor([ 2 98 24 21], shape=(4,), dtype=int32) tf.Tensor([0 0 0 1], shape=(4,), dtype=int32)
    tf.Tensor([16 45 18 35], shape=(4,), dtype=int32) tf.Tensor([0 1 0 1], shape=(4,), dtype=int32)


### Load CSV files

Build a data pipeline from features stored in a CSV file. For this example, Titanic dataset will be used as a toy dataset stored in CSV format.

#### Titanic Dataset



survived|pclass|name|sex|age|sibsp|parch|ticket|fare
--------|------|----|---|---|-----|-----|------|----
1|1|"Allen, Miss. Elisabeth Walton"|female|29|0|0|24160|211.3375
1|1|"Allison, Master. Hudson Trevor"|male|0.9167|1|2|113781|151.5500
0|1|"Allison, Miss. Helen Loraine"|female|2|1|2|113781|151.5500
0|1|"Allison, Mr. Hudson Joshua Creighton"|male|30|1|2|113781|151.5500
...|...|...|...|...|...|...|...|...


```python
# Download Titanic dataset (in csv format).
d = requests.get("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/titanic_dataset.csv")
with open("titanic_dataset.csv", "wb") as f:
    f.write(d.content)
```


```python
# Load Titanic dataset.
# Original features: survived,pclass,name,sex,age,sibsp,parch,ticket,fare
# Select specific columns: survived,pclass,name,sex,age,fare
column_to_use = [0, 1, 2, 3, 4, 8]
record_defaults = [tf.int32, tf.int32, tf.string, tf.string, tf.float32, tf.float32]

# Load the whole dataset file, and slice each line.
data = tf.data.experimental.CsvDataset("titanic_dataset.csv", record_defaults, header=True, select_cols=column_to_use)
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=1000)
# Batch data (aggregate records together).
data = data.batch(batch_size=2)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)
```


```python
for survived, pclass, name, sex, age, fare in data.take(1):
    print(survived.numpy())
    print(pclass.numpy())
    print(name.numpy())
    print(sex.numpy())
    print(age.numpy())
    print(fare.numpy())
```

    [1 1]
    [2 2]
    ['Richards, Master. George Sibley' 'Rugg, Miss. Emily']
    ['male' 'female']
    [ 0.8333 21.    ]
    [18.75 10.5 ]


### Load Images

Build a data pipeline by loading images from disk. For this example, Oxford Flowers dataset will be used.


```python
# Download Oxford 17 flowers dataset
d = requests.get("http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz")
with open("17flowers.tgz", "wb") as f:
    f.write(d.content)
# Extract archive.
with tarfile.open("17flowers.tgz") as t:
    t.extractall()
```


```python
with open('jpg/dataset.csv', 'w') as f:
    c = 0
    for i in range(1360):
        f.write("jpg/image_%04i.jpg,%i\n" % (i+1, c))
        if (i+1) % 80 == 0:
            c += 1
```


```python
# Load Images
with open("jpg/dataset.csv") as f:
    dataset_file = f.read().splitlines()

# Load the whole dataset file, and slice each line.
data = tf.data.Dataset.from_tensor_slices(dataset_file)
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=1000)

# Load and pre-process images.
def load_image(path):
    # Read image from path.
    image = tf.io.read_file(path)
    # Decode the jpeg image to array [0, 255].
    image = tf.image.decode_jpeg(image)
    # Resize images to a common size of 256x256.
    image = tf.image.resize(image, [256, 256])
    # Rescale values to [-1, 1].
    image = 1. - image / 127.5
    return image
# Decode each line from the dataset file.
def parse_records(line):
    # File is in csv format: "image_path,label_id".
    # TensorFlow requires a default value, but it will never be used.
    image_path, image_label = tf.io.decode_csv(line, ["", 0])
    # Apply the function to load images.
    image = load_image(image_path)
    return image, image_label
# Use 'map' to apply the above functions in parallel.
data = data.map(parse_records, num_parallel_calls=4)

# Batch data (aggregate images-array together).
data = data.batch(batch_size=2)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)
```


```python
for batch_x, batch_y in data.take(1):
    print(batch_x, batch_y)
```

    tf.Tensor(
    [[[[-0.90260804 -0.9550551  -0.9444355 ]
       [-0.9538603  -0.9715073  -0.9136642 ]
       [-0.41687727 -0.37570083 -0.25462234]
       ...
       [ 0.4617647   0.422549    0.3754902 ]
       [ 0.4934436   0.45422792  0.4071691 ]
       [ 0.5530829   0.5138672   0.46680838]]
    
      [[-0.9301815  -0.98563874 -0.9595933 ]
       [-0.9379289  -0.95557594 -0.89773285]
       [-0.68581116 -0.6446346  -0.5305033 ]
       ...
       [ 0.46960783  0.43039215  0.38333333]
       [ 0.5009191   0.46170342  0.4146446 ]
       [ 0.56071925  0.52150357  0.4744447 ]]
    
      [[-0.9480392  -0.9862745  -0.96889937]
       [-0.93367803 -0.9485103  -0.8916054 ]
       [-0.9224341  -0.9033165  -0.7915518 ]
       ...
       [ 0.48045343  0.44123775  0.39417893]
       [ 0.51623774  0.47702205  0.42996323]
       [ 0.5740809   0.5348652   0.48780638]]
    
      ...
    
      [[ 0.0824219   0.37201285  0.5615885 ]
       [ 0.09744179  0.3858226   0.57758886]
       [ 0.1170305   0.4023859   0.59906554]
       ...
       [ 0.02599955  0.65661     0.7460593 ]
       [-0.0751493   0.6735256   0.7022212 ]
       [-0.06794965  0.73861444  0.7482958 ]]
    
      [[ 0.10942864  0.39136028  0.5135914 ]
       [ 0.18471968  0.4658088   0.5954542 ]
       [ 0.21578586  0.4813496   0.6320619 ]
       ...
       [ 0.22432214  0.676777    0.8324946 ]
       [ 0.10089612  0.73174024  0.7959444 ]
       [ 0.00907248  0.74025357  0.7495098 ]]
    
      [[ 0.15197992  0.43433285  0.54413676]
       [ 0.20049018  0.48284316  0.60343134]
       [ 0.2664752   0.5252987   0.6713772 ]
       ...
       [ 0.24040669  0.6644263   0.8296224 ]
       [ 0.10060894  0.7192364   0.78786385]
       [ 0.05363435  0.77765393  0.78206575]]]
    
    
     [[[-0.49571514 -0.2133621   0.6807555 ]
       [-0.52243936 -0.2322433   0.66971743]
       [-0.5502666  -0.24438429  0.6732628 ]
       ...
       [-0.61084557 -0.22653186  0.7019608 ]
       [-0.60784316 -0.21568632  0.65843004]
       [-0.6197916  -0.22585356  0.6411722 ]]
    
      [[-0.5225973  -0.24024439  0.6538732 ]
       [-0.54144406 -0.26501226  0.64094764]
       [-0.56139374 -0.27119768  0.6341878 ]
       ...
       [-0.6186887  -0.22824419  0.67053366]
       [-0.59662986 -0.22015929  0.6358456 ]
       [-0.6119485  -0.23387194  0.6130515 ]]
    
      [[-0.54999995 -0.26764703  0.61539805]
       [-0.56739867 -0.28504562  0.6056473 ]
       [-0.58733106 -0.297135    0.5988358 ]
       ...
       [-0.62097263 -0.22653186  0.62466395]
       [-0.60171235 -0.21739864  0.5984136 ]
       [-0.614951   -0.23063731  0.579271  ]]
    
      ...
    
      [[-0.49420047 -0.25567698 -0.29812205]
       [-0.5336498  -0.31243873 -0.34749448]
       [-0.5600954  -0.35433567 -0.38869584]
       ...
       [ 0.4558211   0.22837007  0.47150737]
       [ 0.49019605  0.24705881  0.4980392 ]
       [ 0.5021446   0.25900733  0.5099877 ]]
    
      [[-0.50617576 -0.29696214 -0.31009734]
       [-0.47532892 -0.28324962 -0.28901553]
       [-0.45759463 -0.28628123 -0.28675795]
       ...
       [ 0.46366423  0.2362132   0.4793505 ]
       [ 0.4980392   0.25490195  0.5058824 ]
       [ 0.5099877   0.26685047  0.51783085]]
    
      [[-0.45882356 -0.254902   -0.26274514]
       [-0.4185791  -0.23034382 -0.23034382]
       [-0.37365198 -0.21194851 -0.20410538]
       ...
       [ 0.46366423  0.2362132   0.4793505 ]
       [ 0.4980392   0.25490195  0.5058824 ]
       [ 0.5099877   0.26685047  0.51783085]]]], shape=(2, 256, 256, 3), dtype=float32) tf.Tensor([8 8], shape=(2,), dtype=int32)


### Load data from a Generator


```python
# Create a dummy generator.
def generate_features():
    # Function to generate a random string.
    def random_string(length):
        return ''.join(random.choice(string.ascii_letters) for m in xrange(length))
    # Return a random string, a random vector, and a random int.
    yield random_string(4), np.random.uniform(size=4), random.randint(0, 10)
```


```python
# Load a numpy array using tf data api with `from_tensor_slices`.
data = tf.data.Dataset.from_generator(generate_features, output_types=(tf.string, tf.float32, tf.int32))
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=100)
# Batch data (aggregate records together).
data = data.batch(batch_size=4)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)
```


```python
# Display data.
for batch_str, batch_vector, batch_int in data.take(5):
    print(batch_str, batch_vector, batch_int)
```

    tf.Tensor(['snDw' 'NvMp' 'sXsw' 'qwuk'], shape=(4,), dtype=string) tf.Tensor(
    [[0.22296238 0.03515657 0.3893014  0.6875752 ]
     [0.05003363 0.27605608 0.23262134 0.10671499]
     [0.8992419  0.34516433 0.29739627 0.8413017 ]
     [0.91913974 0.7142106  0.48333576 0.04300505]], shape=(4, 4), dtype=float32) tf.Tensor([ 2 10  4  1], shape=(4,), dtype=int32)
    tf.Tensor(['vdUx' 'InFi' 'nLzy' 'oklE'], shape=(4,), dtype=string) tf.Tensor(
    [[0.6512162  0.8695475  0.7012295  0.6849636 ]
     [0.00812997 0.01264008 0.7774404  0.44849646]
     [0.92055863 0.894824   0.3628448  0.85603875]
     [0.32219294 0.9767527  0.0307372  0.12051418]], shape=(4, 4), dtype=float32) tf.Tensor([9 7 4 0], shape=(4,), dtype=int32)
    tf.Tensor(['ULGI' 'dBbm' 'URgs' 'Pkpt'], shape=(4,), dtype=string) tf.Tensor(
    [[0.39586228 0.7472     0.3759462  0.9277406 ]
     [0.44489694 0.38694733 0.9592599  0.82675934]
     [0.12597603 0.299358   0.6940909  0.34155408]
     [0.3401377  0.97620344 0.6047712  0.51667166]], shape=(4, 4), dtype=float32) tf.Tensor([ 4 10  0  0], shape=(4,), dtype=int32)
    tf.Tensor(['kvao' 'wWvG' 'vrzf' 'cMgG'], shape=(4,), dtype=string) tf.Tensor(
    [[0.8090979  0.65837437 0.9732402  0.9298921 ]
     [0.67059356 0.91655296 0.52894515 0.8964492 ]
     [0.05753202 0.45829964 0.74948853 0.41164723]
     [0.42602295 0.8696292  0.57220364 0.9475169 ]], shape=(4, 4), dtype=float32) tf.Tensor([6 7 6 2], shape=(4,), dtype=int32)
    tf.Tensor(['kyLQ' 'kxbI' 'CkQD' 'PHlJ'], shape=(4,), dtype=string) tf.Tensor(
    [[0.29089147 0.6438517  0.31005543 0.31286424]
     [0.0937152  0.8887667  0.24011584 0.25746483]
     [0.47577712 0.53731906 0.9178111  0.3249844 ]
     [0.38328    0.39294246 0.08126572 0.5995307 ]], shape=(4, 4), dtype=float32) tf.Tensor([3 1 3 2], shape=(4,), dtype=int32)


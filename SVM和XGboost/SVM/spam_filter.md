# 使用SVM进行垃圾邮件分类

## 1.数据集说明

2006 TREC Public Spam Corpora 是一个公开的垃圾邮件语料库，由国际文本检索会议提供，分为英文数据集(trec06p)和中文数据集(trec06c);  
其中所含的邮件均来源于真实邮件保留了邮件的原有格式和内容,[垃圾邮件原始数据集](https://plg.uwaterloo.ca/~gvcormac/treccorpus06/)
下载后进行整理：  
* 正样本路径： ham_data  (样本数：21766)
* 负样本路径： spam_data (样本数：42854)
* 中文停用词：chinese_stop_words.txt

## 2.实现思路

* 对单个邮件进行数据预处理
    - 去除所有非中文字符，如标点符号、英文字符、数字、网站链接等特殊字符
    - 对邮件内容进行分词处理
    - 过滤停用词

* 创建特征矩阵和样本数据集
    - feature_maxtrix:shape=(samples, feature_word_nums)
    - leabel; shape = (samples, 1)
    - 词向量的选择：索引或word2vect,注意二者的区别
    
* 拆分数据集：训练数据集、测试数据集和验证数据集

* 选择模型，这里选择svm

* 训练、测试、调参

## 3.具体实现过程

导入相关库


```python
import os
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from scipy.stats import uniform
```

将邮件转换为特征词矩阵类


```python
class EmailToWordFeatures:
    '''
    功能:将邮件转换为特征词矩阵
    整个过程包括：
    - 对邮件内容进行分词处理
    - 去除所有非中文字符，如标点符号、英文字符、数字、网站链接等特殊字符
    - 过滤停用词
    - 创建特征矩阵
    '''
    def __init__(self,stop_word_file=None,features_vocabulary=None):
        
        self.features_vocabulary = features_vocabulary
        
        self.stop_vocab_dict = {}  # 初始化停用词
        if stop_word_file is not None:
            self.stop_vocab_dict = self._get_stop_words(stop_word_file)
    
    def text_to_feature_matrix(self,words,vocabulary=None,threshold =10):
        cv = CountVectorizer()
        if vocabulary is None:
            cv.fit(words)
        else:
            cv.fit(vocabulary)
        words_to_vect = cv.transform(words)
        words_to_matrix = pd.DataFrame(words_to_vect.toarray())  # 转换成索引矩阵
        print(words_to_matrix.shape)

        # 进行训练特征词选择，给定一个阈值，当单个词在所有邮件中出现的次数的在阈值范围内时及选为训练特征词、
        selected_features = []
        selected_features_index = []
        for key,value in cv.vocabulary_.items():
            if words_to_matrix[value].sum() >= threshold:  # 词在每封邮件中出现的次数与阈值进行比较
                selected_features.append(key)
                selected_features_index.append(value)
        words_to_matrix.rename(columns=dict(zip(selected_features_index,selected_features)),inplace=True)
        return words_to_matrix[selected_features]

    def get_email_words(self,email_path, spam=None):
        '''
        为缩短时间正负样本数各选1000
        '''
        self.max_email = 1000
        self.emails = email_path
        if os.path.isdir(self.emails):
            emails = os.listdir(self.emails)
            is_dir = True
        else:
            emails = [self.emails,]
            is_dir = False
            
        count = 0
        all_email_words = []
        for email in emails:
            if count >= self.max_email:  # 给定读取email数量的阈值
                break
            if is_dir:
                email_path = os.path.join(self.emails,email)
            email_words = self._email_to_words(email_path)
            all_email_words.append(' '.join(email_words))
            count += 1
        return all_email_words
        
    def _email_to_words(self, email):
        '''
        将邮件进行分词处理，去除所有非中文和停用词
        retrun:words_list
        '''   
        email_words = []
        with open(email, 'rb') as pf:
            for line in pf.readlines():
                line = line.strip().decode('gbk','ignore')
                if not self._check_contain_chinese(line):  # 判断是否是中文
                    continue
                word_list = jieba.cut(line, cut_all=False)  # 进行分词处理
                for word in word_list:
                    if word in self.stop_vocab_dict or not self._check_contain_chinese(word):
                        continue  # 判断是否为停用词
                    email_words.append(word)
            return email_words
      
    def _get_stop_words(self,file):
        '''
        获取停用词
        '''
        stop_vocab_dict = {}
        with open(file,'rb') as pf:
            for line in pf.readlines():
                line = line.decode('utf-8','ignore').strip()
                stop_vocab_dict[line] = 1
        return stop_vocab_dict
    
    def _check_contain_chinese(self,check_str):
        '''
        判断邮件中的字符是否有中文
        '''
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False
```

将正负邮件数据集转换为词特征列表，每项为一封邮件


```python
stop_word_file = './trec06c/chinese_stop_words.txt'
ham_file = './trec06c/ham_data'
spam_file = './trec06c/spam_data'
file = './trec06c/data'
```

如果下载原始数据集需运行下面程序


```python
# index_file= '../data/full/index'
# import shutil 
# with open(index_file, 'r') as fp:
#     for line in fp:
        
#         dst_name = os.path.split(line.split(" ")[1])[-2][-3:]  + '_' + os.path.split(line.split(" ")[1])[-1].strip()
#         if line.split(" ")[0] == "spam":
#             shutil.copy(os.path.join(line.split(" ")[1].strip()), os.path.join(spam_file, dst_name))
#         elif line.split(" ")[0] == "ham":
#             shutil.copy(os.path.join(line.split(" ")[1].strip()), os.path.join(ham_file, dst_name))          
```


```python
email_to_features = EmailToWordFeatures(stop_word_file=stop_word_file)
ham_words = email_to_features.get_email_words(ham_file)
spam_words = email_to_features.get_email_words(spam_file)
print('ham email numbers:',len(ham_words))
print('spam email numbers:',len(spam_words))
```

将所有邮件转换为特征词矩阵，及模型输入数据


```python
all_email = []
all_email.extend(ham_words)
all_email.extend(spam_words)
print('all test email numbers:',len(all_email))
words_to_matrix = email_to_features.text_to_feature_matrix(all_email)
print(words_to_matrix)
```

获取标签矩阵


```python
label_matrix = np.zeros((len(all_email),1))
label_matrix[0:len(ham_words),:] = 1 
```

## 4.使用svm模型进行训练


```python
# 拆分数据集
x_train,x_test,y_train,y_test = train_test_split(words_to_matrix,label_matrix,test_size=0.2,random_state=42)

# 使用LinearSVC模型进行训练
svc = LinearSVC(loss='hinge',dual=True)
param_distributions = {'C':uniform(0,10)}
rscv_clf =RandomizedSearchCV(estimator=svc, param_distributions=param_distributions,cv=3,n_iter=200,verbose=2)
rscv_clf.fit(x_train,y_train)

```


```python
print('best_params:',rscv_clf.best_params_)
```


```python
# 使用测试数据集进行测试
y_prab = rscv_clf.predict(x_test)
print('accuracy:',accuracy_score(y_prab,y_test))
```

## 5.分别选择一封正式邮件和垃圾邮件进行预测

<b>正式邮件内容如下：</b>

- 讲的是孔子后人的故事。一个老领导回到家乡，跟儿子感情不和，跟贪财的孙子孔为本和睦。
    老领导的弟弟魏宗万是赶马车的。  
    有个洋妞大概是考察民俗的，在他们家过年。  
    孔为本总想出国，被爷爷教育了。  
    最后，一家人基本和解。  
    顺便问另一类电影，北京青年电影制片厂的。中越战背景。一军人被介绍了一个对象，去相亲。女方是军队医院的护士，犹豫不决，总是在回忆战场上负伤的男友，好像还没死。最后  
    男方表示理解，归队了。  

<b>垃圾邮件如下：</b>

- 注塑、注塑机技术交流  
 
    中国注塑网http://www.yxx.com.cn   

      本着以注塑、注塑机技术交友之宗旨,中国注塑网完全免费注册加入.  
    可以浏览大量注塑技术资料,发布注塑人才求职招聘,发布供求信息,  
    自助建站，网络商店，企业名录,并可进入注塑技术交流论坛提出各  
    种注塑、注塑机疑难问题,中国注塑网资深注塑技术专家当场就相关  
    注塑技术问题的给您满意的回复!  

      热爱注塑及注塑机的朋友们,只要您在百忙之中从您电脑IE浏览器地  
    址栏输入中国注塑网或(www.yxx.com.cn)  定还您一份惊喜!  

    E-mail:yxx@133sh.com  

    _____________________________________________________________

    注:以上广告由"甲天下软件工作室"代发,欲联系邮件广告群发业务,请  
       与: http://www.jtx168.com 联系.  


```python
def email_to_predict_matrix(words,features):
    cv = CountVectorizer()
    words_to_vect = cv.fit_transform(words)
    words_to_marix = pd.DataFrame(words_to_vect.toarray())
    vocabulary = cv.vocabulary_
    
    words_numbers_list = [] # 特征词出现的次数列表
    for feature in features:
        if feature in cv.vocabulary_.keys():
            words_numbers_list.append(words_to_marix[vocabulary[feature]][0])
        else:
            words_numbers_list.append(0)
    words_numbers_matrix = pd.DataFrame([words_numbers_list],columns = features)
    return words_numbers_matrix
```


```python
valid_ham_email = './trec06c/valid_ham_email'
valid_spam_email = './trec06c/valid_spam_email'

email_to_features_valid = EmailToWordFeatures(stop_word_file=stop_word_file)
valid_ham_email_words = email_to_features_valid.get_email_words(valid_ham_email)
valid_spam_email_words = email_to_features_valid.get_email_words(valid_spam_email)

valid_ham_words_maxtrix = email_to_predict_matrix(valid_ham_email_words,words_to_matrix.columns)
valid_spam_words_maxtrix = email_to_predict_matrix(valid_spam_email_words,words_to_matrix.columns)
```


```python
print('测试正式邮件----------')
print('预测结果：',rscv_clf.predict(valid_ham_words_maxtrix))
```


```python
print('测试垃圾邮件：')
print('预测结果：',rscv_clf.predict(valid_spam_words_maxtrix))
```


```python

```


```python

```

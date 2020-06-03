#!/usr/bin/env python
# coding: utf-8

# # Run the code in accesnding order according to the number on each markdown for each section.

# ## 1. Imports

# In[25]:


import os
import sys
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from nltk.tokenize import RegexpTokenizer
import heapq
import math
import copy
import re
from num2words import num2words
from collections import OrderedDict
import pandas as pd
from itertools import permutations
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sn
from scipy import stats as s
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,auc
from matplotlib import pyplot
from keras.utils.np_utils import to_categorical   


# ## 2. Read all the files and create ground truth dictionary to know which file is present in which folder

# In[2]:


path = r'/media/rohit/New Volume/codes/IR/20_newsgroups/subset_Assignment_4'
files = []
files1 = []
folder_dict = {}
print(path)
for r, d, f in os.walk(path):
    for file in sorted(f):
        file_path = os.path.join(r, file)
        if file_path.split('/')[-2] not in folder_dict.keys():
            folder_dict[file_path.split('/')[-2]] = []
        else:
            folder_dict[file_path.split('/')[-2]].append(file.split('/')[-1])
#             print(file.split('/')[-1])
        print(file.split('/')[-1])
        files.append(os.path.join(r, file))
        files1.append(file.split('/')[-1])
print(len(files))


# In[3]:


folder_dict.keys()


# In[4]:


len(folder_dict['rec.sport.hockey'])


# ## 3. Preprocessing function

# In[5]:


def process_query(query):
    punctuations = '''!()-[]{};:"\,<>./?@#$=+%^&*_~'''
    table = str.maketrans({key:" " for key in punctuations})
    query = query.translate(table)
    table = str.maketrans({key:None for key in "'"})
    query = query.translate(table)
    query = query.lower()
    st = PorterStemmer()
    stemm_list = []
    stop = stopwords.words('english')
    tokenizer = RegexpTokenizer(r'\w+')
    query_list = tokenizer.tokenize(query)
    for j in query_list:
#         if j.isnumeric():
#             j = num2words(j)
        if j in stop:
            j = ''
        stemm_list.append(st.stem(j))
    return stemm_list


# ## 4. preprocessing and creating data dictionary

# In[6]:


data_dict = {}
data = []
count = 0
docid_list = []
for f in files:
#     print(f)
#     sys.exit()
    if '.htm' in f:
        continue
    else:
        filehandle = open(f,errors='ignore')
        # read a single line
        file = (filehandle.read().replace('\n',' '))
        data_dict[f.split('/')[-1]] = set(process_query(file))
        docid_list.append(f.split('/')[-1])
        count = count + 1
        print(count)
# close the pointer to that file
filehandle.close()


# ## 5. Term count and document frequency Dictionary

# In[7]:


def create_imp_dicts(X_train):
    term_count_dict = {}
    term_doc_freq = {}
    docid_list = []
    class_doc_count = {}
    count = 0
    for count,file in enumerate(X_train.iloc[:,0]):  
        docid = X_train.iloc[count,-1]
        class_id = X_train.iloc[count,-2]
        length = len(file)
        for term in file:
            if term not in term_count_dict.keys():
                term_count_dict[term] =  {}
            if docid not in term_count_dict[term].keys(): 
                term_count_dict[term][docid] = [1, length] 
            elif docid in term_count_dict[term].keys():
                term_count_dict[term][docid][0] = term_count_dict[term][docid][0] + 1
        for term in set(file):
            if term not in term_doc_freq.keys():
                term_doc_freq[term] = 1
            elif term in term_doc_freq.keys():
                term_doc_freq[term] = term_doc_freq[term] + 1
            if term not in class_doc_count:
                class_doc_count[term] = {}
                class_doc_count[term][class_id] = 1
            elif class_id not in class_doc_count[term]:
                class_doc_count[term][class_id] = 1
            else:
                class_doc_count[term][class_id] += 1
                
    
    number_docs = len(files)
    total_vocab = len(term_doc_freq.keys())
    print(number_docs,total_vocab)
    
    term_freq_dict = copy.deepcopy(term_count_dict)
    for i in term_freq_dict.keys():
        for j in term_freq_dict[i].keys():
            term_freq_dict[i][j] = term_freq_dict[i][j][0]/term_freq_dict[i][j][1]
    
    tf_idf_dict = copy.deepcopy(term_freq_dict)
    for i in tf_idf_dict.keys():
        for j in tf_idf_dict[i].keys():
            tf_idf_dict[i][j] = tf_idf_dict[i][j]*np.log(len(files)/(term_doc_freq[i]+1))
            
    return term_doc_freq,class_doc_count,tf_idf_dict,total_vocab


# ## 6. TF-IDF feature selection step

# In[ ]:


dataset


# In[53]:


def tf_idf_feature_selection_step(corpus,term_doc_freq,p):
    query = list(corpus)

    processed_query_set = set(query)
    query_term_count_dict = {}
    count = 0
    length = len(query)
    for term in query:
        if term not in query_term_count_dict.keys():
            query_term_count_dict[term] =  [1,length]
        else:
            query_term_count_dict[term][0] = query_term_count_dict[term][0] + 1 

    query_tfidf_dict = copy.deepcopy(query_term_count_dict)
    for i in query_tfidf_dict.keys():
        query_tfidf_dict[i] = (query_tfidf_dict[i][0]/query_tfidf_dict[i][1])*np.log(len(files)/(term_doc_freq.setdefault(i,0)+1))
    k_keys_sorted = heapq.nlargest(int(len(query)*(p/100)), query_tfidf_dict,key=query_tfidf_dict.get)
    return k_keys_sorted


# ## 7.load and Split the dataset

# In[9]:


def load_split_dataset(p = 30):
    df = pickle.load( open( "/media/rohit/New Volume/codes/IR/Assignment_5/dataset_NB.pkl", "rb" ) )
    X_train, X_test= train_test_split(df, test_size=(p/100))
    X_train=X_train.reset_index(drop=True)
    X_test=X_test.reset_index(drop=True)
    return X_train,X_test


# ## 8. Classwise data sorting

# In[10]:


def classwise(ground_truth_list, train):
    classwise_dict ={}
    for i in ground_truth_list:
        for j in range(train.shape[0]):
            if train.iloc[j,1] == i:
                if i not in classwise_dict.keys():
                    classwise_dict[i] = train.iloc[j,0]
                else:
                    classwise_dict[i] = classwise_dict[i] + train.iloc[j,0]
    return classwise_dict


# ## 9. counter 

# In[11]:


def Counter(anylist):
    count_dict = {}
    for i in anylist:
        if i not in count_dict.keys():
            count_dict[i] = 1
        else:
            count_dict[i] = count_dict[i] + 1
    return count_dict


# ## 10. For calculating corpus statistics

# In[12]:


def get_corpus_statistics(corpus,classwise_dict,selected_words):
    total_words = len(corpus)
    unique_words = len(selected_words)
    classwise_freq_dict = {}
    classwise_word_count = {}
    for i in classwise_dict.keys():
        word_freq_dict = Counter(classwise_dict[i])
        classwise_freq_dict[i] = {}
        for j in selected_words:
            classwise_freq_dict[i][j] = word_freq_dict.setdefault(j,0)
            if i not in classwise_word_count.keys():
                classwise_word_count[i] = word_freq_dict.setdefault(j,0)
            else:
                classwise_word_count[i] = classwise_word_count[i] + word_freq_dict.setdefault(j,0)
    return classwise_freq_dict, classwise_word_count, total_words, unique_words


# ## 11. Acc and conf matrix

# In[13]:


def calc_accuracy(predicted, true):
    tp_tn = len([1 for i in range(len(predicted)) if predicted[i]==true[i]])
    return tp_tn/len(predicted)


# In[14]:


def compute_confusion(predicted, true,ground_truth_list):
    confusion = np.zeros((len(ground_truth_list), len(ground_truth_list))).astype(int)
    for i in range(len(predicted)):
        confusion[ground_truth_list.index(predicted[i])][ground_truth_list.index(true[i])] += 1
    sn.heatmap(pd.DataFrame(confusion), annot=True,cmap='Blues', fmt='g')
    return confusion


# ## 12. Naive bayes

# In[64]:


def naive_bayes_classifier(p_split,p_feat_select,option):
    ground_truth_list = list(['comp.graphics', 'rec.sport.hockey', 'sci.med', 'sci.space', 'talk.politics.misc'])
    X_train,X_test = load_split_dataset(p_split)
    term_doc_freq,class_doc_count,tf_idf_dict,total_vocab = create_imp_dicts(X_train)
    _,__,tf_idf_dict_test,___ = create_imp_dicts(X_test)
    classwise_dict = classwise(ground_truth_list, X_train)
    corpus = []
    for i in classwise_dict.keys():
        corpus = corpus + classwise_dict[i]
    if option==1:
        selected_words = tf_idf_feature_selection_step(corpus,term_doc_freq,p_feat_select)
        selected_words = selected_words[:int(len(selected_words)*p_feat_select/100)]

    else: 
        selected_words = mutual_information_feature_selection(class_doc_count,classwise_dict,X_train, p_feat_select)

    classwise_freq_dict, classwise_word_count, total_words, unique_words = get_corpus_statistics(corpus,classwise_dict,selected_words)
    train_labels = Counter(X_train.iloc[:,1])
    test_labels = []
    predicted = []
    for i in range(X_test.shape[0]):
        test_labels.append(X_test.iloc[i,1])
        classes_words_probability = []
        for l in classwise_dict.keys():
            words_probability = 0
            for word in X_test.iloc[i,0]:
                fr, cn = classwise_freq_dict.setdefault(l,{}).setdefault(word,0), classwise_word_count[l]
                pp = (fr + 1) / (cn + unique_words)
                words_probability += np.log(pp)
            words_probability += np.log(train_labels[l] / X_train.shape[0])
            classes_words_probability.append(words_probability)
#         print("hiiiiii",classes_words_probability)
        predicted.append(ground_truth_list[np.argmax(classes_words_probability)])

    print(compute_confusion(predicted, test_labels,ground_truth_list))
    print(calc_accuracy(predicted, test_labels))
    y_test = []
    y_score = []
    for i in X_test.iloc[:,-2]:
        if i == ground_truth_list[0]:
            y_test.append(0)
        if i == ground_truth_list[1]:
            y_test.append(1)
        if i == ground_truth_list[2]:
            y_test.append(2)
        if i == ground_truth_list[3]:
            y_test.append(3)
        if i == ground_truth_list[4]:
            y_test.append(4)
    for i in predicted:
        if i == ground_truth_list[0]:
            y_score.append(0)
        if i == ground_truth_list[1]:
            y_score.append(1)
        if i == ground_truth_list[2]:
            y_score.append(2)
        if i == ground_truth_list[3]:
            y_score.append(3)
        if i == ground_truth_list[4]:
            y_score.append(4)
    y_test = to_categorical(y_test)
    y_score = to_categorical(y_score)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
    for i in range(5):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve {}(area = {})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


# In[59]:


naive_bayes_classifier(30,10,0)


# In[65]:


naive_bayes_classifier(50,10,1)


# In[62]:


naive_bayes_classifier(20,10,0)


# ## cosine similarity

# In[38]:


def cosine_sim(a, b):
    cos_sim = np.dot(a, np.transpose(b))/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim
def cosine_similarity(Q,D,k=5):
    sorted_cosine_similarity_dict = {}
    cosine_similarity_dict = {}
    for count,i in enumerate(Q.iloc[:,0]):
        test_doc = Q.iloc[count,-1]
        cosine_similarity_dict[test_doc] = {}
        for count1,d in enumerate(D.iloc[:,0]):
            train_doc = D.iloc[count1,-1]
            cosine_similarity_dict[test_doc][train_doc] = cosine_sim(i, d)
    for i in cosine_similarity_dict.keys():
        sorted_cosine_similarity_dict[i] = OrderedDict(sorted(cosine_similarity_dict[i].items(), key=lambda x: x[1],reverse =True))   
    final_cosine_similarity_dict = {}
    for i in sorted_cosine_similarity_dict.keys():
        final_cosine_similarity_dict[i] = {}
        for j in list(sorted_cosine_similarity_dict[i].keys())[:k]:
            final_cosine_similarity_dict[i][j] = D.iloc[(D.loc[D.iloc[:,-1]==j].index),-2].values[0]
    return final_cosine_similarity_dict


# ## Vectorize Train and test

# In[39]:


def vectorize(selected_words,tf_idf_dict,tf_idf_dict_test,X_train,X_test):
    X = []
    y = []
    train = pd.DataFrame(columns=['X','labels'])
    test = pd.DataFrame(columns=['X','labels'])
    vector_dict = {}
    for doc_id in X_test.iloc[:,-1]:
        vector_dict[doc_id] = np.zeros((len(selected_words)))
    count = 0
    for col,term in enumerate(selected_words):
        for doc in tf_idf_dict[term].keys():
            if doc not in vector_dict.keys():
                vector_dict[doc] = np.zeros((len(selected_words)))
                vector_dict[doc][col] = tf_idf_dict[term][doc]
            else:
                vector_dict[doc][col] = tf_idf_dict[term][doc]
        if term not in tf_idf_dict_test.keys():
            for doc in vector_dict.keys():
                vector_dict[doc][col] = 0
        else:
            for doc in tf_idf_dict_test[term].keys():
                vector_dict[doc][col] = tf_idf_dict_test[term][doc]
    for count,i in enumerate(X_train.iloc[:,-1]):
        X.append(vector_dict[i])
        y.append(X_train.iloc[count,-2])
    X_train['X'] = X
    X_train['labels'] = y
    X=[]
    y=[]
    
    for count,i in enumerate(X_test.iloc[:,-1]):
        X.append(vector_dict[i])
        y.append(X_test.iloc[count,-2])
    X_test['X'] = X
    X_test['labels'] = y
        
    return X_train,X_test


# ## mutual information

# In[40]:


def mutual_information_feature_selection(class_doc_count,classwise_dict,X_train,p_feat_select):
    #['comp.graphics', 'rec.sport.hockey', 'sci.med', 'sci.space', 'talk.politics.misc']
    class_term_MI = {}
    for class_id in classwise_dict.keys():
        class_term_MI[class_id] = {}
    for count,term in enumerate(class_doc_count.keys()):
        if count%100==0:
            print(count)
        class_ids=list(class_doc_count[term].keys())
        for class_id in class_ids:
            class_term_MI[class_id][term] = 0
            N11 = class_doc_count[term][class_id]
            if class_id == 'comp.graphics':
                N10 = class_doc_count[term].setdefault('rec.sport.hockey',0) + class_doc_count[term].setdefault('sci.med',0) + class_doc_count[term].setdefault('sci.space',0) + class_doc_count[term].setdefault('talk.politics.misc',0)
            elif class_id == 'rec.sport.hockey':
                N10 = class_doc_count[term].setdefault('comp.graphics',0) + class_doc_count[term].setdefault('sci.med',0) + class_doc_count[term].setdefault('sci.space',0) + class_doc_count[term].setdefault('talk.politics.misc',0)
            elif class_id == 'sci.med':
                N10 = class_doc_count[term].setdefault('comp.graphics',0) + class_doc_count[term].setdefault('rec.sport.hockey',0) + class_doc_count[term].setdefault('sci.space',0) + class_doc_count[term].setdefault('talk.politics.misc',0)
            elif class_id == 'sci.space':
                N10 = class_doc_count[term].setdefault('comp.graphics',0) + class_doc_count[term].setdefault('sci.med',0) + class_doc_count[term].setdefault('rec.sport.hockey',0) + class_doc_count[term].setdefault('talk.politics.misc',0)
            elif class_id == 'talk.politics.misc':
                N10 = class_doc_count[term].setdefault('comp.graphics',0) + class_doc_count[term].setdefault('sci.med',0) + class_doc_count[term].setdefault('sci.space',0) + class_doc_count[term].setdefault('rec.sport.hockey',0)
#             print(X_train.loc[X_train.iloc[:,-2] == class_id].shape[0])
            N01 = X_train.loc[X_train.iloc[:,-2] == class_id].shape[0] - N11
            N00 = X_train.shape[0] - (N01+N11+N10)
            N1_ = N10 + N11
            N_1 = N01 + N11
            N0_ = N01 + N00
            N_0 = N10 + N00
            N = N11 + N10 + N01 + N00
#             print(N11,N10,N01,N00,N)
            class_term_MI[class_id][term] = (N11/N)*math.log2((N*N11+1)/(N1_*N_1+1)) + (N01/N)*math.log2((N*N01+1)/(N0_*N_1+1)) + (N10/N)*math.log2((N*N10+1)/(N1_*N_0+1)) + (N00/N)*math.log2((N*N00+1)/(N0_*N_0+1))

    selected_words = []
    for class_id in class_term_MI.keys():
        class_term_MI[class_id] = OrderedDict(sorted(class_term_MI[class_id].items(), key=lambda x: x[1], reverse = True))
        length = len(class_term_MI[class_id].keys())
        for term in list(class_term_MI[class_id].keys())[:int(length*(p_feat_select/100))]:
            selected_words.append(term)
    
    return set(selected_words)
    


# ## KNN

# In[51]:


def KNN_classifier(p_split,p_feat_select,option,k):
    ground_truth_list = list(['comp.graphics', 'rec.sport.hockey', 'sci.med', 'sci.space', 'talk.politics.misc'])
    X_train,X_test = load_split_dataset(p_split)
    term_doc_freq,class_doc_count,tf_idf_dict,total_vocab = create_imp_dicts(X_train)
    _,__,tf_idf_dict_test,___ = create_imp_dicts(X_test)
    classwise_dict = classwise(ground_truth_list, X_train)
    corpus = []
    for i in classwise_dict.keys():
        corpus = corpus + classwise_dict[i]
    if option==1:
        selected_words = tf_idf_feature_selection_step(corpus,term_doc_freq,p_feat_select)
        selected_words = selected_words[:int(len(selected_words)*p_feat_select/100)]
    else: 
        selected_words = mutual_information_feature_selection(class_doc_count,classwise_dict,X_train, p_feat_select)
    print(len(selected_words))
    X_train,X_test = vectorize(selected_words,tf_idf_dict,tf_idf_dict_test,X_train,X_test)
    final_cosine_similarity_dict = cosine_similarity(X_test,X_train,k)
    predicted_labels = []
    for i in X_test.iloc[:,-1]:
        l=[]
        l=[final_cosine_similarity_dict[i][s] for s in final_cosine_similarity_dict[i].keys()]
        predicted_labels.append(s.mode(l)[0])
    
    print(compute_confusion(predicted_labels,X_test.iloc[:,-2],ground_truth_list))
    print(calc_accuracy(predicted_labels,X_test.iloc[:,-2]))
    y_test = []
    y_score = []
    for i in X_test.iloc[:,-2]:
        if i == ground_truth_list[0]:
            y_test.append(0)
        if i == ground_truth_list[1]:
            y_test.append(1)
        if i == ground_truth_list[2]:
            y_test.append(2)
        if i == ground_truth_list[3]:
            y_test.append(3)
        if i == ground_truth_list[4]:
            y_test.append(4)
    for i in predicted_labels:
        if i == ground_truth_list[0]:
            y_score.append(0)
        if i == ground_truth_list[1]:
            y_score.append(1)
        if i == ground_truth_list[2]:
            y_score.append(2)
        if i == ground_truth_list[3]:
            y_score.append(3)
        if i == ground_truth_list[4]:
            y_score.append(4)
    y_test = to_categorical(y_test)
    y_score = to_categorical(y_score)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
    for i in range(5):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve {}(area = {})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


# In[56]:


final_cosine_similarity_dict = KNN_classifier(50,10,0,5)


# In[57]:


final_cosine_similarity_dict = KNN_classifier(30,10,0,5)


# In[58]:


final_cosine_similarity_dict = KNN_classifier(20,10,0,5)


# ## for data prep(DO NOT RUN)

# In[ ]:


def create_dataset(data_dict,folder_dict):
    dataset = pd.DataFrame(columns=['X','labels','doc_id'])
    X = []
    y = []
    doc_id = []
    for count,i in enumerate(data_dict.keys()):
        if i in folder_dict['comp.graphics']:
            X.append(list(data_dict[i]))
            y.append('comp.graphics')
            doc_id.append(i)
        elif i in folder_dict['rec.sport.hockey']:
            X.append(list(data_dict[i]))
            y.append('rec.sport.hockey')
            doc_id.append(i)
        elif i in folder_dict['sci.med']:
            X.append(list(data_dict[i]))
            y.append('sci.med')
            doc_id.append(i)

        elif i in folder_dict['sci.space']:
            X.append(list(data_dict[i]))
            y.append('sci.space')
            doc_id.append(i)
        elif i in folder_dict['talk.politics.misc']:
            X.append(list(data_dict[i]))
            y.append('talk.politics.misc')
            doc_id.append(i)
    print(np.array(X).shape,np.array(y).shape)    
    dataset['X'] = X
    dataset['labels'] = np.array(y)
    dataset['doc_id'] = np.array(doc_id)
    dataset = dataset.sample(frac=1,random_state=4)        
    return dataset

dataset = create_dataset(data_dict,folder_dict)

dataset.to_pickle("/media/rohit/New Volume/codes/IR/Assignment_5/dataset_NB.pkl")


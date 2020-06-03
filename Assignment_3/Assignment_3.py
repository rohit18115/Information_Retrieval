#!/usr/bin/env python
# coding: utf-8

# In[109]:


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


# ## Read all the files

# In[110]:


path = r'/media/rohit/New Volume/codes/IR/20_newsgroups'
files = []
print(path)
for r, d, f in os.walk(path):
    for file in sorted(f):
        print(file.split('/')[-1])
        files.append(os.path.join(r, file))
print(len(files))


# In[111]:


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
        if j.isnumeric():
            j = num2words(j)
        elif j in stop:
            j = ''
        stemm_list.append(st.stem(j))
    return stemm_list


# ## Process query function

# In[112]:


data_dict = {}
data = []
count = 0
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
        count = count + 1
        print(count)
# close the pointer to that file
filehandle.close()


# ## Term count and document query Dictionary

# In[113]:


term_count_dict = {}
term_doc_freq = {}
count = 0
for f in files:
    if '.htm' in f:
        continue
    else:
        filehandle = open(f,errors='ignore')
            # read a single line
        file = (filehandle.read().replace('\n',' '))
        docid = f.split('/')[-1]
        file = process_query(file)
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
            


# ## Normalized Term Frequency

# In[114]:


term_freq_dict = copy.deepcopy(term_count_dict)
for i in term_freq_dict.keys():
    for j in term_freq_dict[i].keys():
        term_freq_dict[i][j] = term_freq_dict[i][j][0]/term_freq_dict[i][j][1]


# ## Sorted term frequency dictionary

# In[115]:


sorted_term_freq_dict = {}
for i in term_freq_dict.keys():
    sorted_term_freq_dict[i] = OrderedDict(sorted(term_freq_dict[i].items(), key=lambda x: x[1],reverse =True))


# ## High and low dictionary divided based on mean value

# In[150]:


high_dict_mean = {}
low_dict_mean = {}
for i in sorted_term_freq_dict.keys():
    mean = np.array([val for val in sorted_term_freq_dict[i].values()]).mean()
    high_dict_mean[i] = {key:val for key, val in sorted_term_freq_dict[i].items() if val >= mean}
    print(i," ",len(high_dict_mean[i].keys()))
    low_dict_mean[i] = {key:val for key, val in sorted_term_freq_dict[i].items() if val < mean }
    print(i," ",len(low_dict_mean[i].keys()))


# ## High and low dictionary divided based on median value

# In[152]:


high_dict_median = {}
low_dict_median = {}
for i in sorted_term_freq_dict.keys():
    median = np.median([val for val in sorted_term_freq_dict[i].values()])
    high_dict_median[i] = {key:val for key, val in sorted_term_freq_dict[i].items() if val >= median}
    print(i," ",len(high_dict_median[i].keys()))
    low_dict_median[i] = {key:val for key, val in sorted_term_freq_dict[i].items() if val < median }
    print(i," ",len(low_dict_median[i].keys()))


# In[121]:


median


# In[122]:


mean


# ## Calculating GD for all documents

# In[125]:


f = "/media/rohit/New Volume/codes/IR/Assignment_3/file.txt"
filehandle = open(f,errors='ignore')
file = (filehandle.read())


# In[126]:


file = file.split('\n')


# In[127]:


file = [s.split(' ') for s in file]


# In[128]:


gd_dict = {}
for count,f in enumerate(files):
#     print(f.split('/')[-1])
    gd_dict[f.split('/')[-1]] = int(file[count][1])


# In[129]:


factor=1.0/sum(gd_dict.values())
for k in gd_dict:
    gd_dict[k] = gd_dict[k]*factor


# ## Sorting the high and low list on the basis of GD

# In[ ]:


gd_high_dict = copy.deepcopy(high_dict)
for i in gd_high_dict.keys():
    for j in gd_high_dict[i].keys():
        gd_high_dict[i][j] = [gd_high_dict[i][j], gd_dict[j]]


# In[120]:


sorted_gd_high_dict = {}
for i in gd_high_dict.keys():
    sorted_gd_high_dict[i] = OrderedDict(sorted(gd_high_dict[i].items(), key=lambda x: x[1][1],reverse =True))


# In[121]:


sorted_gd_high_dict['hi']


# In[122]:


gd_low_dict = copy.deepcopy(low_dict)
for i in gd_low_dict.keys():
    for j in gd_low_dict[i].keys():
        gd_low_dict[i][j] = [gd_low_dict[i][j], gd_dict[j]]


# In[123]:


sorted_gd_low_dict = {}
for i in gd_low_dict.keys():
    sorted_gd_low_dict[i] = OrderedDict(sorted(gd_low_dict[i].items(), key=lambda x: x[1][1],reverse =True))


# In[124]:


sorted_gd_high_dict['hi']


# ## Query preprocessing and Query tf-idf

# In[177]:


query = input("Enter the query")
processed_query_list = process_query(query)


# In[178]:


processed_query_set = set(processed_query_list)


# In[184]:


def return_top_k_tfidf(processed_query_set, data_dict, high_dict, low_dict, term_doc_freq, k):
    score_high = {}
    score_low = {}
    docs = ()
    k_keys_sorted_high = []
    k_keys_sorted_low = []
    for term in processed_query_list:
        if len(docs) == 0:
            docs = set(high_dict[term].keys())
        else:
            docs = docs.intersection(set(high_dict[term].keys()))
    for docid in docs:
        for term in processed_query_list:
            if docid not in score_high.keys():
                score_high[docid] = 0
            score_high[docid] = score_high[docid] + high_dict.setdefault(term,{}).setdefault(docid,0)*np.log(len(files)/(term_doc_freq.setdefault(term,0)+1))# weights*title_tf_idf_dict.setdefault(term,{}).setdefault(docid,0) + (1-weights)*body_tf_idf_dict.setdefault(term,{}).setdefault(docid,0)
        score_high[docid] = score_high[docid] + gd_dict.setdefault(docid,0)
    k_keys_sorted_high = heapq.nlargest(min(k,len(score_high.keys())), score_high, key=score_high.get)
    if k > len(score_high.keys()):
        for docid in data_dict.keys():
            for term in processed_query_list:
                if docid not in score_low.keys():
                    score_low[docid] = 0
                score_low[docid] = score_low[docid] + low_dict.setdefault(term,{}).setdefault(docid,0)*np.log(len(files)/(term_doc_freq.setdefault(term,0)+1))
            score_low[docid] = score_low[docid] + gd_dict.setdefault(docid,0)
        k_keys_sorted_low = heapq.nlargest((k-len(score_high.keys())), score_low,key=score_low.get)
    
    return k_keys_sorted_high, k_keys_sorted_low, score_high, score_low


# In[185]:


k_high_docs_mean, k_low_docs_mean, score_high_mean, score_low_mean = return_top_k_tfidf(processed_query_set, data_dict, high_dict_mean, low_dict_mean, term_doc_freq, 20)
k_high_docs_median, k_low_docs_median, score_high_median, score_low_median = return_top_k_tfidf(processed_query_set, data_dict, high_dict_median, low_dict_median, term_doc_freq, 20)


# In[186]:


print("Documents from high list in case of mean: ", k_high_docs_mean)
print("length of high docs returned incase of mean:")
print(len(k_high_docs_mean))
print("Documents from low list in case of mean: ", k_low_docs_mean)
print("length of the low docs in case of mean:")
print(len(k_low_docs_mean))
print("Documents from high list in case of median: ", k_high_docs_median)
print("length of the high docs in case of median:")
print(len(k_high_docs_median))
print("Documents from low list in case of median: ", k_low_docs_median)
print("length of the low docs in case of median:")
print(len(k_low_docs_median))


# # Question 2 part 1

# In[2]:


filename = "/media/rohit/New Volume/codes/IR/Assignment_3/IR-assignment-3-data.txt"

# open the file for reading
filehandle = open(filename, 'r')
data = []
while True:
    # read a single line
    line = (filehandle.readline())
    data.append(line.split(' '))
    if not line:
        break

# close the pointer to that file
filehandle.close()


# In[3]:


data = pd.DataFrame(data)


# In[4]:


data = data.loc[data[1]=='qid:4']


# In[5]:


data_sorted = data.sort_values(by = 0, ascending= False)


# In[23]:


data_sorted


# ## Find number of Files that can be made when sorted on the basis of max DCG

# In[6]:


list_0 = data_sorted.index[data_sorted[0] == '0'].tolist()
list_1 = data_sorted.index[data_sorted[0] == '1'].tolist()
list_2 = data_sorted.index[data_sorted[0] == '2'].tolist()
list_3 = data_sorted.index[data_sorted[0] == '3'].tolist()


# In[22]:


len(list_2)


# In[23]:


import operator
from collections import Counter
from math import factorial
import functools
def npermutations(l):
    num = factorial(len(l))
    return num


# In[26]:


n = npermutations(list_0) * npermutations(list_1) * npermutations(list_2) * npermutations(list_3)


# In[27]:


n


# # Question 2 part 2

# In[38]:


def dcg_k(relevance,k,m = 0):
    r = np.array(relevance)
    num = 0
    for count,i in enumerate(r[:k]):
        if m == 0:
            num = num + (float(i)/math.log2(count + 2))
        if m == 1:
            num = num + ((math.pow(2,float(i))-1)/math.log2(count + 2))
    return num


# In[41]:


ndcg_50 = dcg_k(data.iloc[:,0],50,0)/dcg_k(data_sorted.iloc[:,0],50,0)
ndcg_whole = dcg_k(data.iloc[:,0],data.shape[0],0)/dcg_k(data_sorted.iloc[:,0],data.shape[0],0)


# In[43]:


ndcg_50, ndcg_whole


# ## Question 2 part 3

# In[74]:


for count,i in enumerate(data.iloc[:,76]):
    data.iloc[count,76] = float(str(i).split(':')[1])


# In[83]:


rel_tf_idf_score = data.loc[:,[0,76]].sort_values(by = 76,ascending = False)
rel_tf_idf_score


# In[106]:


relevant_retrieved = 0
total_relevant = len(list_1) + len(list_2) + len(list_3)
total = total_relevant + len(list_0)
recall_list = []
precision_list = []
for count,i in enumerate(rel_tf_idf_score.iloc[:,0]):
    if int(i) != 0:
        relevant_retrieved = relevant_retrieved + 1
        recall = relevant_retrieved/total_relevant
        precision = relevant_retrieved/(count+1)
        recall_list.append(recall)
        precision_list.append(precision)
    else:
        recall = relevant_retrieved/total_relevant
        precision = relevant_retrieved/(count+1)
        recall_list.append(recall)
        precision_list.append(precision)        


# In[ ]:





# In[107]:


from matplotlib import pyplot
pyplot.plot(recall_list, precision_list, marker='.')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# In[ ]:





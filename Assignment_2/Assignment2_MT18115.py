#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget http://archives.textfiles.com/stories.zip')


# In[348]:


get_ipython().system('pip install num2words')


# In[358]:


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


# In[5]:


path = r'/media/rohit/New Volume/codes/IR/Assignment_2/stories'
files = []
print(path)
for r, d, f in os.walk(path):
    for file in f:        
        files.append(os.path.join(r, file))
print(len(files))


# In[372]:


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


# In[373]:


# open the file for reading
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


# In[112]:


data_dict['zombies.txt']


# ## Preprocess Query

# ## Jaccard coeff

# In[60]:


def jaccard_coef(doc_set,query_set):
    numerator = len(doc_set & query_set)
    denominator = math.sqrt(len(doc_set | query_set))
    return numerator/denominator


# ## Return top k doc on the basis of JC

# In[122]:


def return_top_k(processed_query_set, data_dict,k):
    coef_dict = {}
    for i in data_dict.keys():
        #print(i)
        coef_dict[i] = jaccard_coef(data_dict[i],processed_query_set)
#         print(i," ",coef_dict[i])
    k_keys_sorted = heapq.nlargest(k, coef_dict,key=coef_dict.get)
    return k_keys_sorted, coef_dict


# In[123]:


k = []
coef_dict = {}
k, coef_dict = return_top_k(processed_query_set,data_dict, 5)


# In[124]:


k


# ## TF-IDF Body

# ## term count dictionary(contains count of the term and length of the document) and term document frequency

# In[91]:


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
            


# ## term_frequency dictionary

# In[95]:


term_freq_dict = copy.deepcopy(term_count_dict)
for i in term_freq_dict.keys():
    for j in term_freq_dict[i].keys():
        term_freq_dict[i][j] = term_freq_dict[i][j][0]/term_freq_dict[i][j][1]


# In[96]:


term_freq_dict['cat']['uglyduck.txt']


# ## log normalized term frequency

# In[97]:


log_term_freq_dict = copy.deepcopy(term_count_dict)
for i in log_term_freq_dict.keys():
    for j in log_term_freq_dict[i].keys():
        log_term_freq_dict[i][j] = 1 + math.log(log_term_freq_dict[i][j][0])


# ## Body tf-idf with term count 

# In[102]:


count_tf_idf_dict = copy.deepcopy(term_count_dict)
for i in count_tf_idf_dict.keys():
    for j in count_tf_idf_dict[i].keys():
        count_tf_idf_dict[i][j] = count_tf_idf_dict[i][j][0]*np.log(len(files)/(term_doc_freq[i]+1))


# ## Body tf-idf with normal term frequency

# In[341]:


tf_idf_dict = copy.deepcopy(term_freq_dict)
for i in tf_idf_dict.keys():
    for j in tf_idf_dict[i].keys():
        tf_idf_dict[i][j] = tf_idf_dict[i][j]*np.log(len(files)/(term_doc_freq[i]+1))


# ## Body tf-idf with log normalized term frequency

# In[342]:


log_tf_idf_dict = copy.deepcopy(log_term_freq_dict)
for i in log_tf_idf_dict.keys():
    for j in log_tf_idf_dict[i].keys():
        log_tf_idf_dict[i][j] = log_tf_idf_dict[i][j]*np.log(len(files)/(term_doc_freq[i]+1))


# In[154]:


title_dict = {}
# open the file for reading
filehandle = open("/media/rohit/New Volume/codes/IR/Assignment_2/stories/index.html",errors='ignore')
text = filehandle.read().strip()
filehandle.close()
##https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
file_name = re.findall('><A HREF="(.*)">', text)
file_title = re.findall('<BR><TD> (.*)\n', text)
filehandle = open("/media/rohit/New Volume/codes/IR/Assignment_2/stories/SRE/index.html",errors='ignore')
text = filehandle.read().strip()
filehandle.close()
##https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
file_name_SRE = re.findall('><A HREF="(.*)">', text)
file_title_SRE = re.findall('<BR><TD> (.*)\n', text)
file_name = file_name + file_name_SRE
file_name = file_name[2:]
file_title = file_title + file_title_SRE
for count,i in enumerate(file_name):
    title_dict[i] = file_title[count].split()


# In[155]:


title_dict['yukon.txt']


# In[132]:


file_name[0],file_title[0]


# ## title raw term count 

# In[134]:


title_term_count_dict = {}
title_term_doc_freq = {}
count = 0
for i,title in enumerate(file_title):
        title = process_query(title)
        length = len(title)
        docid = file_name[i]
        for term in title:
            if term not in title_term_count_dict.keys():
                title_term_count_dict[term] =  {}
            if docid not in title_term_count_dict[term].keys(): 
                title_term_count_dict[term][docid] = [1, length] 
            elif docid in title_term_count_dict[term].keys():
                title_term_count_dict[term][docid][0] = title_term_count_dict[term][docid][0] + 1
        for term in set(title):
            if term not in title_term_doc_freq.keys():
                title_term_doc_freq[term] = 1
            elif term in title_term_doc_freq.keys():
                title_term_doc_freq[term] = title_term_doc_freq[term] + 1


# In[135]:


title_term_count_dict['100']


# ## Title term frequency dictionary

# In[137]:


title_term_freq_dict = copy.deepcopy(title_term_count_dict)
for i in title_term_freq_dict.keys():
    for j in title_term_freq_dict[i].keys():
        title_term_freq_dict[i][j] = title_term_freq_dict[i][j][0]/title_term_freq_dict[i][j][1]


# In[138]:


title_term_freq_dict['100']


# ## title log term frequency dictionary

# In[139]:


title_log_term_freq_dict = copy.deepcopy(title_term_count_dict)
for i in title_log_term_freq_dict.keys():
    for j in title_log_term_freq_dict[i].keys():
        title_log_term_freq_dict[i][j] = 1 + math.log(title_log_term_freq_dict[i][j][0])


# In[140]:


title_log_term_freq_dict['100']


# ## title tf-idf with raw term count

# In[335]:


title_count_tf_idf_dict = copy.deepcopy(title_term_count_dict)
for i in title_count_tf_idf_dict.keys():
    for j in title_count_tf_idf_dict[i].keys():
        title_count_tf_idf_dict[i][j] = title_count_tf_idf_dict[i][j][0]*np.log(len(files)/(title_term_doc_freq[i]+1))


# In[338]:


title_count_tf_idf_dict['100']


# ## title tf-idf with normal term frequency

# In[336]:


title_tf_idf_dict = copy.deepcopy(title_term_freq_dict)
for i in title_tf_idf_dict.keys():
    for j in title_tf_idf_dict[i].keys():
        title_tf_idf_dict[i][j] = title_tf_idf_dict[i][j]*np.log(len(files)/(title_term_doc_freq[i]+1))


# In[337]:


title_tf_idf_dict['100']


# ## title tf-idf with log normalized term frequency
# 

# In[339]:


title_log_tf_idf_dict = copy.deepcopy(title_log_term_freq_dict)
for i in title_log_tf_idf_dict.keys():
    for j in title_log_tf_idf_dict[i].keys():
        title_log_tf_idf_dict[i][j] = title_log_tf_idf_dict[i][j]*np.log(len(files)/(title_term_doc_freq[i]+1))


# In[340]:


title_log_term_freq_dict['100']


# In[161]:


score = {}
for docid in data_dict.keys():
    for term in processed_query_list:
        if docid not in score.keys():
            score[docid] = 0
#         print(term,"7",title_dict[docid])
        if (term in title_dict[docid]) and (docid in count_tf_idf_dict[term].keys()):
#             print("hi")
            score[docid] = score[docid] + title_count_tf_idf_dict[term][docid] + count_tf_idf_dict[term][docid]
        elif (term in title_dict[docid]) and (docid not in count_tf_idf_dict[term].keys()):
            score[docid] = score[docid] + 0
        elif (term not in title_dict[docid]) and (docid in count_tf_idf_dict[term].keys()):
            score[docid] = score[docid] + count_tf_idf_dict[term][docid]
        elif (term not in title_dict[docid]) and (docid in count_tf_idf_dict[term].keys()):
            score[docid] = score[docid] + 0


# In[163]:


score['yukon.txt']


# In[343]:


def return_top_k_tfidf(processed_query_set, data_dict, title_tf_idf_dict, body_tf_idf_dict, k, weights):
    score = {}
    for docid in data_dict.keys():
        for term in processed_query_list:
            if docid not in score.keys():
                score[docid] = 0
    #         print(term,"7",title_dict[docid])
            score[docid] = score[docid] + weights*title_tf_idf_dict.setdefault(term,{}).setdefault(docid,0) + (1-weights)*body_tf_idf_dict.setdefault(term,{}).setdefault(docid,0)
    k_keys_sorted = heapq.nlargest(k, score,key=score.get)
    return k_keys_sorted, score


# In[ ]:


score[docid] = score[docid] + weights*title_count_tf_idf_dict.setdefault(term,{}).setdefault(docid,0) + (1-weights)*count_tf_idf_dict.setdefault(term,{}).setdefault(docid,0)


# In[243]:


term = 'gay'
docid = 'aesop11.txt'


# In[244]:


count_tf_idf_dict.setdefault(term,{}).setdefault(docid,0)


# In[242]:


count_tf_idf_dict['gay']


# In[308]:


query = input("Enter the query")
processed_query_list = process_query(query)


# In[55]:


processed_query_set = set(processed_query_list)


# In[247]:


k = []
coef_dict = {}
weights = 0.7

k, score_tfidf_dict = return_top_k_tfidf(processed_query_set,data_dict, 5,weights)


# In[248]:


k


# In[249]:


score_tfidf_dict[k[2]]


# In[250]:


score_tfidf_dict[k[1]]


# ## Query tf-idf

# In[309]:


query_term_count_dict = {}
count = 0
length = len(processed_query_list)
for term in processed_query_list:
    if term not in query_term_count_dict.keys():
        query_term_count_dict[term] =  [1,length]
    else:
        query_term_count_dict[term][0] = query_term_count_dict[term][0] + 1 


# In[310]:


query_term_count_dict.keys()


# In[311]:


query_tfidf_dict = copy.deepcopy(query_term_count_dict)
for i in query_tfidf_dict.keys():
    query_tfidf_dict[i] = (query_tfidf_dict[i][0]/query_tfidf_dict[i][1])*np.log(len(files)/(term_doc_freq.setdefault(i,0)+1))


# In[312]:


query_tf_dict['potti'] 


# ## Cosine similarity

# In[345]:


def cosine_similarity(query_tfidf_dict,title_count_tf_idf_dict,count_tf_idf_dict,weights,k):
    cosine_similarity_dict = {}
    for docid in data_dict.keys():
        numerator = 0
        denominator = 0
        temp1 = 0
        temp2 = 0
        for term in query_tfidf_dict.keys():
            doc_tfidf = (weights*title_count_tf_idf_dict.setdefault(term,{}).setdefault(docid,0) + (1-weights)*count_tf_idf_dict.setdefault(term,{}).setdefault(docid,0))
            numerator = numerator + (query_tfidf_dict[term] * doc_tfidf)
            temp1 = temp1 + math.pow(query_tfidf_dict[term],2)
        all_words_set = set(title_count_tf_idf_dict.keys())|set(count_tf_idf_dict.keys())
        for term in all_words_set:
            lol = (weights*title_count_tf_idf_dict.setdefault(term,{}).setdefault(docid,0) + (1-weights)*count_tf_idf_dict.setdefault(term,{}).setdefault(docid,0))
            temp2 = temp2 +math.pow(lol,2)
        denominator = math.sqrt(temp1) * math.sqrt(temp2)
        cosine_similarity_dict[docid] = numerator/denominator
#     print(cosine_similarity_dict)
    k_keys_sorted = heapq.nlargest(k, cosine_similarity_dict,key=cosine_similarity_dict.get)
    return k_keys_sorted, cosine_similarity_dict


# In[346]:


k = 5
k_docs = []
weights = 0.7
cosine_similarity_dict = {}
coef_dict = {}
score_tfidf_dict = {}
print("------------------------------Jaccard Coefficient--------------------------------")
k_docs, coef_dict = return_top_k(processed_query_set,data_dict, k)
print(k_docs)
print("------------------------------tf-idf with raw count as term frequency------------------------------")
k_docs, score_tfidf_dict = return_top_k_tfidf(processed_query_set,data_dict,title_count_tf_idf_dict, count_tf_idf_dict, k,weights)
print(k_docs)
print("------------------------------tf-idf with normal term frequency------------------------------------")
k_docs, score_tfidf_dict = return_top_k_tfidf(processed_query_set,data_dict,title_tf_idf_dict, tf_idf_dict, k,weights)
print(k_docs)
print("-----------------------tf-idf with normal log normalized term frequency----------------------------")
k_docs, score_tfidf_dict = return_top_k_tfidf(processed_query_set,data_dict,title_log_tf_idf_dict, log_tf_idf_dict, k,weights)
print(k_docs)
print("------------------------------------cosine similarity-----------------------------------------------")
k_docs, cosine_similarity_dict = cosine_similarity(query_tfidf_dict,title_count_tf_idf_dict,count_tf_idf_dict,weights,k)
print(k_docs)


# In[361]:


num2words(142)


# ## Edit distance

# In[435]:


edit_distance_dict = {}
data = []
count = 0
filehandle = open("/media/rohit/New Volume/codes/IR/Assignment_2/english2.txt",errors='ignore')
file = (filehandle.read().replace('\n',' '))
# print(file)
for term in file.split(' '):
    print(term)
    edit_distance_dict[term] = 0
# close the pointer to that file
filehandle.close()


# In[406]:


def edit_distance(word1,word2):
    i = 0
    j = 0
    matrix = np.zeros((len(word1)+1,len(word2)+1))
    while i < len(word1)+1:
        matrix[i][0] = i*2
        i = i + 1
    while j < len(word2)+1:
        matrix[0][j] = j*2
        j = j + 1
    i = 1
    while i < len(word1)+1:
        j = 1
        while j < len(word2)+1:
            if word1[i-1]==word2[j-1]:
                matrix[i][j] = min(matrix[i-1][j],matrix[i][j-1], matrix[i-1][j-1])
            else:
                matrix[i][j] = min(matrix[i-1][j] + 1,matrix[i][j-1]+2, matrix[i-1][j-1]+3)
            j = j+1
        i=i+1
    return matrix[len(word1)][len(word2)]  


# In[420]:


def edit_distance1(s1,s2):
    m = len(s1)
    n = len(s2)
    aux = np.zeros([m+1,n+1],dtype=int)
    dire = np.zeros([m+1,n+1],dtype=int)
    for i in range(0,m+1):
        aux[i][0] = i
    for i in range(0,n+1):
        aux[0][i] = i
    for i in range(1,m+1):
        for j in range(1,n+1):
            if(s1[i-1]==s2[j-1]):
                aux[i][j] = aux[i-1][j-1]
            else:
                mini = min(aux[i-1][j-1],aux[i][j-1],aux[i-1][j])
                aux[i][j] = 1+ mini
                if(mini == aux[i-1][j]):
                    dire[i][j] = -1
                elif(mini == aux[i][j-1]):
                    dire[i][j] = 1
    i = m
    j = n
    cost = 0
    while i>0 and j>0:
        if(s1[i-1]==s2[j-1]):
            #print("same")
            i-=1
            j-=1
        else:
            if dire[i][j] == 0:
                #print("replace")
                cost+=3
                i-=1
                j-=1
            elif dire[i][j] == 1:
                #print("insert")
                cost+=2
                j-=1
            else:
                #print("delete")
                cost+=1
                i-=1
        if(j==0):
            cost = cost + (1*i)
        if(i==0):
            cost = cost + (2*j)
    return cost


# In[386]:


query = input("Enter the query")
processed_query_list = process_query(query)


# In[403]:


cost,matrix=edit_distance('redlol','borneo')


# In[404]:


cost


# In[405]:


matrix


# In[441]:


def return_top_k_ed(query):
    tokenizer = RegexpTokenizer(r'\w+')
    query_list = tokenizer.tokenize(query)
    print(query_list)
    for i in query_list:
        if i not in edit_distance_dict.keys():
#             print('lol')
            for j in edit_distance_dict.keys():
#                 print('lel')
                edit_distance_dict[j] = edit_distance1(i,j)
            k_keys_sorted = heapq.nsmallest(k, edit_distance_dict,key=edit_distance_dict.get)
            print(k_keys_sorted)
        else:
            continue          


# In[443]:


return_top_k_ed('helo ad sory')


# In[431]:


edit_distance_dict.keys()


# In[ ]:





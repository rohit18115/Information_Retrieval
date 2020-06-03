#!/usr/bin/env python
# coding: utf-8

# # Run the code in accesnding order according to the number on each markdown for each section.

# ## 1. Imports

# In[45]:


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


# ## 2. Read all the files and create ground truth dictionary to know which file is present in which folder

# In[122]:


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


# ## 3. Preprocessing function

# In[4]:


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


# ## 4. preprocessing and creating data dictionary

# In[5]:


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

# In[6]:


term_count_dict = {}
term_doc_freq = {}
docid_list = []
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
            


# ## 6. Normalized Term Frequency

# In[7]:


term_freq_dict = copy.deepcopy(term_count_dict)
for i in term_freq_dict.keys():
    for j in term_freq_dict[i].keys():
        term_freq_dict[i][j] = term_freq_dict[i][j][0]/term_freq_dict[i][j][1]


# ## 7. tf-idf

# In[8]:


tf_idf_dict = copy.deepcopy(term_freq_dict)
for i in tf_idf_dict.keys():
    for j in tf_idf_dict[i].keys():
        tf_idf_dict[i][j] = tf_idf_dict[i][j]*np.log(len(files)/(term_doc_freq[i]+1))


# ## 8. vectorize tf-idf

# In[9]:


number_docs = len(files)
total_vocab = len(tf_idf_dict.keys())
print(number_docs,total_vocab)


# In[132]:


vector_dict = {}
count = 0
for col,term in enumerate(tf_idf_dict.keys()):
    for doc in tf_idf_dict[term].keys():
        
        if doc not in vector_dict.keys():
            count = count + 1 
            vector_dict[doc] = np.zeros((total_vocab))
            vector_dict[doc][col] = tf_idf_dict[term][doc]

        else:
            vector_dict[doc][col] = tf_idf_dict[term][doc]


# ## 9. Function to vectorize Query tf-idf

# In[70]:


def vectorize_query(total_vocab,tf_idf_dict,processed_query_list,query_tfidf_dict):
    Q = np.zeros((total_vocab))
    for col,term in enumerate(tf_idf_dict.keys()):
        if term in processed_query_list:
            print(col,term)
            Q[col] = query_tfidf_dict[term]
    return Q


# ## 10. Query preprocessing and Query tf-idf

# In[66]:


def query_step(total_vocab,tf_idf_dict):
    query = input("Enter the query")
    processed_query_list = process_query(query)
#     print(processed_query_list)

    processed_query_set = set(processed_query_list)

    query_term_count_dict = {}
    count = 0
    length = len(processed_query_list)
    for term in processed_query_list:
        if term not in query_term_count_dict.keys():
            query_term_count_dict[term] =  [1,length]
        else:
            query_term_count_dict[term][0] = query_term_count_dict[term][0] + 1 

    query_tfidf_dict = copy.deepcopy(query_term_count_dict)
    for i in query_tfidf_dict.keys():
        query_tfidf_dict[i] = (query_tfidf_dict[i][0]/query_tfidf_dict[i][1])*np.log(len(files)/(term_doc_freq.setdefault(i,0)+1))
    
    ground_truth = input("Enter ground truth folder: ")
    ground_truth_list = list(folder_dict[ground_truth])
    
    
    Q = vectorize_query(total_vocab,tf_idf_dict,processed_query_list,query_tfidf_dict)
    return ground_truth_list,Q


# ## 11. cosine similarity

# In[135]:


def cosine_sim(a, b):
    cos_sim = np.dot(a, np.transpose(b))/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim
def cosine_similarity(Q,D,k):
    cosine_similarity_dict = {}
    
    for d in D.keys():
#         print(D.keys())
        cosine_similarity_dict[d] = cosine_sim(Q, D[d])
#         print(cosine_similarity_dict[d])
        
    k_keys_sorted = heapq.nlargest(k, cosine_similarity_dict,key=cosine_similarity_dict.get)
    return k_keys_sorted, cosine_similarity_dict


# ## 12. Function to take feedback and print docs with * which are marked as relevant

# In[149]:


def get_feedback(result,relevant,fract,output_main):
    if len(output_main)==0:
        for i in result:
            output_main.append(i)
    for count,i in enumerate(result):
        if i in relevant:
            print(str(count)+" "+i+'*')
        else:
            print(str(count)+" "+i)
    result = [int(s) for s in result]
    number = len(result)*fract/100
    relevant = input("Enter at most "+str(int(number))+" relevant documents: ").split(' ')
#     relevant = [result[int(s)] for s in relevant]
    relevant = relevant[:(int(number))]
    non_relevant = set(result) - set(relevant)
    for count,i in enumerate(result):
        if str(i) in relevant:
            output_main[count] = (str(count)+" "+str(i)+'*')
        else:
            output_main[count] = output_main[count]
    return relevant, list(non_relevant), output_main


# ## 13. Function to calculate centroid

# In[16]:


def centroid(docid):
    l = []
    sum = []
    centroid = []
    for i in docid:
        l.append(vector_dict[str(i)])
    sum = np.sum(l,axis = 0)
    centroid = (sum/len(docid))
    return centroid


# ## 14. Function to generate TSNE

# In[17]:


def gen_tsne(Q_r, Q_nr, Q_m):
    tsne = TSNE(n_components=3, random_state=10)

    feature_vector = []
    labels = []

    for i in Q_r:
        feature_vector.append(vector_dict[str(i)])
        labels.append(0)
    for i in Q_nr:
        feature_vector.append(vector_dict[str(i)])
        labels.append(1)

    feature_vector.append(Q_m)
    labels.append(2)
    print(np.array(feature_vector).shape)
    transformed_data = tsne.fit_transform(np.array(feature_vector))
    k = np.array(transformed_data)
    print(k.shape)
    t = ("Relevant", "Non Relevant", "Query")
    plt.scatter(k[:, 0], k[:, 1], c=labels, s=60, alpha=0.8, label="Violet-R, Aqua-NR")
    plt.title("Rocchio Algorithm")
    plt.legend()
    plt.grid(True)
    plt.show()


# ## 15. Function to generate PR curve

# In[151]:


def get_PnR(output,ground_truth):
    relevant_retrieved = 0
    total_relevant = len(ground_truth)
    recall_list = []
    precision_list = []
    P = []
    print(output)
    for count,i in enumerate(output):
#         print("lol",i)
        if '*' in i:
            relevant_retrieved = relevant_retrieved + 1
            recall = relevant_retrieved/total_relevant
            precision = relevant_retrieved/(count+1)
            print(recall,precision)
            recall_list.append(recall)
            precision_list.append(precision)
            P.append(precision)
        else:
            recall = relevant_retrieved/total_relevant
            precision = relevant_retrieved/(count+1)
            recall_list.append(recall)
            precision_list.append(precision)
    AP = sum(P)/len(P)
    plt.plot(recall_list, precision_list, marker='.')
# axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
# show the legend
    plt.legend()
# show the plot
    plt.show()
    return AP


# ## 16. Main function to call all other functions and run rocchio iterations and return average precision 

# In[144]:


def iterations_rocchio(vector_dict,Q,ground_truth,iterations,alpha,beta,gamma,k,fract):
    flag = False
    output_main = []
    for i in range(iterations):
        if flag == True:
            relevant, non_relevant, prev = previous[0],previous[1],previous[2]
            new = (alpha*prev) + (beta*centroid(relevant)) - (gamma*centroid(non_relevant))
        else:
            new = Q
            flag = True
            relevant = set()
            non_relevant = set()
        
        result,_ = cosine_similarity(new,vector_dict,k)
        print(set(result)&set(ground_truth))
        docs0, docs1, output = get_feedback(result,str(relevant),fract,output_main)
        output_main = output
        relevant = relevant | set(docs0)
        non_relevant = non_relevant | set(docs1)
        print(non_relevant)
        print("relevant: "+str(relevant))
        print("Non relevant: "+str(non_relevant))
        previous = [relevant,non_relevant,new]
        gen_tsne(relevant,non_relevant,new)
        AP=get_PnR(output_main,ground_truth)
        print("Average Precision: ",AP)

    return AP


# ## Run rocchio itterations and plot TSNE and PR curve and calculate mean average precision

# In[150]:


ground_truth,Q1 = query_step(total_vocab, tf_idf_dict)
AP1 = iterations_rocchio(vector_dict,Q1,ground_truth,3,1,0.7,0.25,100,10)
ground_truth,Q2 = query_step(total_vocab, tf_idf_dict)
AP2 = iterations_rocchio(vector_dict,Q2,ground_truth,3,1,0.7,0.25,100,10)
ground_truth,Q3 = query_step(total_vocab, tf_idf_dict)
AP3 = iterations_rocchio(vector_dict,Q3,ground_truth,3,1,0.7,0.25,100,10)
MAP = (AP1 + AP2 + AP3)/3
print(MAP)


# In[ ]:





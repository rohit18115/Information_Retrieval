#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
nltk.download()


# # Basic Text Pre-processing of text data
# ## Lower casing
# ## Punctuation removal
# ## Stopwords removal
# ## Spelling correction
# ## Tokenization
# ## Stemming
# ## Lemmatization
# # plan
# ## do all the preprocessing 
# ## do tokenization
# ## find terms from the tokens
# ## make inverted index
# ## input query
# ## preprocess query
# ## do operations using query and inverted index

# # Read the data

# In[107]:


import os
import sys
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np


# In[108]:


path = r'/media/rohit/New Volume/codes/IR/20_newsgroups'
files = []
print(path)
for r, d, f in os.walk(path):
    for file in f:        
        files.append(os.path.join(r, file))
print(len(files))


# In[109]:



# open the file for reading
data_dict = {}
data = []
count = 0
for f in files:
    print(f)
#     sys.exit()
    filehandle = open(f,errors='ignore')
    # read a single line
    file = (filehandle.read().replace('\n',' '))
    data_dict[count] = file
    count = count + 1
    data.append(file)
# close the pointer to that file
filehandle.close()


# In[110]:


len(data_dict.keys())


# ## Merging operations

# In[183]:


def intersect(a,b):
    c = []
    compare = 0
    i = 0
    j = 0
    while i < len(a) and j <len(b):
        if a[i] == b[j]:
            compare = compare +1
            c.append(a[i])
            i = i+1
            j = j+1
        elif a[i]<b[j]:
            compare = compare +1
            i = i+1
        else:
            compare = compare +1
            j = j+1
    return c,compare


# In[186]:


def union(a,b):
    c = []
    compare = 0
    i=0
    j=0
    while i < len(a) and j <len(b):
        if a[i] == b[j]:
            compare = compare +1
            c.append(a[i])
            i = i+1
            j = j+1
        elif a[i]<b[j]:
            compare = compare +1
            c.append(a[i])
            c.append(b[j])
            i = i+1
            j=j+1
        else:
            compare = compare +1
            c.append(b[j])
            c.append(a[i])
            i=i+1
            j = j+1
    return c,compare


# In[325]:


def andnot(a,b):
    c = []
    compare = 0
    i=0
    j=0
    while i < len(a):
#         print(j)
        if j==len(b):
            c.extend(a[i:])
            break
        if a[i] == b[j]:
            compare = compare +1
            i = i+1
            j = j+1
        elif a[i]<b[j]:
            compare = compare +1
            c.append(a[i])
            i = i+1
        elif a[i]>b[j]:
            compare = compare +1
            j = j+1
#         else:
#             compare = compare +1
#             c.extend(a[i:])
#             break
    return c,compare


# In[238]:


def ornot(a,b,d):
    c = []
    temp = []
    compare = 0
    i=0
    j=0
    k = 0
    l = 0
    c.extend(d)
    temp,compare = intersect(a,b)
    temp2 = list(set(b) - set(temp))
    while l<len(temp2):
        c.remove(temp2[l])
        l = l+1
    return c,compare


# # basic text processing for data

# ## Lowering case

# In[141]:


def lower_query(query):
    query = query.lower()
    return query        


# In[111]:


def lower(dataset):
    for i in dataset.keys():
        dataset[i] = dataset[i].lower()
    return dataset
        


# ## Removing punctuation

# In[417]:


def remove_punct(word):
    punctuations = '''!()-[]{};:'"\,<>./?@#$=+%^&*_~'''
    no_punct = ""
    for char in word:
        if char not in punctuations:
            no_punct = no_punct + char
        elif char in punctuations:
            no_punct = no_punct + ''
    return no_punct


# In[112]:


def rempunct(dataset):
    punctuations = '''!()-[]{};:'"\,<>./?@#$=+%^&*_~'''
    for i in dataset.keys():
        no_punct = ""
        for char in dataset[i]:
            if char not in punctuations:
                no_punct = no_punct + char
            elif char in punctuations:
                no_punct = no_punct + ' '
        dataset[i] = no_punct

    return dataset


# ## Remove Stop Words

# In[404]:


def rem_stop_words(word):
    stop = stopwords.words('english')
    if word in stop:
        word = ''
    return word


# In[113]:


def remstopwords(dataset):
    stop = stopwords.words('english')
    for i in dataset.keys():
        dataset_list = []
        non_stop_list = []
        dataset_list = dataset[i].split()
        for j in dataset_list:
            if j not in stop:
                non_stop_list.append(j)
        dataset[i] = " ".join(non_stop_list)
    return dataset


# ## Stemming

# In[142]:


def stemming_query(query):
    st = PorterStemmer()
    stemm_list = []
    dataset_list = []
    query_list = query.split()
    for j in query_list:
        stemm_list.append(st.stem(j))
        query = " ".join(stemm_list)
    return query


# In[114]:


def stemming(dataset):
    st = PorterStemmer()
    for i in dataset.keys():
        stemm_list = []
        dataset_list = []
        dataset_list = dataset[i].split()
        for j in dataset_list:
            stemm_list.append(st.stem(j))
        dataset[i] = " ".join(stemm_list)
    return dataset


# ## Tokenization

# In[143]:


def tokenization_query(query):
    query_list = []
    query_list = query.split()
    return query_list


# In[115]:


def tokenization(dataset):
    token_list = []
    for i in dataset.keys():
        dataset_list = []
        dataset_list = dataset[i].split()
        token_list.append(dataset_list)
    flatten = lambda token_list: [item for sublist in token_list for item in sublist]
    return flatten(token_list)


# In[116]:


data_dict = lower(data_dict)


# In[117]:


data_dict = rempunct(data_dict)


# In[118]:


data_dict = remstopwords(data_dict)


# In[119]:


data_dict = stemming(data_dict)


# In[120]:


token_list = tokenization(data_dict)


# In[121]:


len(token_list)


# ## index terms for the inverted index

# In[122]:


index_terms = list(set(token_list))


# In[123]:


len(index_terms)


# ## inverted index

# In[127]:


inverted_index = {} # dictionary contains the same keys as term_freq dict and the posting list. 
term_freq = {} # dictionary contains term and length of the postings for that term


# In[128]:


#initializing inverted_index and term_freq
count = 0
for i in index_terms:
    postings = []
    if count%500==0:
        print(count)
    for j in data_dict.keys():
        if i in data_dict[j]:
#             print(i,j)
            postings.append(j)
            inverted_index[i]=postings
#             print(inverted_index[i])
    term_freq[i] = len(postings)
    count = count + 1


# In[129]:


np.save("/media/rohit/New Volume/codes/IR/Assignment1/inverted_index.npy",inverted_index,allow_pickle=True)
np.save("/media/rohit/New Volume/codes/IR/Assignment1/term_freq.npy",term_freq,allow_pickle=True)


# In[125]:


data_dict[0]


# In[104]:


inverted_index.keys()


# In[105]:


print(term_freq['brunt'])


# In[169]:


print(inverted_index['ux'])


# ## input a query

# In[147]:


query = input("Enter the query")


# In[148]:


query


# ## Process the query

# In[149]:


processed_query_list = tokenization_query(stemming_query(lower_query(query)))


# In[150]:


processed_query_list


# In[185]:


intersect([10,11,12],[10,12,13])


# In[199]:


union([10,11,12],[10,12,13])


# In[235]:


andnot([10,11,12,14,15,18,20,21],[10,12,13,14,15,16])


# In[243]:


ornot([10,11,12,15],[10,13,14],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])


# In[391]:


def query_output(processed_query_list):
    expr = processed_query_list
    stack = list()
    index = ""
    uni_set=[]
    for i in data_dict.keys():
        uni_set.append(i)
    while len(expr) > 0:
        c = expr.pop(0)
        print(c)
        if c not in ['and','or','not']:
            print("1")
            index=c
            stack.append(index)
            print(stack)
            index = ""
            if len(stack) ==3:
                print("4")
                check = stack.pop()
                print(stack)
                if check == 'not':
                    print("5")
                    stack.push(check)
                    print(stack)
                    continue
                elif check !='not':
                    print("6")
                    op = stack.pop()
                    print(stack)
                    index1 = stack.pop()
                    print(stack)
                    if op == "and":
                        print("7")
                        if type(index1) != tuple:
                            stack.append(intersect(inverted_index[index1],inverted_index[check]))
                            print(stack)
                        else:
                            stack.append(intersect(index1[0],inverted_index[check]))
                            print(stack)
                    elif op == "or":
                        print("8")
                        if type(index1) != tuple:
                            stack.append(union(inverted_index[index1],inverted_index[check]))
                            print(stack)
                        else:
                            stack.append(union(index1[0],inverted_index[check]))
                            print(stack)
            elif len(stack) == 4:
                print("9")
                index2 = stack.pop()
                print(stack)
                op2 = stack.pop()
                print(stack)
                op1 = stack.pop()
                print(stack)
                index1 = stack.pop()
                print(stack)
                
                if op1 == "and" and op2 == "not":
                    print("10")
                    if type(index1) != tuple:
                        stack.append(andnot(inverted_index[index1],inverted_index[index2]))
                        print(stack)
                    else:
                        stack.append(andnot(index1[0],inverted_index[index2]))
                        print(stack)
                elif op1 == "or" and op2 == "not":
                    print("11")
                    if type(index1) != tuple:
                        stack.append(ornot(inverted_index[index1],inverted_index[index2],uni_set))
                        print(stack)
                    else:
                        stack.append(ornot(index1[0],inverted_index[index2],uni_set))
                        print(stack)
        else:
            if c in ['and','or','not']:
                print("2")
                stack.append(c)
                print(stack)
            
                
    return stack.pop()

expr =processed_query_list
print(expr)
answer=query_output(expr)


# # The above function gives output for the first question 
# ## It returns a tuple in which the first element is a list that contains the documents. And the second element is the number of arguments

# In[388]:


query = input("Enter the query")
processed_query_list = tokenization_query(stemming_query(lower_query(query)))


# In[389]:


expr = processed_query_list


# In[390]:


expr


# In[387]:


uni_set=[]
for i in data_dict.keys():
        uni_set.append(i)


# # Question 2

# In[395]:


path = r'/media/rohit/New Volume/codes/IR/20_newsgroups/subset'
files = []
print(path)
for r, d, f in os.walk(path):
    for file in f:        
        files.append(os.path.join(r, file))
print(len(files))


# In[455]:


positional_index = {}
position = []
for docid,f in enumerate(files):
#     print(f)
#     sys.exit()
    with open(f,errors='ignore') as file:
        index = 0
        for line in file:
            for word in line.split():
                term=stemming_query(rem_stop_words(remove_punct(lower_query(word))))
                if term not in positional_index.keys():
                    positional_index[term] =  {}
                if docid not in positional_index[term].keys(): 
                    positional_index[term][docid] = []    
                positional_index[term][docid].append(index)
                index = index + 1


# In[490]:


a = list(positional_index['cmu'].keys())


# In[491]:


a


# In[466]:


a


# In[ ]:





# In[539]:


query = input("Enter the query")
processed_query_list = tokenization_query(stemming_query(lower_query(query)))


# In[540]:


processed_query_list


# In[541]:


def q2_output(processed_query_list,positional_index):
    expr = processed_query_list
    length = len(expr)
    print(length)
    i = 0
    l = []
    final_list = []
    compare = 0
    while i<length:
        if i>0:
            l,temp = intersect(l,list(positional_index[expr[i]].keys()))
            compare = compare + temp
            i = i+1
        else:
            l = list(positional_index[expr[i]].keys())
            i = i+1
    i=0
    j=0
    k=0
    while i<len(l):
        print(i,j,k)
        if list(positional_index[expr[j]][l[i]])[k] +1 in list(positional_index[expr[j+1]][l[i]]):
            j = j + 1
            k=0
        else:
            k = k+1
        if k == len(positional_index[expr[j]][l[i]]):
            i = i + 1
            k=0
        print(i,j,k)
#         print(positional_index[expr[j]][l[i]])
        if j == length - 1:
            final_list.append(l[i])
            print(final_list)
            i = i + 1
            j=0
            k=0
    return final_list,compare


# In[542]:


q2_output(processed_query_list,positional_index)


# # The above function gives output for the Second question 
# ## It returns a tuple in which the first element is a list that contains the documents. And the second element is the number of arguments

# In[ ]:





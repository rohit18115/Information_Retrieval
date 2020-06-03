READ ME:

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



# # The return_top_k() function gives output for the top k docs based on jaccard coefficient.
# ## It returns a list of top k docs and the dictionary itself that contains the jc score according to each document.

# # The count_top_k() function gives output for the top k docs based on raw count tf-idf.
# ## It returns a list of top k docs and the dictionary itself that contains the tf-idf score according to each document.

# # The count_top_k() function gives output for the top k docs based on normalized tf-idf.
# ## It returns a list of top k docs and the dictionary itself that contains the tf-idf score according to each document.

# # The count_top_k() function gives output for the top k docs based on log_normalized tf-idf.
# ## It returns a list of top k docs and the dictionary itself that contains the tf-idf score according to each document.

# # The edit_distance() function gives output for the Second question 
# ## It returns a list in which the most similar words in comparision to the wrong words are returned.



The tasks of Assignment 2 were mainly focused on retrieving top K documents using different measures like: Jaccard coefficient, tf-idf and cosine similarity.
Preprocessing:
    • Basic Text Pre-processing of text data
        ◦ Lower casing
        ◦ Punctuation removal
        ◦ Tokenization
        ◦ Stemming

Plan of execution:
    • Things to calculate:
        ◦ Jaccard coefficient
        ◦ Term frequency
            ▪ Raw count
            ▪ Standard tf
            ▪ Log normalized tf
        ◦ Tf-idf
            ▪ Based on raw count
            ▪ Based on standard tf
            ▪ Based on log-normalized tf
        ◦ Cosine similarity based on tf-idf vectors


Explanation :

    • After all the preprocessing of the documents a data dictionary was made in order to systematically store the set of unique terms for each of the documents.
    • The process of calculating the jaccard coefficient was pretty easy and short. Which consisted of only minor set manipulations like intersection and union between word sets of documents and query. And then finally returning top k documents from the new coefficients dictionary that was made for storing jaccard coefficient for each document.
    •  Comparatively the process of calculating variations of tf-idf was very repetitive and tedious. Which consisted of storing variations of tfs in different dictionaries. And accessing them respectively for calculating variations of tf-idfs for different documents which was calculated every time a query was given. And proper priority was given to the Title and body of every document while calculating the tf-idfs.
    • Cosine similarity was calculated between the query and documents. And top k documents were returned for the particular query.

     
Some Assumptions:
    • For calculating cosine similarity, instead of forming vectors of tf-idf for each term in a documents i stored the tf-idf values of each term in the document in the form of a dictionary and then accessing tf-idf values for the required term for performing operations.
    •  Numerical values have been converted into words.
    • Stop words have been removed.


Result for question1:
Query: zombies
------------------------------Jaccard Coefficient--------------------------------
['gloves.txt', 'quarter.c11', 'toilet.s', 'wolfcran.txt', 'foxnstrk.txt']
------------------------------tf-idf with raw count as term frequency------------------------------
['zombies.txt', 'keepmodu.txt', 'sick-kid.txt', 'quest', 'social.vikings']
------------------------------tf-idf with normal term frequency------------------------------------
['zombies.txt', 'social.vikings', 'socialvikings.txt', 'imonly17.txt', '14.lws']
-----------------------tf-idf with normal log normalized term frequency----------------------------
['zombies.txt', 'keepmodu.txt', 'sick-kid.txt', 'quest', 'social.vikings']
------------------------------------cosine similarity-----------------------------------------------
['zombies.txt', 'socialvikings.txt', 'social.vikings', 'keepmodu.txt', 'imonly17.txt']

Jaccard coefficient is an incompetent method in comparison to other similarity methods like tf-idf and cosine similarity. As seen above it is highly influenced by the number of words in the document. And if we talk about other variations of tf-idf based methods cosine similarity stands at the top. Infact log normalized tf-idf is some what comparable to it. As both are length normalized. 

Edit distance:
These are the results returned by edit distance:

Query_list: ['helo', 'ad', 'sory']

['', 'he', 'hello', 'helot', 'ho']

['', 'soy', 'or', 'so', 'sorry']

The tasks of Assignment 3 were mainly focused on creating a champion list ordered by Static quality score. This can be done by first creating the high list and low list. And on the basis of that return top k documents for the query.
Preprocessing:
    • Basic Text Pre-processing of text data
        ◦ Lower casing
        ◦ Punctuation removal
        ◦ Tokenization
        ◦ Stemming
        ◦ num3words
Plan of execution for question 1:
    • Preprocess the huge dataset.
    • Create a data_dict for the whole dataset containing docid as keys and list of terms as values.
    • Create term count and document frequency dictionary.
    • Create term frequency dictionary using term count dictionary with term as keys and docid and tf values dictionary as values.
    • Sort the term frequency dictionary on the basis of tf values.
    • Divide the sorted term frequency dictionary into high list and low list.
    • The value of r(position at which we will split the sorted term frequency dictionary) is different for every term. And can be decided by taking the mean or median of tf values of each term.
    • Then calculate the value of normalized static quality score for every document based on the number of reviews given to each document.
    • Then we’ll search the high list for only those documents that will contain all the query terms and low list will contain results only for documents that contain atleast one term.
    • Then we finally return the top k documents based on the net score.
Plan of execution for question 2 :
    • We read the whole list of urls for all the queries.
    • Then filter out all the urls which are for the query id 4
    • We observed that max DCG is same as sorting the ideal relevance order that is if we retrieve the urls with high relevance before than the lower relevance documents. So we sort the urls with decreasing order of relevance score.
    • Then finally find the number of permutations for all relevance values and multiply it.
    • In part 2 of the second question we calculate the dcg at k and then divide it by the ideal dcg to get the nDCG value at k. Similarly for the whole dataset.
    • In part 3 of second question sort the relevance scores on the basis of decreasing value of tf-idf and finally calculate the recall and precision in similar order and plot the precision and recall values to get the PR curve.
Some Assumptions:
    • Used tf-idf in net-score instead of calculating cosine similarity.
    • Done a comparison on r values on the basis of mean and median. 
    • Numerical values have been converted into words.
    • Stop words have been removed.


Result for question1:


Difference in the length of high list and low list on the basis of mean and median:




MEAN:
xref   2496
xref   3333
cantaloup   6311
cantaloup   10019
srv   6116
srv   10214
cs   5627
cs   10703
cmu   6039
cmu   10291
edu   6300
edu   10030
alt   1166
alt   2126
atheism   575
atheism   781

MEDIAN:
xref   2917
xref   2912
cantaloup   8192
cantaloup   8138
srv   8166
srv   8164
cs   8179
cs   8151
cmu   8165
cmu   8165
edu   8170
edu   8160
alt   1651
alt   1641
atheism   682
atheism   674


As we can see above that the length of high list is less in case of mean as compared to median of the tf values. This means that mean works well in selecting the r value.
Query: I downloaded the whole bunch last week and have been morphing  away the afternoons since.  The programmes are all a bit buggy and definitely not-ready-to-spread-to-the-masses, but they are very well written.

Documents from high list in case of mean:  ['76103', '37914', '9891', '54123', '54134', '9962', '9571', '53838', '39063', '38646', '38425', '76828', '38824', '76868', '9919', '54219', '54164', '54131', '76509', '54007']
length of high docs returned incase of mean:
20
Documents from low list in case of mean:  []
length of the low docs in case of mean:
0
Documents from high list in case of median:  ['37914', '76103', '9891', '54123', '54134', '9962', '9571', '53838', '39063', '9930', '38646', '38425', '76828', '38824', '76868', '9919', '54164', '53568', '38762', '9964']
length of the high docs in case of median:
20
Documents from low list in case of median:  []
length of the low docs in case of median:
0
 

Question 2: 



Question 3: 
Given in separate image file .

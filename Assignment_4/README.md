## The tasks of Assignment 4 were mainly focused on improving the MAP and PR curve using relevance feedback using rocchio algorithm.
## Preprocessing:
* Basic Text Pre-processing of text data
* Lower casing
Punctuation removal
* Tokenization
* Stemming
* num3words
* Plan of execution for question 1:
* Preprocess the huge dataset.
* Create a data_dict for the whole dataset containing docid as keys and list of terms as values.
* Create term count and document frequency dictionary.
* Create term frequency dictionary using term count dictionary with term as keys and docid and tf values dictionary as values.
* Create a tf-idf dictionary for both query and all other documents.
* Vectorize both the dictionaries.
* Run rocchio iterations after taking relevant feedback from user.
*  Create PR curve and TSNE plot after every iteration.
* Calculate AP after every iteration and for every query
* Calculate MAP for every query.
* Some Assumptions:
*Used tf-idf in net-score instead of calculating cosine similarity.
*Numerical values have been converted into words.
*Stop words have been removed.





Result:


Query 1:
Iteration 1:

![PR curve](/Assignment_4/images/PR_curve1.png)

![Scatter Plot](/Assignment_4/images/scatter1.png)



Iteration 2:

![PR curve](/Assignment_4/images/PR_curve2.png)

![Scatter Plot](/Assignment_4/images/scatter2.png)


Iteration 3:

![PR curve](/Assignment_4/images/PR_curve3.png)

![Scatter Plot](/Assignment_4/images/scatter3.png)


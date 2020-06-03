The tasks of Assignment 5 were mainly focused on feature selection using tf-idf scores and mutual information and then performing text classification using Naive bayes and KNN algorithms.
Preprocessing:
    • Basic Text Pre-processing of text data
        ◦ Lower casing
        ◦ Punctuation removal
        ◦ Tokenization
        ◦ Stemming
        ◦ num2words
Plan of execution:
    • Preprocess the huge dataset.
    • Create a data_dict for the whole dataset containing docid as keys and list of terms as values.
    • Create term count and document frequency dictionary.
    • Create a term frequency dictionary using term count dictionary with term as keys and docid and tf values dictionary as values.
    • Create a tf-idf dictionary for both query and all other documents.
    • Create dataframe of the dataset with 3 columns, first column containing the doc words, second column consisting of the labels, third column containing the doc_ids
    • Perform feature selection based on the option chosen by the user.(0 for mutual info and 1 for tf-idf) 
    • In case of KNN
        ◦ Vectorize the train and test sets
        ◦ And then finally use those to perform text classification using KNN
    • In case of NB:
        ◦ Calculate the class wise statistics like class wise word frequency, class wise word count etc. 
        ◦ Then calculate the word probabilities which will be further used to classify the text documents into one of the five classes .
Some Assumptions:
    • Numerical values have been converted into words.
    • Stop words have been removed.
Result Analysis:
As observed from the result below for both models the MI feature selection works the best. And in general the Naive bayes algo works better than KNN
And the value of k=5 works best for the KNN. 













Naive Bayes: Format: Confusion matrix and ROC curve for each class
Test split: 30
Percentage of feature selected: 60
Feature selection method: TF-idf








Naive Bayes: Format: Confusion matrix and ROC curve for each class
Test split: 50
Percentage of feature selected: 60
Feature Selection: TF-IDF








Naive Bayes: Format: Confusion matrix and ROC curve for each class
Test split: 20
Percentage of feature selected: 60
Feature Selection: TF-IDF








Naive Bayes: Format: Confusion matrix and ROC curve for each class
Test split: 30
Percentage of feature selected: 10
Feature Selection: MI







Naive Bayes: Format: Confusion matrix and ROC curve for each class
Test split: 50
Percentage of feature selected: 10
Feature Selection: MI







Naive Bayes: Format: Confusion matrix and ROC curve for each class
Test split: 20
Percentage of feature selected: 10
Feature Selection: MI







KNN: Format: Confusion matrix and ROC curve for each class
Test split: 50
Percentage of feature selection: 10, k = 5
Feature selection : TF-IDF









KNN: Format: Confusion matrix and ROC curve for each class
Test split: 30
Percentage of feature selection: 10, k = 5
Feture_selection:TF-IDF









KNN: Format: Confusion matrix and ROC curve for each class
Test split: 20
Percentage of feature selection: 10, k = 5
Feture_selection:TF-IDF









KNN: Format: Confusion matrix and ROC curve for each class
Test split: 50
Percentage of feature selection: 10, k = 5
Feture_selection:MI








KNN: Format: Confusion matrix and ROC curve for each class
Test split: 30
Percentage of feature selection: 10, k = 5
Feture_selection:MI








KNN: Format: Confusion matrix and ROC curve for each class
Test split: 20
Percentage of feature selection: 10, k = 5
Feture_selection:MI












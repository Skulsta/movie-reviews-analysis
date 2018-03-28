Movie Reviews Analysis
----------------------
Using Doc2Vec and Logistic Regression to predict whether a movie review is positive or negative.

The scikit-learn library is used for training and testing the classifier.

Gensim's Doc2Vec model is used for building and training word vectors.

The code includes 4 review files, all of them in lowercase and without
punctuation. 2 train sets which include 12.5k movie reviews each and 2
test sets which include 5k reviews. All files separate reviews by newlines.

A trained Doc2Vec model is included, which has been trained over the
course of 10 epochs. The code for training the model is included in the
main file, but commented out to decrease compilation time.

Running the main file will output the classification accuracy level.
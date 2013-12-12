import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold

def make_xy(data,vectorizer=None):
    if vectorizer==None:
        vectorizer = CountVectorizer(min_df=0)
    text = [words for i,words in data.text.iteritems()]
    vectorizer.fit(text)
    x = vectorizer.transform(text)
    #x = x.toarray()
    x = sparse.csc_matrix(x)
    performance = data.performance
    perfarray = np.array(performance)
    return x,perfarray





def log_likelihood(clf,x,y):
    mat = clf.predict_log_proba(x)
    return sum(mat[y==1,1])+sum(mat[y==0,0])




def cv_score(clf, x, y, score_func):
    """
    Uses 5-fold cross validation to estimate a score of a classifier
    
    Inputs
    ------
    clf : Classifier object
    x : Input feature vector
    y : Input class labels
    score_func : Function like log_likelihood, that takes (clf, x, y) as input,
                 and returns a score
                 
    Returns
    -------
    The average score obtained by randomly splitting (x, y) into training and 
    test sets, fitting on the training set, and evaluating score_func on the test set
    
    Examples
    cv_score(clf, x, y, log_likelihood)
    """
    result = 0
    nfold = 5
    for train, test in KFold(y.size, nfold): # split data into train/test groups, 5 times
        clf.fit(x[train], y[train]) # fit
        result += score_func(clf, x[test], y[test]) # evaluate score function on held-out data
    return result / nfold # average

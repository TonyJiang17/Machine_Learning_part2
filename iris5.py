#
# read iris data
#

import numpy as np            
import pandas as pd

from sklearn import tree      # for decision trees
from sklearn import ensemble  # for random forests

try: # different imports for different versions of scikit-learn
    from sklearn.model_selection import cross_val_score   # simpler cv this week
except ImportError:
    try:
        from sklearn.cross_validation import cross_val_score
    except:
        print("No cross_val_score!")
        

#
# Here are the correct answers to the csv's "unknown" flowers
#
answers = [ 'virginica',   # index 0 (row 1 in the csv) 
            'virginica',   # index 1 (row 2 in the csv) 
            'versicolor',  # and so on...
            'versicolor',
            'setosa',
            'setosa',
            'virginica',
            'versicolor',
            'setosa']



print("+++ Start of pandas' datahandling +++\n")

# df is a "dataframe":
df = pd.read_csv('iris5.csv', header=0)   # read the file w/header row #0

# Now, let's take a look at a bit of the dataframe, df:
df.head()                                 # first five lines
df.info()                                 # column details

# One important feature is the conversion from string to numeric datatypes!
# For _input_ features, numpy and scikit-learn need numeric datatypes
def transform(s):
    """ from string to number
          setosa -> 0
          versicolor -> 1
          virginica -> 2
    """
    d = { 'unknown':-1, 'setosa':0, 'versicolor':1, 'virginica':2 }
    return d[s]
    
# 
# this applies the function transform to a whole column
#
#df['irisname'] = df['irisname'].map(transform)  # apply the function to the column

print("\n+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++\n")

print("     +++++ Decision Trees +++++\n\n")

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_all = df.iloc[:,0:4].values        # iloc == "integer locations" of rows/cols
y_all = df[ 'irisname' ].values      # individually addressable columns (by name)

X_labeled = X_all[9:,:]  # make the 10 into 0 to keep all of the data
y_labeled = y_all[9:]    # same for this line

#
# scramble data
# 
indices = np.random.permutation(len(X_labeled))  
X_data_full = X_labeled[indices]
y_data_full = y_labeled[indices]

X_train = X_data_full
y_train = y_data_full

#
# some labels to make the graphical trees more readable...
#
print("Some labels for the graphical tree:")
feature_names = ['sepallen', 'sepalwid', 'petallen', 'petalwid']
target_names = ['setosa', 'versicolor', 'virginica']

#
# show the creation of three tree files (at three max_depths)
#
for max_depth in [1,2,3]:
    # the DT classifier
    dtree = tree.DecisionTreeClassifier(max_depth=max_depth)

    # train it (build the tree)
    dtree = dtree.fit(X_train, y_train) 

    # write out the dtree to tree.dot (or another filename of your choosing...)
    filename = 'tree' + str(max_depth) + '.dot'
    tree.export_graphviz(dtree, out_file=filename,   # the filename constructed above...!
                            feature_names=feature_names,  filled=True, 
                            rotate=False, # LR vs UD
                            class_names=target_names, 
                            leaves_parallel=True )  # lots of options!
    #
    # Visualize tree graphs at www.webgraphviz.com
    #
    print("Wrote the file", filename)  
    #


#
# cross-validation and scoring to determine parameter: max_depth
# 
for max_depth in range(1,12):
    # create our classifier
    dtree = tree.DecisionTreeClassifier(max_depth=max_depth)
    #
    # cross-validate to tune our model (this week, all-at-once)
    #
    scores = cross_val_score(dtree, X_train, y_train, cv=5)
    average_cv_score = scores.mean()
    print("For depth=", max_depth, "average CV score = ", average_cv_score)  
    # print("      Scores:", scores)

# import sys
# print("bye!")
# sys.exit(0)

MAX_DEPTH = 3   # choose a MAX_DEPTH based on cross-validation... 
print("\nChoosing MAX_DEPTH =", MAX_DEPTH, "\n")

#
# now, train the model with ALL of the training data...  and predict the unknown labels
#

X_unknown = X_all[0:9,0:4]              # the final testing data
X_train = X_all[9:,0:4]              # the training data

y_unknown = y_all[0:9]                  # the final testing outputs/labels (unknown)
y_train = y_all[9:]                  # the training outputs/labels (known)

# our decision-tree classifier...
dtree = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
dtree = dtree.fit(X_train, y_train) 

#
# and... Predict the unknown data labels
#
print("Decision-tree predictions:\n")
predicted_labels = dtree.predict(X_unknown)
answer_labels = answers

#
# formatted printing! (docs.python.org/3/library/string.html#formatstrings)
#
s = "{0:<11} | {1:<11}".format("Predicted","Answer")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)
# the table...
for p, a in zip( predicted_labels, answer_labels ):
    s = "{0:<11} | {1:<11}".format(p,a)
    print(s)

#
# feature importances!
#
print()
print("dtree.feature_importances_ are\n      ", dtree.feature_importances_) 
print("Order:", feature_names[0:4])


#
# Random Forests Next
# 

print("\n\n")
print("     +++++ Random Forests +++++\n\n")

#
# The data is already in good shape -- let's start from the original dataframe:
#
X_all = df.iloc[:,0:4].values        # iloc == "integer locations" of rows/cols
y_all = df[ 'irisname' ].values      # individually addressable columns (by name)

X_labeled = X_all[9:,:]  # just the input features, X, that HAVE output labels
y_labeled = y_all[9:]    # here are the output labels, y, for X_labeled

#
# Scramble data
# 
indices = np.random.permutation(len(X_labeled))  
X_data_full = X_labeled[indices]
y_data_full = y_labeled[indices]

X_train = X_data_full
y_train = y_data_full


#
# cross-validation to determine the Random Forest's parameters (max_depth and n_estimators)
#

highest_CV_score = 0
best_max_depth = 0
best_number_estimator = 0

for m_depth in range(1,20):
    for n_est in range(50,250,50):

        rforest = ensemble.RandomForestClassifier(max_depth=m_depth, n_estimators=n_est)

        # an example call to run 5x cross-validation on the labeled data
        scores = cross_val_score(rforest, X_train, y_train, cv=5)
        print("CV scores:", scores)
        print("CV scores' average:", scores.mean())

        average_cv_scores_RT = scores.mean()
        # comparing with highest CV score to determine whether this pair of depth and n_estimator is good or not:
        if (average_cv_scores_RT > highest_CV_score):
            highest_CV_score = average_cv_score_RT
            best_max_depth = m_depth
            best_number_estimator = n_est

print("The best pair of max_depth and n_estimators are: ", best_max_depth, "and", best_number_estimator)
print("\nThe CV score for that pair is = ", highest_CV_score)

#test using whole dataset
X_test = X_all[0:9,0:4]              # the final testing data
X_train = X_all[9:,0:4]              # the training data

y_test = y_all[0:9]                  # the final testing outputs/labels (unknown)
y_train = y_all[9:]                  # the training outputs/labels (known)

# these next lines is where the full training data is used for the model
MAX_DEPTH = best_max_depth
NUM_TREES = best_number_estimator
print()
print("Using MAX_DEPTH=", MAX_DEPTH, "and NUM_TREES=", NUM_TREES)
rforest = ensemble.RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=NUM_TREES)
rforest = rforest.fit(X_train, y_train) 

# Show results:
print("Random-forest predictions:\n")
predicted_labels = rforest.predict(X_test)
answer_labels = answers  

#
# formatted printing again (see above for reference link)
#
s = "{0:<11} | {1:<11}".format("Predicted","Answer")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)
# the table
for p, a in zip( predicted_labels, answer_labels ):
    s = "{0:<11} | {1:<11}".format(p,a)
    print(s)

#
# feature importances
#
print("\nrforest.feature_importances_ are\n      ", rforest.feature_importances_) 
print("Order:", feature_names[0:4])

# The individual trees are in  rforest.estimators_  [a list of decision trees!]

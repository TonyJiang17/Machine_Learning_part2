import numpy as np            
import pandas as pd

from sklearn import tree      
from sklearn import ensemble  

try: 
    from sklearn.model_selection import cross_val_score   
except ImportError:
    try:
        from sklearn.cross_validation import cross_val_score
    except:
        print("No cross_val_score!")


print("+++ Start of pandas' datahandling +++\n")

# df is a "dataframe":
df = pd.read_csv('Nasdaq.csv', header=0)   # read the file w/header row #0

#drop unnecessary columns
df = df.drop('Close', axis=1)
df = df.drop('Adj Close', axis = 1)
df = df.drop('Volume', axis=1)
df = df.drop('Date', axis=1)
df = df.drop('Open', axis=1)
df = df.drop('Low', axis=1)
df = df.drop('Market', axis=1)


# Now, let's take a look at a bit of the dataframe, df:
df.head()                                 
df.info()                                 

df = df.dropna() #drop all unfilled data

print("\n+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++\n")

print("     +++++ Decision Trees +++++\n\n")

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_all = df.drop('High', axis=1).values  # you can pick any value, I picked High 
y_all = df[ 'High' ].values 
total_sample = 1509 #total sample size
split = total_sample - 25 #split the dataset into testing and training (number of split is arbitrary)
X_labeled = X_all[:split,:]  # Marking where I want to start my training data 
y_labeled = y_all[:split]    

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
feature_names = ['Yesterday_Open', 'Yesterday_High', 'Yesterday_Low', 'Yesterday_Close', 'Yesterday_AdjClose','Yesterday_Volumne']
target_names = ['Today High']


#
# cross-validation and scoring to determine parameter: max_depth
# 
max_depth_DT = 0
max_CV_DT = 0
for max_depth in range(1,15): #looping through max_depth to find the optimal
    # create our classifier
    dtree = tree.DecisionTreeRegressor(max_depth=max_depth,random_state=0)

    scores = cross_val_score(dtree, X_train, y_train, cv=5)
    average_cv_score_DT = scores.mean()
    print("For depth=", max_depth, "average CV score = ", average_cv_score_DT)  
    if (max_CV_DT < average_cv_score_DT):
        max_CV_DT = average_cv_score_DT
        max_depth_DT = max_depth
print("The best max_depth for Decision Tree is: ", max_depth_DT)
print("The CV score for that max_depth is: ", max_CV_DT)

# import sys
# print("bye!")
# sys.exit(0)

MAX_DEPTH = max_depth_DT   # choose a MAX_DEPTH based on cross-validation... 
print("\nChoosing MAX_DEPTH =", MAX_DEPTH, "\n")

#
# now, train the model with ALL of the training data...  and predict the 'test' labels
#

X_unknown = X_all[split:,:]              # the final testing data
X_train = X_all[:split,:]              # the training data

y_unknown = y_all[split:]                  # the final testing outputs/labels (unknown)
y_train = y_all[:split]                  # the training outputs/labels (known)

# our decision-tree classifier
dtree = tree.DecisionTreeRegressor(max_depth=MAX_DEPTH,random_state=0)
dtree = dtree.fit(X_train, y_train) 

#
# Predict the test data labels
#
print("Decision-tree predictions:\n")
predicted_labels = dtree.predict(X_unknown)
answer_labels = y_unknown

#
# formatted printing
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
print("Order:", feature_names[0:])

##printing the dot file of the optimal decision tree with best max_depth
filename = 'StockDtree' + str(max_depth_DT) + '.dot'
tree.export_graphviz(dtree, out_file=filename,   # the filename constructed above...!
                        feature_names=feature_names,  filled=True, 
                        rotate=False, # LR vs UD
                        class_names=target_names, 
                        leaves_parallel=True )  # lots of options!

print("Wrote the file", filename)  


#
# Random Forests!
# 

print("\n\n")
print("     +++++ Random Forests +++++\n\n")


X_labeled = X_all[:split,:]  
y_labeled = y_all[:split]    

#
# scramble data
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

#looping through both max_depth and num_estimator to fine the optimal pair
for m_depth in range(1,12):
    for n_est in range(50,250,50): 

        rforest = ensemble.RandomForestRegressor(max_depth=m_depth, n_estimators=n_est,random_state=0)

        # an example call to run 5x cross-validation on the labeled data
        scores = cross_val_score(rforest, X_train, y_train, cv=5)
        print("CV scores:", scores)
        print("CV scores' average:", scores.mean())

        average_cv_scores_RT = scores.mean()
        # comparing with highest CV score to determine whether this pair of depth and n_estimator is good or not:
        if (average_cv_scores_RT > highest_CV_score):
            highest_CV_score = average_cv_scores_RT
            best_max_depth = m_depth
            best_number_estimator = n_est

print("The best pair of max_depth and n_estimators are: ", best_max_depth, "and", best_number_estimator)
print("\nThe CV score for that pair is = ", highest_CV_score)

#
# now, train the model with ALL of the training data and predict the labels of the test set
#

X_test = X_all[split:,:]              # the final testing data
X_train = X_all[:split,:]              # the training data

y_test = y_all[split:]                  # the final testing outputs/labels (unknown)
y_train = y_all[:split]                  # the training outputs/labels (known)

# these next lines is where the full training data is used for the model
MAX_DEPTH = best_max_depth#2
NUM_TREES = best_number_estimator#10
print()
print("Using MAX_DEPTH=", MAX_DEPTH, "and NUM_TREES=", NUM_TREES)
rforest = ensemble.RandomForestRegressor(max_depth=MAX_DEPTH, n_estimators=NUM_TREES,random_state=0)
rforest = rforest.fit(X_train, y_train) 

# here are some examples, printed out:
print("Random-forest predictions:\n")
predicted_labels = rforest.predict(X_test)
answer_labels = y_test  # note that we're "cheating" here!

#
# formatted printing again (see above for reference link)
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
# feature importances
#
print("\nrforest.feature_importances_ are\n      ", rforest.feature_importances_) 
print("Order:", feature_names[0:])

# The individual trees are in  rforest.estimators_  [a list of decision trees!]

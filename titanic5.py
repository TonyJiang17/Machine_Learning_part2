#
# titanic5: modeling the Titanic data with DTs and RFs
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
# The "answers" to the 30 unlabeled passengers:
#
answers = [0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,
            1,0,1,1,1,1,0,0,0,1,1,0,1,0]

#

print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('titanic5.csv', header=0)    # read the file w/header row #0
#
# drop columns here
#
df = df.drop('name', axis=1)  # axis = 1 means column
df = df.drop('home.dest', axis=1)  
df = df.drop('ticket', axis=1)
df = df.drop('cabin', axis = 1)

df.head()                                 # first five lines
df.info()                                 # column details

df = df.dropna() # drop unfilled data
# One important one is the conversion from string to numeric datatypes!
# You need to define a function, to help out...
def tr_mf(s):
    """ from string to number
    """
    d = { 'male':0, 'female':1 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column

def tr_sur(s):
    """ transforming the survived column from int to string
    """
    dic = {0:'Died', 1:'Survived', -1:'unsure'}
    return dic[s]
#didn't apply because it is easier to compare with the answer this way    
#df['survived'] = df['survived'].map(tr_sur) #apply to the dataset

def tr_emb(x):
    """transform the embarked column from string to int 
    """
    dic = {'C':0, 'Q':1, 'S':2}
    return dic[x]

df['embarked'] = df['embarked'].map(tr_emb)


print("\n+++ End of pandas +++\n")

#

print("+++ Start of numpy/scikit-learn +++\n")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays 
X_all = df.drop('survived', axis=1).values       
y_all = df[ 'survived' ].values      

 

print("     +++++ Decision Trees +++++\n\n")

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
#handling the arrays into testing and training sets
X_labeled = X_all[30:,:]  # set aside unlabeled data aside first 
y_labeled = y_all[30:]    

#
# we can scramble the data - but only the labeled data!
# 
indices = np.random.permutation(len(X_labeled))  # this scrambles the data each time
X_data_full = X_labeled[indices]
y_data_full = y_labeled[indices]

X_train = X_data_full
y_train = y_data_full

#
# some labels to make the graphical trees more readable...
#
print("Some labels for the graphical tree:")
feature_names = ['pclass', 'sex', 'age', 'sibsp','parch','fare','embarked']
target_names = ['Did Not Survive','survived']

#
# cross-validation and scoring to determine parameter: max_depth
# 
max_depth_DT = 0
max_CV_DT = 0
for max_depth in range(1,20): #looping through max_depth to find the optimal 
    # create our classifier
    dtree = tree.DecisionTreeClassifier(max_depth=max_depth)
    #
    # cross-validate to tune our model (this week, all-at-once)
    #
    scores = cross_val_score(dtree, X_train, y_train, cv=5)
    average_cv_score_DT = scores.mean()
    print("For depth=", max_depth, "average CV score = ", average_cv_score_DT)  
    # print("      Scores:", scores)
    if (max_CV_DT < average_cv_score_DT):
        max_CV_DT = average_cv_score_DT
        max_depth_DT = max_depth
print("The best max_depth for Decision Tree is: ", max_depth_DT)
print("The CV score for that max_depth is: ", max_CV_DT)

#
# show the creation of three tree files (at three max_depths)
#

# the DT classifier
dtree = tree.DecisionTreeClassifier(max_depth=max_depth_DT)

# train it (build the tree)
dtree = dtree.fit(X_train, y_train) 

# write out the dtree to tree.dot (or another filename of your choosing...)
filename = 'TitanicDtree' + str(max_depth_DT) + '.dot'
tree.export_graphviz(dtree, out_file=filename,   # the filename constructed above...!
                        feature_names=feature_names,  filled=True, 
                        rotate=False, # LR vs UD
                        class_names=target_names, 
                        leaves_parallel=True )  # lots of options!
    #
    # Visualize the resulting graphs (the trees) at www.webgraphviz.com
    #
print("Wrote the file", filename)  
#

MAX_DEPTH = max_depth_DT   # choose a MAX_DEPTH based on cross-validation... 
print("\nChoosing MAX_DEPTH =", MAX_DEPTH, "\n")

#
# now, train the model with ALL of the training data...  and predict the unknown labels
#

X_unknown = X_all[:30,:]              # the final testing data
X_train = X_all[30:,:]              # the training data

y_unknown = y_all[0:30]                  # the final testing outputs/labels (unknown)
y_train = y_all[30:]                  # the training outputs/labels (known)

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
print("Order:", feature_names[0:])

#Randomized Tree
print("\n\n")
print("     +++++ Random Forests +++++\n\n")

#
# The data is already in good shape -- let's start from the original dataframe:
#

X_labeled = X_all[30:,:]  # just the input features, X, that HAVE output labels
y_labeled = y_all[30:]    # here are the output labels, y, for X_labeled

#
# we can scramble the data - but only the labeled data!
# 
indices = np.random.permutation(len(X_labeled))  # this scrambles the data each time
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
# looping through both max_depth and num-estimators to find the optimal pair
for m_depth in range(1,20):
    for n_est in range(50,250,50):

        rforest = ensemble.RandomForestClassifier(max_depth=m_depth, n_estimators=n_est)

        # an example call to run 5x cross-validation on the labeled data
        scores = cross_val_score(rforest, X_train, y_train, cv=5)
        print("CV scores:", scores)
        print("CV scores' average:", scores.mean())
        # you'll want to take the average of these...
        average_cv_scores_RT = scores.mean()
        # comparing with highest CV score to determine whether this pair of depth and n_estimator is good or not:
        if (average_cv_scores_RT > highest_CV_score):
            highest_CV_score = average_cv_scores_RT
            best_max_depth = m_depth
            best_number_estimator = n_est

print("The best pair of max_depth and n_estimators are: ", best_max_depth, "and", best_number_estimator)
print("\nThe CV score for that pair is = ", highest_CV_score)

#
# now, train the model with ALL of the training data...  and predict the unknown labels
#

X_test = X_all[:30,:]              # the final testing data
X_train = X_all[30:,:]              # the training data

y_test = y_all[0:30]                  # the final testing outputs/labels (unknown)
y_train = y_all[30:]                  # the training outputs/labels (known)

# these next lines is where the full training data is used for the model
MAX_DEPTH = best_max_depth#2
NUM_TREES = best_number_estimator#10
print()
print("Using MAX_DEPTH=", MAX_DEPTH, "and NUM_TREES=", NUM_TREES)
rforest = ensemble.RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=NUM_TREES)
rforest = rforest.fit(X_train, y_train)

# here are some examples, printed out:
print("Random-forest predictions:\n")
predicted_labels = rforest.predict(X_test)
answer_labels = answers  # note that we're "cheating" here!

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

""" Comments and Results:
brief summary of the first 2 layers of the Decision Tree:
In the top block(first layer), it's saying that there are in total 1013 valid data samples(passengers), out of which 
601 did not survive and 412 survived. The decision this layer makes is whether a passenger is a male or female (sex <= 0.5),
if the passenger is a male, it is more likely he did not survive, while if the passenger is a female, it is more likely she survived.

In the second layer, there are two boxes (one for male passengers and one for female passengers). For the male passenger box, There are 
a total of 638 numbers of valid data samples, and out of those 507 did not survive and 131 survived. The decision for the male passenger box is 
whether the male passenger is older or younger than age 9.5 (age <= 9.5), and male passengers who are older have higher likelyhood to not have survived 
than those who are younger. 

On the other side of second lyaer is the female box. It shows that there are 375 females passengers, and out of which 94 did not survive and 281 survived. 
The decision for the male passenger box is whether the female passenger is in passenger class number less 2.5 or not (pclass <= 2.5). It shows that female
passenger in higher classes have greater likelyhood of survival than those with in lower class. 

Results:
1. The average cross-validated test-set accuracy for your best DT model: 
The best max_depth for Decision Tree is:  3                                                                 
The CV score for that max_depth is:  0.794633114427

Predicted   | Answer                                                                                                               
-------     | -------                                                                                                              
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 1                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
1           | 0                                                                                                                    
1           | 1                                                                                                                    
1           | 0                                                                                                                    
1           | 1                                                                                                                    
0           | 1                                                                                                                    
1           | 1                                                                                                                    
1           | 0                                                                                                                    
1           | 1                                                                                                                    
1           | 1                                                                                                                    
1           | 1                                                                                                                    
0           | 1                                                                                                                    
1           | 0                                                                                                                    
0           | 0                                                                                                                    
1           | 0                                                                                                                    
0           | 1                                                                                                                    
1           | 1                                                                                                                    
0           | 0                                                                                                                    
0           | 1                                                                                                                    
1           | 0  

dtree.feature_importances_ are                                                                                                     
       [ 0.21996659  0.6316222   0.05008153  0.05891991  0.          0.03940977                                                    
  0.        ]                                                                                                                      
Order: ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']


2. The average cross-validated test-set accuracy for your best RF model:
The best pair of max_depth and n_estimators are:  6 and 50                                                 
The CV score for that pair is =  0.808479868561

Predicted   | Answer                                                                                                               
-------     | -------                                                                                                              
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 1                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
0           | 0                                                                                                                    
1           | 0                                                                                                                    
1           | 1                                                                                                                    
1           | 0                                                                                                                    
1           | 1                                                                                                                    
0           | 1                                                                                                                    
1           | 1                                                                                                                    
1           | 0                                                                                                                    
1           | 1                                                                                                                    
1           | 1                                                                                                                    
1           | 1                                                                                                                    
0           | 1                                                                                                                    
1           | 0                                                                                                                    
0           | 0                                                                                                                    
1           | 0                                                                                                                    
0           | 1                                                                                                                    
1           | 1                                                                                                                    
0           | 0                                                                                                                    
0           | 1                                                                                                                    
1           | 0

rforest.feature_importances_ are                                                                                                   
       [ 0.1152408   0.46765541  0.12120033  0.04444372  0.03731233  0.17289811                                                    
  0.04124931]                                                                                                                      
Order: ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

"""
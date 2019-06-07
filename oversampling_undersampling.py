# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:28:52 2019

@author: Lalit.Chouhan
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
problem_data= pd.read_csv('.\\train_ps0mmDv_9yr6iGN\\train\\problem_data.csv')
train_submissions= pd.read_csv('.\\train_ps0mmDv_9yr6iGN\\train\\train_submissions.csv')
user_data=pd.read_csv('.\\train_ps0mmDv_9yr6iGN\\train\\user_data.csv')
#preparing training dataset
train_submissions= pd.merge(train_submissions,problem_data, on="problem_id")
train_submissions=pd.merge(train_submissions,user_data, on="user_id")
train_submissions['ID']=train_submissions['user_id']+'_'+train_submissions['problem_id']

y = train_submissions.loc[:,'attempts_range'].values
train_row=train_submissions.shape[0]
# preparing testing dataset
test_submissions=pd.read_csv('test_submissions_NeDLEvX.csv')
test_submissions= pd.merge(test_submissions,problem_data, on="problem_id")
test_submissions=pd.merge(test_submissions,user_data, on="user_id")
test_row=test_submissions.shape[0]



#------------------ Preparing Data set-----------------
test_submissions.drop(['user_id','problem_id'], axis=1, inplace=True)
train_submissions.drop(['user_id','problem_id','attempts_range'], axis=1, inplace=True)

Dataset=pd.concat([train_submissions, test_submissions], ignore_index=True)

Dataset_desc=Dataset.describe()
Dataset.info()
X=Dataset.iloc[:,1:].values
Dataset.isnull().sum()

# Data Preprocessing.............

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 6:7])
X[:, 6:7] = imputer.transform(X[:,6:7])


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1].astype(str))

X[:,4] = labelencoder_X.fit_transform(X[:,4].astype(str))
X[:,8] = labelencoder_X.fit_transform(X[:,8].astype(str))
X[:,12] = labelencoder_X.fit_transform(X[:,12].astype(str))


testX=X[train_row:]
X=X[:train_row]

#========================================================================================
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

from sklearn.utils import resample
# concatenate our training data back together
Xt = pd.concat([pd.DataFrame(X_train), pd.DataFrame( y_train)], axis=1)
unique_elements, counts_elements = np.unique(Xt.iloc[:,-1], return_counts=True)

# separate minority and majority classes
one= Xt[Xt.iloc[:,-1]==1]
two=Xt[Xt.iloc[:,-1]==2]
three=Xt[Xt.iloc[:,-1]==3]
four=Xt[Xt.iloc[:,-1]==4]
five=Xt[Xt.iloc[:,-1]==5]
six=Xt[Xt.iloc[:,-1]==6]

# upsample minority
one_upsampled = resample(one,
                          replace=True, # sample with replacement
                          n_samples=len(six), # match number in majority class
                          random_state=27) # reproducible results
two_upsampled = resample(two,
                          replace=True, # sample with replacement
                          n_samples=len(six), # match number in majority class
                          random_state=27) # reproducible results
three_upsampled = resample(three,
                          replace=True, # sample with replacement
                          n_samples=len(six), # match number in majority class
                          random_state=27) # reproducible results
four_upsampled = resample(four,
                          replace=True, # sample with replacement
                          n_samples=len(six), # match number in majority class
                          random_state=27) # reproducible results
five_upsampled = resample(five,
                          replace=True, # sample with replacement
                          n_samples=len(six), # match number in majority class
                          random_state=27) # reproducible results
# combine majority and upsampled minority
upsampled = pd.concat([one_upsampled,two_upsampled,three_upsampled,four_upsampled,five_upsampled,six],ignore_index=True)
2457*6

unique_elements, counts_elements = np.unique(upsampled.iloc[:,-1], return_counts=True)



y_train=upsampled.iloc[:,-1]
X_train=upsampled.iloc[:,:-1]



    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#================Naive Bayes==================
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import f1_score

f1=f1_score(y_test, y_pred, average='weighted')
f1



#============== Decision Tree ===============
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import f1_score
f1=f1_score(y_test, y_pred, average='weighted')
f1

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train.values, y_train.values)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import f1_score

f2=f1_score(y_test, y_pred, average='weighted')
f2

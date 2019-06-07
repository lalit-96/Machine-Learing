# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:31:25 2019

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

# preparing testing dataset
test_submissions=pd.read_csv('test_submissions_NeDLEvX.csv')
test_submissions= pd.merge(test_submissions,problem_data, on="problem_id")
test_submissions=pd.merge(test_submissions,user_data, on="user_id")

test_submissions.drop(['user_id','problem_id'], axis=1, inplace=True)

X = train_submissions.loc[:, ['level_type','points','submission_count','problem_solved','contribution','country','follower_count','max_rating','rating','rank']].values
y = train_submissions.loc[:,'attempts_range'].values


# Data Preprocessing.............

temp=train_submissions.describe()

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:2])
X[:, 1:2] = imputer.transform(X[:,1:2])



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0].astype(str))

X[:,5] = labelencoder_X.fit_transform(X[:,5].astype(str))
X[:,9] = labelencoder_X.fit_transform(X[:,9].astype(str))
df=pd.DataFrame(X)

temp2=df.describe()

#====================================================================================================
# Test submissions preprocessing

X2_test = test_submissions.loc[:, ['level_type','points','submission_count','problem_solved','contribution','country','follower_count','max_rating','rating','rank']].values

X2_test
# Data Preprocessing.............

temp3=test_submissions.describe()

# Taking care of missing data

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X2_test[:, 1:2])
X2_test[:, 1:2] = imputer.transform(X2_test[:,1:2])

# Encoding categorical data
# Encoding the Independent Variable
labelencoder_X2 = LabelEncoder()
X2_test[:,0] = labelencoder_X2.fit_transform(X2_test[:,0].astype(str))

X2_test[:,5] = labelencoder_X2.fit_transform(X2_test[:,5].astype(str))
X2_test[:,9] = labelencoder_X2.fit_transform(X2_test[:,9].astype(str))
df2=pd.DataFrame(X2_test)

#========================================================================================
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 0)

#=============== Random Forest ==========================
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 1)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import f1_score

f1=f1_score(y_test, y_pred, average='weighted')
f1


submission_file=test_submissions.loc[:,['ID','level_type']]
attempts_range = classifier.predict(X2_test)
len(attempts_range)
submission_file['attempts_range']=attempts_range

del submission_file['level_type']

submission_file.to_csv('test_predictions2.csv', encoding='utf-8', index=False)


#======================= That is it  form my side ========================
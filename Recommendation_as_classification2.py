# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:36:27 2019

@author: Lalit.Chouhan
"""

# Naive Bayes

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

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
X = X.astype(np.float64)
SL = 0.05
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
X_opt = X_opt.astype(np.float64)
X_Modeled = backwardElimination(X_opt, SL)


#------------------X-X-X------------------
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((X.shape[0],X.shape[1])).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
X_opt = X_opt.astype(np.float64)
X_Modeled = backwardElimination(X_opt, SL)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#========================================================================================
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

















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


#==============Logistic Regression ============

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import f1_score

f1=f1_score(y_test, y_pred, average='weighted')
f1


















#=================== Testing===================================

#Create a Dictionary of series
d1 = {'Name':pd.Series(['Tom','James','Ricky']),
   'Age':pd.Series([25,26,25]),
   'Rating':pd.Series([4.23,3.24,3.98]),
   'Country': pd.Series(['india','us','india']),

   }
d2 = {
      'Rating':pd.Series([2.56,3.20,4.6,3.8]),
      'Name':pd.Series(['Vin','Steve','Smith','Jack']),
   'Age':pd.Series([23,30,29,23]),

   'Country': pd.Series(['uk','uk','us','china']),
   'Nature': pd.Series(['average','good','worst','average'])
   }
#Create a DataFrame
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)


bigdata = pd.concat([df1, df2], ignore_index=True)
bigdata2 = pd.concat([df1, df2])

bigdata2.index



# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
df.loc[:,'Country_encoded'] = labelencoder_X.fit_transform(df.loc[:,'Country'])


df.loc[:,'Nature_encoded'] = labelencoder_X.fit_transform(df.loc[:,'Nature'])

#=========================================================================

# Data Preprocessing.............
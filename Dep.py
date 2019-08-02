import pandas as pd
import numpy as np

from sklearn import preprocessing,svm
from sklearn.model_selection import cross_validate,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC,SVC,SVR
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler,RobustScaler,OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import sys
import _pickle as cPickle

df = pd.read_csv('5_20_l_data.csv',encoding='utf-8')
df.head()

df.shape

df.dropna(inplace = True)
df.drop(df[df['Quantity'] == 0].index, inplace = True) 

X = np.array(df[['Quantity','Buyer ID','Region',
			 'Transport Type','Dispatch Mode']])

y = np.array(df[['Days']])
y = y.reshape((len(y),))

leMode = preprocessing.LabelEncoder()
leMode2 = preprocessing.LabelEncoder()
#leMode.fit(['seller_to_buyer','seller_to_warehouse','buyer_to_seller'])
#X[:,0] = leMode1.fit_transform(X[:,0])
X[:,1] = leMode.fit_transform(X[:,1])
X[:,2] = leMode.fit_transform(X[:,2])
X[:,3] = leMode.fit_transform(X[:,3])
X[:,4] = leMode.fit_transform(X[:,4])
#X[:,5] = leMode.fit_transform(X[:,5])
#X[:,6] = leMode.fit_transform(X[:,6])

leY = preprocessing.LabelEncoder()
y = leY.fit_transform(y)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X[:,0].reshape(-1, 1))
X[:,0] = scaler.transform(X[:,0].reshape(-1, 1)).reshape((X[:,0].shape[0]))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

## SVM Classifier

clf_svm = SVC(gamma ='auto' ,kernel='rbf', C=10,probability=True)
e_SVM = clf_svm.fit(X_train, y_train).score(X_test,y_test)
e_SVM_Train_Data = clf_svm.fit(X_train, y_train).score(X_train, y_train)
print("Accuracy on Train Data with --SVM Classifier-- is ", e_SVM_Train_Data*100,"%")
print("Accuracy on Test Data with --SVM Classifier-- is ", e_SVM*100,"%")

with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf_svm, fid)

with open('my_dumped_enc.pkl', 'wb') as fid1:
    cPickle.dump(leMode, fid1)


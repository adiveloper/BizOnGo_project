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

with open('my_dumped_classifier.pkl', 'rb') as fid:
    clf_svm = cPickle.load(fid)

with open('my_dumped_enc.pkl', 'rb') as fid1:
    leMode  = cPickle.load(fid1)
	
def predict(Xp):
	
	Xp[1] = leMode.fit_transform([Xp[1]])
	Xp[2] = leMode.fit_transform([Xp[2]])
	Xp[3] = leMode.fit_transform([Xp[3]])
	Xp[4] = leMode.fit_transform([Xp[4]])
	scaler = MinMaxScaler()
	scaler.fit(np.array(Xp[0]).reshape(-1, 1))
	Xp[0] = scaler.transform(np.array([Xp[0]]).reshape(-1, 1)).reshape((1,-1))
	Xp = np.array(Xp).reshape((-1,5))

	c = clf_svm.predict_proba(Xp).T
	return c*100
Xxc =[]
#Xxc = [2500,'West',12341,'Seller Transport','Seller to Buyer']
Xxc.append(int(input("Enter Quantity ")))
Xxc.append(input("Enter Region	 "))
Xxc.append(int(input("Enter Buyer Id ")))
Xxc.append(input('Transport Type '))
Xxc.append(input('Dispatch Mode	 '))
print(Xxc)
print(predict(Xxc))
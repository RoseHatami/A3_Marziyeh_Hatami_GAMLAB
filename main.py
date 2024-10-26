'''
APM:

Bande khedmateton nomre dadam ama lotf konid hatman bad az dore zaman bzarid
va takmil konid
moafagh bashid

'''


#-----------Import Libs----------------------


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#-----------Import Data----------------------
data = load_breast_cancer()

#-----------Step1 : X and Y ----------------------
x_RealData = data.data
y_RealData = data.target


#-------------LR MODEL-------------------------
x_Test,x_Train,y_Test,y_Train=train_test_split(x_RealData,y_RealData,test_size=0.25,random_state=42)

Log_model = LogisticRegression(max_iter=10000)

Log_model.fit(x_Train, y_Train)

y_LRPred=Log_model.predict(x_Test)

acc=accuracy_score(y_Test, y_LRPred)
print('accuracy: ', acc)


#bejaye estefade az train test az cross validation estefade konid mesle man :
#*******
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR





kf= KFold(n_splits=5 ,shuffle=True ,random_state=42) 
model=LogisticRegression(max_iter=10000)
myparams={} #logistic regression niaz b parameter nadare ama KNN, DT , RF , SVR niaz darand
gs=GridSearchCV(model,myparams,cv=kf,scoring='accuracy')
gs.fit(x,y)
gs.best_params_
gs.best_params_

#haminakr ro baraye done done model anjam bedid ama model va params ro taghir bdid
#******
KNN_model=KNeighborsClassifier(n_neighbors=5)
#KNN_kf= KFold(n_splits=5 ,shuffle=True ,random_state=42) 
KNN_params={'n_neighbors':[1,2,3,5,10],
            'metric':['minkowski' , 'manhattan','euclidean'] }
KNN_gs=GridSearchCV(KNN_model,KNN_params,cv=kf,scoring='accuracy')
KNN_gs.fit(x_RealData,y_RealData)
#y_KNNPred=KNN_gs.predict(x_Test)
#acc=accuracy_score(y_Test, y_KNNPred)
#print('accuracy: ', acc)
KNN_gs.best_params_
KNN_gs.best_params_


DT_model=DecisionTreeRegressor(random_state=42)
DT_params={ 'max_depth':[1,2,3,4,5,6,7,10]}
DT_gs=GridSearchCV(DT_model, DT_params,cv=kf,scoring='neg_mean_absolute_percentage_error')
DT_gs.fit(x,y)
DT_gs.best_params_
DT_gs.best_params_


svr_model=SVR()
svr_params={'kernel':['poly','rbf','linear'],
           'C':[0.001,0.01,1]}
svr_gs=GridSearchCV(svr_model, svr_params,cv=kf,scoring='accuracy')
svr_gs.fit(x,y)

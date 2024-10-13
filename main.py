import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
x_RealData = data.data
y_RealData = data.target


x_Test,x_Train,y_Test,y_Train=train_test_split(x_RealData,y_RealData,test_size=0.25,random_state=42)

Log_model = LogisticRegression(max_iter=10000)

Log_model.fit(x_Train, y_Train)

y_LRPred=Log_model.predict(x_Test)

acc=accuracy_score(y_Test, y_LRPred)
print('accuracy: ', acc)

plt.scatter(y_Test, y_LRPred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()








from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


KNN_model=KNeighborsClassifier(n_neighbors=5)

KNN_kf= KFold(n_splits=5 ,shuffle=True ,random_state=42) 

KNN_params={'n_neighbors':[1,2,3,5,10],
            'metric':['minkowski' , 'manhattan'] }


KNN_gs=GridSearchCV(KNN_model,KNN_params,cv=KNN_kf,scoring='accuracy')

KNN_gs.fit(x_RealData,y_RealData)

y_KNNPred=KNN_gs.predict(x_Test)

acc=accuracy_score(y_Test, y_KNNPred)
print('accuracy: ', acc)

plt.scatter(y_Test, y_KNNPred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('KNN Predictions vs True Values')
plt.show()

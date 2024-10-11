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


x_LRTest,x_LRTrain,y_LRTest,y_LRTrain=train_test_split(x_RealData,y,test_size=0.25,random_state=42)

Log_model = LogisticRegression(max_iter=10000)

Log_model.fit(x_LRTrain, y_LRTrain)

y_LRPred=Log_model.predict(x_LRTest)

plt.scatter(y_LRTest, y_LRPred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

accuracy = Log_model.score(x_RealData, y_RealData)
print(f"Accuracy: {accuracy}")

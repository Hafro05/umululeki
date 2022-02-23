#umeluleki

import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

#Importing the dataset 
dataset = pd.read_csv('lao.csv',sep= ',', header= None) 
data = dataset.iloc[:, :] 


#seperating the predicting column from the whole dataset 
X = data.iloc[:, 1:].values 
y = dataset.iloc[:, 0].values
  
#Encoding the predicting variable 
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y)

#Splitting the data into test and train dataset 
X_train, X_test, y_train, y_test = train_test_split( 
              X, y, test_size = 0.2, random_state = 0) 

print(X_train)

#Using the random forest classifier for the prediction 
classifier = RandomForestClassifier() 
classifier=classifier.fit(X_train,y_train) 
predicted=classifier.predict(X_test)
  
#printing the results 
print ('Confusion Matrix :') 
print(confusion_matrix(y_test, predicted)) 
print ('Accuracy Score :',accuracy_score(y_test, predicted)) 
print ('Report : ') 
print (classification_report(y_test, predicted)) 

pickl = {
    'classifier':classifier
}

pickle.dump( pickl, open( 'models' + ".p", "wb" ) )
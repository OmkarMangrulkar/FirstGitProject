import numpy as np
import pandas as pd

dataset = pd.read_csv(r"C:\Users\omkar mangrulkar\Desktop\10DPC\Day2\diabetes (1).csv")

print(dataset.info())
print(dataset.describe())

# Finding the total number of diabetec people
print(dataset['Outcome'].value_counts())

# 0 -> Non Diabetic -> 500
# 1 -> Diabetic -> 268

dataset.groupby('Outcome').mean()

# Sepreting the data into X and y
X = dataset.drop(columns='Outcome',axis=1)
y = dataset['Outcome']

# Data standardization
from sklearn.model_selection import train_test_split
Xtrain , Xtest , ytrain ,ytest = train_test_split(X,y,test_size=0.2,random_state=2,stratify=y)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit_transform(Xtrain, ytrain)

# Training the support vector machine classifier
from sklearn.svm import SVC
classifier=SVC(kernel='linear')
classifier.fit(Xtrain, ytrain)

# MODEL EVALUATION
# Accuracy score
# Finding accuracy score of training data
from sklearn.metrics import accuracy_score
Xtrain_prediction = classifier.predict(Xtrain)
training_data_accuracy = accuracy_score(ytrain, Xtrain_prediction)

print('Accuracy score of training data : ',training_data_accuracy) # 78%


# Finding accuracy score of test data

Xtest_prediction = classifier.predict(Xtest)
test_data_accuracy = accuracy_score(ytest, Xtest_prediction)

print('Accuracy score of test data : ',test_data_accuracy) # 77%

#--- Makeing Predictive System ---
input_data = (9,119,80,35,0,29,0.263,29)

 #Changing the input_data to numpy array
input_np_array = np.asarray(input_data)

 #Reshape the input_data
input_data_reshaped = input_np_array.reshape(1,-1)
 
 #Standradize the input_data
std_data = sc.transform(input_data_reshaped)
print(std_data)

 #Predict
predict_data = classifier.predict(std_data)
print(predict_data)

if (predict_data[0] == [0]):
    print('The person is not diabetic.')
else:
    print('Person is not Diabeties.')
 








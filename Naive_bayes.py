import pandas as pd
import numpy as np

data=pd.read_csv(r"C:\Users\omkar mangrulkar\Desktop\Machin Learning\AllDataSet\Social_Network_Ads.csv")

X = data.drop(['User ID','Gender','Purchased'],axis=1)
y = data['Purchased']

from sklearn.model_selection import train_test_split
X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=1)

from sklearn.preprocessing import Normalizer
norm = Normalizer()
X_train = norm.fit_transform(X_train)
X_test = norm.fit_transform(X_test)

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

# MultinomialNB() = 62%
# GaussianNB() = 65%
# BernoulliNB() = 62%

y_pred =classifier.predict(X_test)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)
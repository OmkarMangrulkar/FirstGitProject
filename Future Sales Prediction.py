import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\omkar mangrulkar\Downloads\archive (7)\Advertising.csv")

data.info()

#-- Let’s have a look at whether this dataset contains any null values or not:

data.isnull().sum()

data = data.drop('Unnamed: 0',axis=1)

#-- So this dataset doesn’t have any null values. Now let’s visualize the relationship 
#   between the amount spent on advertising on TV and units sold:

import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(data['Sales'],data['TV'])
plt.xlabel('Sales')
plt.ylabel('TV')
plt.show()

sns.scatterplot(data=data,x='Sales',y='Radio',)
plt.xlabel('Sales')
plt.ylabel('Radio')
plt.show()

plt.scatter(data['Sales'],data['Newspaper'])
plt.xlabel('Sales')
plt.ylabel('Newspaper')
plt.show()

correlation = data.corr()
print(correlation['Sales'].sort_values(ascending=False))

#-- Out of all the amount spent on advertising on various platforms,
#   I can see that the amount spent on advertising the product on TV results
#   in more sales of the product. Now let’s have a look at the correlation of
#   all the columns with the sales column:

x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
lr =LinearRegression()
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))

#features = [[TV, Radio, Newspaper]]
features = np.array([[230.1, 37.8, 69.2]])
print(lr.predict(features))
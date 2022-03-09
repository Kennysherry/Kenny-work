# Part 1
# Bring the setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Load the dataset
dataset = pd.read_csv('~/Downloads/Wine 1.csv')
dataset.columns = ['fixed acidity','volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free silful dioxide', 
                  'total sulfur dioxide', 'density', 'pH','sulphates', 'alcohol','quality']

# Statistical information:
dataset.info()
dataset.describe()
print(dataset.describe())

# Part 2
# Visualization:
# Scatterplot of pH to quality
target_column = "pH"
feature_name = "quality"
_ = sns.scatterplot(data=dataset, x=feature_name, y=target_column).set(title = 'Kaining analysis of pH to quality')

# Pairplot
sns.pairplot(data=dataset, diag_kind='hist', hue= 'quality').set(title = 'kaining pairwise relation study')
plt.show()

# Boxplot of alcohol
ax = sns.boxplot(x=dataset["alcohol"]).set(title = 'Kaining boxplot of alcohol')

# Data preprocessing
# Doing heatmap to investigate the correlation
corr = dataset.corr()
plt.figure(figsize = (9,8))
sns.heatmap(corr, cmap = 'BuPu', annot = True).set(title ='Kaining heatmap of features')
plt.show()

# standalize data and set training set and test set 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = dataset.iloc[:, :-1]
Y = dataset['quality']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 30, test_size = 0.30)
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)

# Part 3
# Classification
# Decision Tree Classifier
myClassifier = DecisionTreeClassifier(random_state=30)
myClassifier.fit(X_train, y_train)
prediction = myClassifier.predict(X_test)
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))

# Crossvalidation score of Decision Tree
from sklearn.model_selection import KFold, cross_val_score
myScores=cross_val_score(DecisionTreeClassifier(), X,y, scoring='accuracy',cv=KFold(n_splits=10))
print(myScores)

# Random Forrest Classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X,y)
print(rf.predict(X))
prediction = rf.predict(X_test)
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))

# Crossvalidation Score of Random Forest
myScores=cross_val_score(RandomForestClassifier(), X,y, scoring='accuracy',cv=KFold(n_splits=10))
print(myScores)

# Kmeans classifier
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=12, max_iter=300)
kmeans.fit(X,y)
predictions=kmeans.predict(X,y)
prediction = kmeans.predict(X_test)
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))

# Crossvalidation Score of Kmeans
myScores=cross_val_score(KMeans(), X,y, scoring='accuracy',cv=KFold(n_splits=10))
print(myScores)

# Bagging Classifier
from sklearn.ensemble import BaggingClassifier
model_bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=200) 
model_bag.fit(X_train, y_train)
score = model_bag.score(X_test, y_test)
prediction = model_bag.predict(X_test)
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))

# Crossvalidation Score of Bagging
myScores=cross_val_score(model_bag(), X,y, scoring='accuracy',cv=KFold(n_splits=10))
print(myScores)

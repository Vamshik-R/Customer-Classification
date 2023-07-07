#!/usr/bin/env python
# coding: utf-8

# # Importing required libraries

# In[2]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pickle
import pandas as pd
import seaborn as sns


# # Reading data from dataset

# In[3]:


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values #all the numerical features 
y = dataset.iloc[:, -1].values #label (or target)


# In[4]:


#dataset information
dataset.info()


# In[5]:


dataset.head()


# In[77]:


dataset.tail()


# # Data Preprosessing

# In[6]:


print(dataset.describe())
#Missing Data Check
print(dataset.isnull().sum())


# This dataset does not have any missing data

# In[7]:


#Checking for duplicated values
dataset.duplicated().sum()


# There are no duplicate values

# In[8]:


# Dropping Unecessary columns
dataset.drop('User ID',axis=1,inplace=True)
dataset.head()


# In[9]:


#Converting Categorical data to numeric data
dataset['Gender'].replace(to_replace=['Male', 'Female'], value=[1,2], inplace=True)


# Correlation between the features/variables are

# In[28]:


cor = dataset.corr()
cor


# In[10]:


dataset['Purchased'].value_counts()


# In[11]:


dataset.Age.describe()


# Average age of the considered population is around 37

# In[12]:


dataset.EstimatedSalary.describe()


# Average Salary of the population is  around 70000

# Checking if there is any relation between Age and Estimated salary

# In[88]:


plt.scatter(dataset.Age, dataset.EstimatedSalary)


# We do not observe any major dependencies between Age and Salaries

# In[13]:


dataset.head()


# # Training

# In[14]:


#Determining the columns that drive the decision of the "To purchase or not"
feature_cols = ['Gender','Age','EstimatedSalary']
X = dataset[feature_cols]
Y = dataset['Purchased']


# In[15]:


#splitting dataset into training and testing sets(75% training and 25% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_test


# In[16]:


X_train


# In[36]:


y_train


# In[17]:


X_test


# # Decision Tree Classifier

# In[18]:


from sklearn import tree

tr = tree.DecisionTreeClassifier()
tr.fit(X_train, y_train)
y_pred = tr.predict(X_test)
y_pred


# Feature Scaling is required to normalize the data from within a specified minimum and maximum range

# This step is needed to normalize the range of independent vavriables/features from a minimum to a maximum range

# Training the model and implementing the Test set

# In[19]:


from sklearn import model_selection
kfold = model_selection.KFold(n_splits = 10)
tr = tree.DecisionTreeClassifier()
tr.fit(X_train, y_train)
results = model_selection.cross_val_score(tr, X_train, y_train, cv = kfold)
results


# In[20]:


tr_train_score = tr.score(X_train, y_train)

tr_test_score= tr.score(X_test, y_test)


print('Decision Tree Classifier Train Score is : ' , tr_train_score)

print('Decision Tree Classifier Test Score is : ' , tr_test_score)


# # Accuracy of the Classifier 

# In[23]:


from sklearn.metrics import accuracy_score

tr_acc = accuracy_score(y_test, y_pred)
print('accuracy_score : ', tr_acc)


# In[25]:


from sklearn import  metrics

tr_acc = 100*tr.score(X_test, y_test)
print('Decision Tree Classifier Predictions: \n', tr.predict(X_test), '\n Accuracy:', tr_acc, '%')


# # Confusion Matrix

# In[26]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[27]:


from sklearn.metrics import confusion_matrix
cm_matrix = pd.DataFrame(data=cm, columns=['Actual_Buy', 'Actual_Buy'], 
                                 index=['Predict_Buy', 'Predict_Buy'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # Verification

# In[28]:


cross_check = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})
cross_check


# In[29]:


from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn import datasets,tree
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
#Prepare the data
x = iris.data
y = iris.target
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(x, y)


# # Plotting the decision tree

# In[30]:


fn=['Gender', 'Age', 'EstimatedSalary']      
cn=['Purchased', 'Not Purchased']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=100)
tree.plot_tree(tr,
               feature_names = fn, 
               class_names=cn,
               fontsize=6,
               max_depth = 5,
               filled = True);


# # KNN Model

# In[31]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled  = sc.transform(X_test)


# In[32]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train_scaled, y_train)


# In[59]:


y_pred = classifier.predict(X_test_scaled)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[60]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # Confusion Matrix

# In[33]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[34]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual_Buy', 'Actual_Buy'], 
                                 index=['Predict_Buy', 'Predict_Buy'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # Plotting for K values ranging from 1 to 30
# 

# In[63]:


testAccuracy = []
trainAccuracy = []
for k in range(1,30):
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,y_train)
    trainAccuracy.append(model.score(X_train,y_train))
    testAccuracy.append(model.score(X_test,y_test))


# In[64]:


from matplotlib import pyplot as plt,style
style.use('ggplot')


# In[69]:


#create a plot using the information from the above loop
plt.figure(figsize=(12,6))
plt.plot(range(1,30),trainAccuracy,label="Train Score",marker="o",markerfacecolor="teal",color="blue",linestyle="dashed")
plt.plot(range(1,30),testAccuracy,label="Test Score",marker="o",markerfacecolor="red",color="black",linestyle="dashed")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Score")
plt.title("Nbd Vs Score")
plt.show()


# The optimal value hence obtained is 4

# Creating a model using K = 4

# Print accuracy of model with K = 4

# In[71]:


print(accuracy_score(y_test,y_pred))


# In[72]:


from sklearn.pipeline import Pipeline
model_steps_20=[('sipStanderise',StandardScaler()),('shipModel',KNeighborsClassifier(n_neighbors=8,metric='minkowski',p=2))]
pipelineModel=Pipeline(steps=model_steps_20)
pipelineModel.fit(X_train,y_train)
print("score is:"+ str(pipelineModel.score(X_train,y_train)))
print("********************************")
pipelineModel.score(X_test,y_test)
predic_test_y=pipelineModel.predict(X_train)
print(pd.crosstab(y_train,predic_test_y))


# Printing classification Report
# 

# In[73]:


print(classification_report(y_test,y_pred))


# In[ ]:





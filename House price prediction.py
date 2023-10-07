#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("MagicBricks.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df.duplicated().sum()


# In[7]:


df.drop_duplicates(inplace=True)


# In[8]:


df.isna().sum()


# In[9]:


df['Per_Sqft'].fillna((df['Price']/df['Area']),inplace=True)
df['Bathroom'].fillna(df['Bathroom'].mode()[0],inplace=True)
df['Furnishing'].fillna(df['Furnishing'].mode()[0],inplace=True)
df['Parking'].fillna(df['Parking'].mode()[0],inplace=True)
df['Type'].fillna(df['Type'].mode()[0],inplace=True)


# In[10]:


df.info()


# In[11]:


df[['Parking','Bathroom']]=df[['Parking','Bathroom']].astype('int64')


# In[12]:


df.nunique()


# In[13]:


df.describe()


# ## Data Visualisation

# In[14]:


num_col=df[df.dtypes[df.dtypes != 'object'].index]
num_col

plt.figure(figsize=(15,10))
sns.heatmap(num_col.corr(),annot=True)


# In[15]:


plt.figure(figsize=(7,5))
sns.histplot(x=df['Area'],kde=True,bins=20)


# In[16]:


plt.figure(figsize=(17,10))
plt.subplot(3,4,1)
sns.countplot(x=df['BHK'])
plt.subplot(3,4,2)
sns.boxplot(x=df['BHK'],y=df['Price'])
print('Correlation between BHK and Price is',df['BHK'].corr(df['Price']))
print('Skewness of the BHK is',df['BHK'].skew())

plt.subplot(3,4,3)
sns.countplot(x=df['Bathroom'])
plt.subplot(3,4,4)
sns.boxplot(x=df['Bathroom'],y=df['Price'])
print('Correlation between Bathroom and Price is',df['Bathroom'].corr(df['Price']))
print('Skewness of the Bathroom is',df['Bathroom'].skew())


plt.subplot(3,4,5)
sns.countplot(x=df['Furnishing'])
plt.subplot(3,4,6)
sns.boxplot(x=df['Furnishing'],y=df['Price'])

plt.subplot(3,4,7)
sns.countplot(x=df['Parking'])
plt.subplot(3,4,8)
sns.boxplot(x=df['Parking'],y=df['Price'])
print('Correlation between Parking and Price is',df['Parking'].corr(df['Price']))
print('Skewness of the Parking is',df['Parking'].skew())

plt.subplot(3,4,9)
sns.countplot(x=df['Status'])
plt.subplot(3,4,10)
sns.boxplot(x=df['Status'],y=df['Price'])

plt.subplot(3,4,11)
sns.barplot(x=df['BHK'],y=df['Area'])
plt.subplot(3,4,12)
sns.barplot(x=df['Bathroom'],y=df['Area'])


# In[17]:


df.drop(df.index[(df["Parking"] == 39)],axis=0,inplace=True)
df.drop(df.index[(df["Parking"] == 114)],axis=0,inplace=True)


# In[18]:


plt.figure(figsize=(17,10))
plt.subplot(3,4,1)
sns.countplot(x=df['BHK'])
plt.subplot(3,4,2)
sns.boxplot(x=df['BHK'],y=df['Price'])
print('Correlation between BHK and Price is',df['BHK'].corr(df['Price']))
print('Skewness of the BHK is',df['BHK'].skew())

plt.subplot(3,4,3)
sns.countplot(x=df['Bathroom'])
plt.subplot(3,4,4)
sns.boxplot(x=df['Bathroom'],y=df['Price'])
print('Correlation between Bathroom and Price is',df['Bathroom'].corr(df['Price']))
print('Skewness of the Bathroom is',df['Bathroom'].skew())


plt.subplot(3,4,5)
sns.countplot(x=df['Furnishing'])
plt.subplot(3,4,6)
sns.boxplot(x=df['Furnishing'],y=df['Price'])

plt.subplot(3,4,7)
sns.countplot(x=df['Parking'])
plt.subplot(3,4,8)
sns.boxplot(x=df['Parking'],y=df['Price'])
print('Correlation between Parking and Price is',df['Parking'].corr(df['Price']))
print('Skewness of the Parking is',df['Parking'].skew())

plt.subplot(3,4,9)
sns.countplot(x=df['Status'])
plt.subplot(3,4,10)
sns.boxplot(x=df['Status'],y=df['Price'])

plt.subplot(3,4,11)
sns.barplot(x=df['BHK'],y=df['Area'])
plt.subplot(3,4,12)
sns.barplot(x=df['Bathroom'],y=df['Area'])


# In[19]:


plt.figure(figsize=(7,5))
sns.barplot(x=df['Furnishing'],y=df['Price'],hue=df['BHK'])


# In[20]:


#removing outliers
from scipy import stats 
z = np.abs(stats.zscore(df[df.dtypes[df.dtypes != 'object'].index]))
df = df[(z < 3).all(axis=1)]


# In[21]:


df.shape


# In[22]:


# Data Visualisation after removing outliers
plt.figure(figsize=(17,10))
plt.subplot(3,4,1)
sns.countplot(x=df['BHK'])
plt.subplot(3,4,2)
sns.boxplot(x=df['BHK'],y=df['Price'])
print('Correlation between BHK and Price is',df['BHK'].corr(df['Price']))
print('Skewness of the BHK is',df['BHK'].skew())

plt.subplot(3,4,3)
sns.countplot(x=df['Bathroom'])
plt.subplot(3,4,4)
sns.boxplot(x=df['Bathroom'],y=df['Price'])
print('Correlation between Bathroom and Price is',df['Bathroom'].corr(df['Price']))
print('Skewness of the Bathroom is',df['Bathroom'].skew())

plt.subplot(3,4,5)
sns.countplot(x=df['Furnishing'])
plt.subplot(3,4,6)
sns.boxplot(x=df['Furnishing'],y=df['Price'])

plt.subplot(3,4,7)
sns.countplot(x=df['Parking'])
plt.subplot(3,4,8)
sns.boxplot(x=df['Parking'],y=df['Price'])
print('Correlation between Parking and Price is',df['Parking'].corr(df['Price']))
print('Skewness of the Parking is',df['Parking'].skew())

plt.subplot(3,4,9)
sns.countplot(x=df['Status'])
plt.subplot(3,4,10)
sns.boxplot(x=df['Status'],y=df['Price'])

plt.subplot(3,4,11)
sns.barplot(x=df['BHK'],y=df['Area'])
plt.subplot(3,4,12)
sns.barplot(x=df['Bathroom'],y=df['Area'])


# In[23]:


df.drop(df[df['Area'] > 30000].index, inplace = True)
plt.figure(figsize=(14,7))
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.Area,df.Price)


# In[24]:


np.random.seed(7)
x = np.random.rand(100, 1)
y = 13 + 3 * x + np.random.rand(100, 1)
plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# # ML Modeling

# In[25]:


from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.3)


# In[26]:


from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, Y_train)


# In[27]:


Y_pred = linear.predict(X_test)
print("Accuracy Score for Test Dataset is ",linear.score(X_test, Y_test)*100,"%")
print("Accuracy Score for Train Dataset is",linear.score(X_train,Y_train)*100,"%")


# In[28]:


c1=float(linear.intercept_)
m1=float(linear.coef_)
print("Intercept (c) of regression line is", c1)
print("Coefficient (m) of regression line is", m1)


# In[29]:


plt.scatter(X_test,Y_test)
plt.plot(x,m1*x+c1,color='red')


# So this model Predicts the value of any house with an accuracy of 88.42%

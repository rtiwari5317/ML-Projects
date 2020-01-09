
# coding: utf-8

# In[1]:


#Importing necessary Libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


# In[2]:


# Reading the Data Set
df = pd.read_csv('C:/Users/Rahul_Tiwari4/Downloads/ML Projects/DL-and-ML-Practical-Tutorials-Package/DL and ML Practical Tutorials - Package/Project 1/Car_Purchasing_Data.csv', encoding='ISO-8859-1')
df.head(10) # Top 10 rows for the Data Set.


# In[3]:


#Statistical Summary for the Car Purchase Data Set
df.describe()


# In[4]:


df.isnull().sum() # checking for missing values for any columns


# In[5]:


#Visualizing the Data Set
sns.pairplot(df)


# In[6]:


# Dropping the Unnecessary Columns for the Independent Variables and storing data into 'X' variable.
X = df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
X.shape


# In[7]:


# Taking Output Variable into 'y' Variable.
y = df['Car Purchase Amount']
y.shape


# In[8]:


#Feature Engineering / Scaling the data
from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)


# In[9]:


# Reshaping the Data so that it can be Scaled accordingly
y = y.values.reshape(-1,1)
y.shape


# In[10]:


scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)


# In[14]:


#Splitting the Data Set into Train & Test Data Sets.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,test_size= 0.25)


# In[13]:


# Importing necessary libraries for using Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

X_test_sample = np.array([[0, 0.4370344,  0.53515116, 0.57836085, 0.22342985]])
y_predict_sample = lr.predict(X_test_sample)

print('Expected Scaled Purchase Amount=', y_predict_sample)

y_predict_sample_orig = scaler_y.inverse_transform(y_predict_sample)
# Above statement will provide the expected price given by customer for Car Purchase.
print('Expected Purchase Amount=', y_predict_sample_orig)

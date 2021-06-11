#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns


# In[3]:


Train = pd.read_csv("C:/Users/vinay/Downloads/SalaryData_Train(1).csv")
Test = pd.read_csv("C:/Users/vinay/Downloads/SalaryData_Test(1).csv")
string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]


# In[4]:


##Preprocessing the data. As, there are categorical variables
from sklearn.preprocessing import LabelEncoder


# In[5]:


number = LabelEncoder()
for i in string_columns:
        Train[i]= number.fit_transform(Train[i])
        Test[i]=number.fit_transform(Test[i])


# In[6]:


##Capturing the column names which can help in futher process
colnames = Train.columns
colnames
len(colnames)


# In[7]:


x_train = Train[colnames[0:13]]
y_train = Train[colnames[13]]
x_test = Test[colnames[0:13]]
y_test = Test[colnames[13]]


# In[8]:


##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
x_train = norm_func(x_train)
x_test =  norm_func(x_test)


# In[9]:


from sklearn.svm import SVC
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)


# In[10]:


np.mean(pred_test_linear==y_test) # Accuracy = 81%


# In[11]:


# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)


# In[12]:


np.mean(pred_test_poly==y_test) # Accuracy = 84%


# In[13]:


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)


# In[14]:


np.mean(pred_test_rbf==y_test) # Accuracy = 84%


# In[15]:


#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)


# In[16]:


np.mean(pred_test_sig==y_test) #Accuracy = 84%


# In[ ]:





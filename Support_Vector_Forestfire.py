#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np 
import seaborn as sns


# In[6]:


forestfires = pd.read_csv("C:/Users/vinay/Downloads/forestfires.csv")


# In[9]:


data = forestfires.describe();data


# In[10]:


##Dropping the month and day columns
forestfires.drop(["month","day"],axis=1,inplace =True)


# In[12]:


##Normalising the data as there is scale difference
predictors = forestfires.iloc[:,0:28]
target = forestfires.iloc[:,28]
target


# In[13]:


def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)


# In[14]:


fires = norm_func(predictors)
fires


# In[15]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)


# In[17]:


model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)


# In[18]:


np.mean(pred_test_linear==y_test) # Accuracy = 96%


# In[19]:


# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)


# In[20]:


np.mean(pred_test_poly==y_test) #Accuacy = 79%


# In[21]:


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)


# In[23]:


np.mean(pred_test_rbf==y_test) #Accuracy = 76.9%


# In[24]:


#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)


# In[25]:


np.mean(pred_test_sig==y_test) #Accuracy = 76%


# In[ ]:





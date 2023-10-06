#!/usr/bin/env python
# coding: utf-8

# ### Bank Note Authentication

# __Data was obtained from images captured from both genuine and counterfeit banknote-like specimens. To digitize these specimens, we employed an industrial camera typically employed for print inspection. The resulting images are 400x400 pixels in size. These high-quality grayscale images, acquired at a resolution of approximately 660 dots per inch (dpi), were then subjected to feature extraction using the Wavelet Transform technique.__

# In[50]:


# models to read, view, clean and process the data
import pandas as pd
import numpy as np

# To visualize data as graphs and charts
import matplotlib.pyplot as plt
import seaborn as sns

# for splitting data and selecting suitable model with highest accuracy and appropriate hyperparameters
from sklearn.model_selection import train_test_split # Splitting data

# to import models
from sklearn.ensemble import RandomForestClassifier


# To evaluate model accuracy
from sklearn.metrics import accuracy_score, confusion_matrix


# In[51]:


bank_df=pd.read_csv('BankNote_Authentication.csv')


# In[52]:


bank_df


# In[53]:


bank_df.isnull().sum()


# In[54]:


bank_df['class'].value_counts()


# # Correlation among the features

# In[56]:


plt.figure(figsize=(10,5))
sns.heatmap(bank_df.corr(), annot=True, fmt='0.1g')


# # Independent and Dependent features

# In[11]:


x=bank_df.iloc[:,:-1]
y=bank_df.iloc[:,-1]


# In[12]:


print("X", x.head(10))
print('------------------------------------')
print("Y", y.head(10))


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)  #splitting of data for training and testing


# # Model development

# In[15]:


rfc_model=RandomForestClassifier()
rfc_model.fit(X_train,y_train)


# In[35]:


prediction =rfc_model.predict(X_test)
prediction


# In[24]:


acc_score=accuracy_score(y_test,prediction)
print('Accuracy of RFC model is :',acc_score )

cm= confusion_matrix(y_test,prediction)
sns.heatmap(cm, annot=True)


# In[57]:


### Create a Pickle file using serialization 
# import pickle
# pickle_out = open("rfcmodel.pkl","wb")
# pickle.dump(rfc_model, pickle_out)
# pickle_out.close()


# In[39]:


# result = rfc_model.predict([[0,1,0,1]])
# print ("Note authenticity :", result)







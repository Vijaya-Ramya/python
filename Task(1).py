#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[2]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head(25)


# In[16]:


plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.scatter(data.Hours,data.Scores, color='red' , marker='+') 
plt.show()


# In[17]:


reg = linear_model.LinearRegression()
reg.fit(data[['Hours']],data.Scores)


# In[37]:


data.Hours.plot(kind='hist')


# In[24]:


print(reg.intercept_)


# In[25]:


print(reg.coef_)


# In[27]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# In[29]:


line = reg.coef_*X+reg.intercept_
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[34]:


reg.predict([[9.25]])


# In[ ]:





# In[ ]:





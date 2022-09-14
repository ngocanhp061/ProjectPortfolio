#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction Project
# ### 1. Problem Definition:
# - Goal: predict the sales price for each house
# 
# ### 2. Feature Selection:
# - Choose features to train ML Model 
# - Need to use "Feature Engineering" to identify
# 
# ### 3. Splitting the datasets:
# - 'data': dataset
# - 'X': 'data[features]'
# - 'y': target variable 'SalePrice"
# 
# ### 4. Training Machine Learning Model

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np


# In[4]:


data = pd.read_csv("train.csv", index_col="Id")


# In[5]:


data.head()


# In[6]:


data.columns


# ## 2. Feature Selections

# In[7]:


features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]


# ## 3. Splitting dataset into X and y

# In[10]:


X = data[features]
y = data["SalePrice"]


# In[9]:


X


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=0)


# In[13]:


X_train


# ### 4. Training Machine Learning Model

# In[14]:


from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=1)


# In[16]:


#Fit training data into model
dt_model.fit(X_train, y_train)


# In[18]:


#y predict
y_preds = dt_model.predict(X_valid.head())


# In[19]:


y_preds


# In[20]:


pd.DataFrame({'y':y_valid.head(), 'y_preds':y_preds})


# In[22]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train, y_train)


# In[23]:


rf_val_preds = rf_model.predict(X_valid)


# In[24]:


rf_val_preds[:5]


# ### Predict with a new input (test it) 
# 

# In[26]:


X_valid.head()


# In[27]:


rf_model.predict([[6969, 2021, 1000, 800, 4, 5, 8]])


# ### 5. Model Evaluation
